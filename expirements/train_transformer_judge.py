"""Train and save a Transformer-based Shakespeare judge.

Binary classifier head on top of a frozen TransformerMonkey.

Run:
  python3 expirements/train_transformer_judge.py --negatives oe+random
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from classifier.transformer_judge import TransformerJudge
from monkeys.TransformerMonkey import CharTokenizer, TransformerMonkey, EOS_TOKEN
from monkeys.bigram import BigramModel
from utils.load_datasets import load_old_english_dataset, load_shakespeare_dataset


@dataclass(frozen=True)
class BaseModelConfig:
    block_size: int = 128
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 3
    dropout: float = 0.2


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _join_lines(lines: Sequence[str], max_lines: int | None = None) -> str:
    if max_lines is not None:
        lines = lines[: max_lines]
    return EOS_TOKEN.join(str(x) for x in lines)


def _encode_clip(tokenizer: CharTokenizer, text: str, block_size: int) -> list[int]:
    ids = tokenizer.encode(text)
    if not ids:
        return [0]
    if len(ids) > block_size:
        ids = ids[-block_size:]
    return ids


def _can_encode(tokenizer: CharTokenizer, text: str) -> bool:
    try:
        tokenizer.encode(text)
        return True
    except KeyError:
        return False


def _random_texts_from_tokenizer(
    tokenizer: CharTokenizer,
    *,
    n: int,
    length: int,
    seed: int,
) -> list[str]:
    """Random strings built from tokenizer characters."""
    chars = [c for c in tokenizer.chars if c != EOS_TOKEN]
    if not chars:
        return [""] * n

    g = torch.Generator().manual_seed(seed)
    idxs = torch.randint(0, len(chars), (n, length), generator=g)
    return ["".join(chars[j] for j in row.tolist()) for row in idxs]


def _generate_bigram_negatives(
    shakespeare_lines: Sequence[str],
    *,
    n: int,
    length: int,
    seed: int,
) -> list[str]:
    """Bigram samples trained on Shakespeare lines (Shakespeare-ish negatives)."""
    model = BigramModel(smoothing=0.5).fit(shakespeare_lines)
    return [model.generate(length, seed=seed + i) for i in range(n)]


def _generate_transformer_negatives(
    model: TransformerMonkey,
    tokenizer: CharTokenizer,
    *,
    n: int,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    device: str,
) -> list[str]:
    """Samples from the base transformer model (monkey negatives)."""
    model.eval()
    out: list[str] = []
    with torch.no_grad():
        for i in range(n):
            # Add a little randomness by sampling from a shifted prompt slice.
            p = prompt
            if len(p) > 10:
                start = (seed + i) % min(10, len(p) - 1)
                p = prompt[start:]
            x = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
            y = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8)
            out.append(tokenizer.decode(y[0].tolist()))
    return out


class TextLabelDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], *, tokenizer: CharTokenizer, block_size: int):
        assert len(texts) == len(labels)
        self.texts = list(texts)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.block_size = int(block_size)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        ids = _encode_clip(self.tokenizer, text, self.block_size)
        return ids, len(ids), self.labels[idx]


def collate_batch(batch, *, pad_id: int = 0):
    """Pad a batch of variable-length token-id lists."""
    ids_list, lengths, labels = zip(*batch)
    max_len = max(lengths)
    x = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    y = torch.stack(labels).view(-1, 1)
    return x, lengths_t, y


def _format_counts(label: str, items: Sequence[str]) -> str:
    counts = Counter(items)
    parts = [f"{k}={counts[k]}" for k in sorted(counts)]
    return f"{label}: " + (", ".join(parts) if parts else "none")


@torch.no_grad()
def evaluate(
    judge: TransformerJudge,
    loader: DataLoader,
    *,
    device: str,
    pad_id: int = 0,
) -> dict[str, float]:
    judge.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    loss_fn = nn.BCELoss()

    for x, lengths, y in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        p = judge(x, lengths=lengths, pad_id=pad_id)
        loss = loss_fn(p, y).item()
        losses.append(loss)

        y_true.extend(y.view(-1).int().tolist())
        y_pred.extend((p.view(-1) >= 0.5).int().tolist())

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-9, (prec + rec))

    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def pretrain_base_lm(
    model: TransformerMonkey,
    tokenizer: CharTokenizer,
    *,
    train_text: str,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
) -> None:
    """Quick LM pretrain so the base model isn't random."""
    data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long, device=device)
    block_size = int(model.block_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for step in range(steps):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), device=device)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"base_lm step {step:5d} | loss {loss.item():.4f}")

    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-lines", type=int, default=None, help="Optional cap on dataset lines for faster runs")

    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--base-pretrain-steps", type=int, default=1000)
    ap.add_argument("--base-pretrain-batch", type=int, default=32)
    ap.add_argument("--base-pretrain-lr", type=float, default=1e-3)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)

    ap.add_argument(
        "--negatives",
        choices=(
            "oe",
            "random",
            "oe+random",
            "oe+random+bigram",
            "oe+random+transformer",
            "oe+random+bigram+transformer",
        ),
        default="oe+random",
        help="What to use as negative examples (non-Shakespeare)",
    )
    ap.add_argument(
        "--n-random-neg",
        type=int,
        default=None,
        help="How many random-negative samples to add (default: same as #Shakespeare lines used)",
    )
    ap.add_argument("--random-len", type=int, default=200, help="Length of each random-negative string")
    ap.add_argument(
        "--n-hard-neg",
        type=int,
        default=None,
        help="How many hard-negative samples to add per source (default: same as #positives)",
    )
    ap.add_argument("--hard-len", type=int, default=300, help="Length for bigram hard negatives")
    ap.add_argument("--hard-prompt", type=str, default="To be, or not to be ", help="Prompt for transformer hard negatives")
    ap.add_argument("--hard-max-new", type=int, default=200, help="Max new tokens for transformer hard negatives")

    ap.add_argument("--out", type=str, default="models/transformer_judge.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.val_frac < 0 or args.test_frac < 0 or (args.val_frac + args.test_frac) >= 1.0:
        raise ValueError("Need val_frac >= 0, test_frac >= 0, and val_frac + test_frac < 1.")

    device = _device()
    print(f"Device: {device}")

    print("Loading datasets...")
    oe_lines = list(load_old_english_dataset(0.0, 0.0, seed=args.seed))
    sp_lines = list(load_shakespeare_dataset(0.0, 0.0, seed=args.seed))

    if args.max_lines is not None:
        oe_lines = oe_lines[: args.max_lines]
        sp_lines = sp_lines[: args.max_lines]

    print(f"Old English lines: {len(oe_lines)} | Shakespeare lines: {len(sp_lines)}")

    n_pos = len(sp_lines)
    n_oe = min(len(oe_lines), n_pos)
    gen = torch.Generator().manual_seed(args.seed)
    oe_idx = torch.randperm(len(oe_lines), generator=gen).tolist()[:n_oe]
    sp_idx = torch.randperm(len(sp_lines), generator=gen).tolist()[:n_pos]
    oe_lines = [oe_lines[i] for i in oe_idx]
    sp_lines = [sp_lines[i] for i in sp_idx]
    print(f"Using positives={len(sp_lines)} | oe_negatives={len(oe_lines)}")

    print("Building tokenizer (union of OE + Shakespeare chars)...")
    tokenizer = CharTokenizer(_join_lines(oe_lines + sp_lines))

    random_negs: list[str] = []
    if "random" in args.negatives:
        n_rand = int(args.n_random_neg) if args.n_random_neg is not None else len(sp_lines)
        random_negs = _random_texts_from_tokenizer(
            tokenizer, n=n_rand, length=int(args.random_len), seed=args.seed + 999
        )
        print(f"Added random negatives: {len(random_negs)} (len={args.random_len})")

    base_cfg = BaseModelConfig(
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )

    base_model = TransformerMonkey(
        tokenizer.vocab_size,
        tokenizer=tokenizer,
        block_size=base_cfg.block_size,
        n_embd=base_cfg.n_embd,
        n_head=base_cfg.n_head,
        n_layer=base_cfg.n_layer,
        dropout=base_cfg.dropout,
    ).to(device)

    print("Pretraining base LM on Old English (quick)...")
    pretrain_base_lm(
        base_model,
        tokenizer,
        train_text=_join_lines(oe_lines),
        device=device,
        steps=args.base_pretrain_steps,
        batch_size=args.base_pretrain_batch,
        lr=args.base_pretrain_lr,
    )

    print("Preparing judge dataset...")
    neg_texts: list[str] = []
    neg_sources: list[str] = []
    if args.negatives == "oe":
        neg_texts = list(oe_lines)
        neg_sources = ["old_english"] * len(oe_lines)
    elif args.negatives == "random":
        neg_texts = list(random_negs)
        neg_sources = ["random"] * len(random_negs)
    else:
        neg_texts = list(oe_lines) + list(random_negs)
        neg_sources = (["old_english"] * len(oe_lines)) + (["random"] * len(random_negs))

    # Hard negatives: monkey outputs that look more Shakespeare-ish than OE/random.
    n_hard = int(args.n_hard_neg) if args.n_hard_neg is not None else len(sp_lines)
    if "bigram" in args.negatives:
        bigram_raw = _generate_bigram_negatives(sp_lines, n=n_hard, length=int(args.hard_len), seed=args.seed + 2026)
        bigram_ok = [t for t in bigram_raw if _can_encode(tokenizer, t)]
        neg_texts.extend(bigram_ok)
        neg_sources.extend(["bigram_hard"] * len(bigram_ok))
        print(f"Added bigram hard negatives: {len(bigram_ok)} (requested {n_hard})")
    if "transformer" in args.negatives:
        tr_raw = _generate_transformer_negatives(
            base_model,
            tokenizer,
            n=n_hard,
            prompt=str(args.hard_prompt),
            max_new_tokens=int(args.hard_max_new),
            seed=args.seed + 404,
            device=device,
        )
        tr_ok = [t for t in tr_raw if _can_encode(tokenizer, t)]
        neg_texts.extend(tr_ok)
        neg_sources.extend(["transformer_hard"] * len(tr_ok))
        print(f"Added transformer hard negatives: {len(tr_ok)} (requested {n_hard})")

    texts = list(sp_lines) + neg_texts
    labels = [1] * len(sp_lines) + [0] * len(neg_texts)
    sources = (["shakespeare"] * len(sp_lines)) + neg_sources
    print(_format_counts("Dataset composition before balancing", sources))

    g_bal = torch.Generator().manual_seed(args.seed + 1234)
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]
    n_bin = min(len(pos_idx), len(neg_idx))
    pos_keep = torch.randperm(len(pos_idx), generator=g_bal).tolist()[:n_bin]
    neg_keep = torch.randperm(len(neg_idx), generator=g_bal).tolist()[:n_bin]
    keep = [pos_idx[i] for i in pos_keep] + [neg_idx[i] for i in neg_keep]
    keep.sort()
    texts = [texts[i] for i in keep]
    labels = [labels[i] for i in keep]
    sources = [sources[i] for i in keep]
    print(f"Balanced binary dataset: pos={n_bin} neg={n_bin} (total={len(texts)})")
    print(_format_counts("Dataset composition after balancing", sources))

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(texts), generator=g).tolist()
    texts = [texts[i] for i in perm]
    labels = [labels[i] for i in perm]
    sources = [sources[i] for i in perm]

    n_total = len(texts)
    n_test = int(n_total * float(args.test_frac))
    n_val = int(n_total * float(args.val_frac))
    test_end = n_test
    val_end = n_test + n_val

    test_texts, val_texts, train_texts = texts[:test_end], texts[test_end:val_end], texts[val_end:]
    test_labels, val_labels, train_labels = labels[:test_end], labels[test_end:val_end], labels[val_end:]
    test_sources, val_sources, train_sources = sources[:test_end], sources[test_end:val_end], sources[val_end:]

    print(
        f"Split sizes | train={len(train_texts)} val={len(val_texts)} test={len(test_texts)} "
        f"(val_frac={args.val_frac:.2f}, test_frac={args.test_frac:.2f})"
    )
    print(_format_counts("Train composition", train_sources))
    print(_format_counts("Val composition", val_sources))
    print(_format_counts("Test composition", test_sources))

    train_ds = TextLabelDataset(train_texts, train_labels, tokenizer=tokenizer, block_size=base_cfg.block_size)
    val_ds = TextLabelDataset(val_texts, val_labels, tokenizer=tokenizer, block_size=base_cfg.block_size)
    test_ds = TextLabelDataset(test_texts, test_labels, tokenizer=tokenizer, block_size=base_cfg.block_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id=0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id=0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id=0),
    )

    judge = TransformerJudge(base_model).to(device)

    optimizer = torch.optim.Adam(judge.classifier.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()
    best_state = None
    best_val_f1 = float("-inf")

    print("Training judge head...")
    for epoch in range(args.epochs):
        judge.train()
        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            p = judge(x, lengths=lengths, pad_id=0)
            loss = loss_fn(p, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        metrics = evaluate(judge, val_loader, device=device, pad_id=0)
        print(
            f"epoch {epoch+1}/{args.epochs} | "
            f"val loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}"
        )
        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in judge.state_dict().items()}

    if best_state is not None:
        judge.load_state_dict(best_state)

    test_metrics = evaluate(judge, test_loader, device=device, pad_id=0)
    print(
        "\nFinal test metrics | "
        f"loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['accuracy']:.3f} "
        f"precision={test_metrics['precision']:.3f} "
        f"recall={test_metrics['recall']:.3f} "
        f"f1={test_metrics['f1']:.3f}"
    )

    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "base_model_config": asdict(base_cfg),
        "judge_state_dict": judge.state_dict(),
        "tokenizer": tokenizer,
        "pad_id": 0,
    }
    torch.save(ckpt, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

