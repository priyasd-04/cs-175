# Author(s): Amish Kunal
"""Evaluate perplexity of different text sources under a Shakespeare-finetuned LM.

Fine-tunes a TransformerMonkey on Shakespeare (if no checkpoint provided),
then computes per-character perplexity for each text source.

Lower perplexity = more Shakespeare-like.

Run:
  python3 expirements/eval_perplexity.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path
from statistics import mean

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifier.transformer_judge import TransformerJudge
from src.monkeys.bigram import BigramModel
from src.monkeys.generator import random_monkey
from src.monkeys.TransformerMonkey import CharTokenizer, EOS_TOKEN, TransformerMonkey
from src.utils.load_datasets import load_old_english_dataset, load_shakespeare_dataset


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_base_model(ckpt_path: Path, device: str) -> tuple[TransformerMonkey, CharTokenizer, dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    tokenizer = ckpt["tokenizer"]
    cfg = ckpt["base_model_config"]

    model = TransformerMonkey(
        tokenizer.vocab_size,
        tokenizer=tokenizer,
        block_size=int(cfg["block_size"]),
        n_embd=int(cfg["n_embd"]),
        n_head=int(cfg["n_head"]),
        n_layer=int(cfg["n_layer"]),
        dropout=float(cfg.get("dropout", 0.2)),
    ).to(device)

    judge = TransformerJudge(model).to(device)
    judge.load_state_dict(ckpt["judge_state_dict"])
    return copy.deepcopy(judge.transformer), tokenizer, cfg


def _encode_safe(tokenizer: CharTokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(text)
    except KeyError:
        return []


def _finetune_on_shakespeare(
    model: TransformerMonkey,
    tokenizer: CharTokenizer,
    *,
    sp_text: str,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
) -> None:
    data = torch.tensor(_encode_safe(tokenizer, sp_text), dtype=torch.long, device=device)
    block = int(model.block_size)
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for step in range(steps):
        ix = torch.randint(len(data) - block - 1, (batch_size,), device=device)
        x = torch.stack([data[i : i + block] for i in ix])
        y = torch.stack([data[i + 1 : i + block + 1] for i in ix])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"  finetune step {step:5d} | loss {loss.item():.4f}")
    model.eval()


@torch.no_grad()
def _compute_perplexity(
    model: TransformerMonkey,
    tokenizer: CharTokenizer,
    text: str,
    *,
    device: str,
) -> float | None:
    ids = _encode_safe(tokenizer, text)
    if len(ids) < 2:
        return None

    block = int(model.block_size)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_tokens = 0

    for start in range(0, len(ids) - 1, block):
        end = min(start + block, len(ids) - 1)
        x = torch.tensor([ids[start:end]], dtype=torch.long, device=device)
        y = torch.tensor([ids[start + 1 : end + 1]], dtype=torch.long, device=device)

        logits, _ = model(x)
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)

        loss = loss_fn(logits, y)
        chunk_len = y.size(0)
        total_loss += loss.item() * chunk_len
        n_tokens += chunk_len

    if n_tokens == 0:
        return None

    avg_loss = total_loss / n_tokens
    return math.exp(avg_loss)


@torch.no_grad()
def _gen_transformer_texts(
    model: TransformerMonkey,
    tokenizer: CharTokenizer,
    *,
    n: int,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    out: list[str] = []
    for _ in range(n):
        ids = _encode_safe(tokenizer, prompt)
        if not ids:
            ids = [0]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8)
        out.append(tokenizer.decode(y[0].tolist()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-ckpt", type=str, default="models/transformer_judge.pt")
    ap.add_argument("--ft-steps", type=int, default=2000)
    ap.add_argument("--ft-batch", type=int, default=32)
    ap.add_argument("--ft-lr", type=float, default=1e-3)
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--sample-len", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-csv", type=str, default="results/perplexity_by_source.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = _device()
    n = int(args.n_samples)

    ckpt_path = (REPO_ROOT / args.judge_ckpt).resolve()
    print(f"Loading base model from {ckpt_path}")
    model, tokenizer, cfg = _load_base_model(ckpt_path, device)

    print("Loading datasets...")
    sp = list(load_shakespeare_dataset(0.0, 0.0, seed=args.seed))
    oe = list(load_old_english_dataset(0.0, 0.0, seed=args.seed))

    sp_text = EOS_TOKEN.join(str(x) for x in sp)

    print(f"Fine-tuning on Shakespeare ({args.ft_steps} steps)...")
    _finetune_on_shakespeare(
        model, tokenizer,
        sp_text=sp_text,
        device=device,
        steps=args.ft_steps,
        batch_size=args.ft_batch,
        lr=args.ft_lr,
    )

    real_sp = [str(x) for x in sp[:n]]
    real_oe = [str(x) for x in oe[:n]]

    print("Generating baseline texts...")
    bigram = BigramModel(smoothing=0.5).fit([str(x) for x in sp[:2000]])
    bigram_texts = [bigram.generate(args.sample_len, seed=args.seed + i) for i in range(n)]
    random_texts = [random_monkey(args.sample_len) for _ in range(n)]

    print("Generating transformer texts...")
    base_for_gen, _, _ = _load_base_model(ckpt_path, device)
    transformer_texts = _gen_transformer_texts(
        base_for_gen, tokenizer,
        n=n, prompt="To be, or not to be ", max_new_tokens=200, device=device,
    )

    sources: dict[str, list[str]] = {
        "real_sp": real_sp,
        "real_oe": real_oe,
        "random": random_texts,
        "bigram": bigram_texts,
        "transformer": transformer_texts,
    }

    print("\nComputing perplexity per source...")
    results: list[dict[str, object]] = []
    for src, texts in sources.items():
        ppls: list[float] = []
        for t in texts:
            ppl = _compute_perplexity(model, tokenizer, t, device=device)
            if ppl is not None:
                ppls.append(ppl)
        avg_ppl = mean(ppls) if ppls else float("nan")
        min_ppl = min(ppls) if ppls else float("nan")
        max_ppl = max(ppls) if ppls else float("nan")
        results.append({
            "source": src,
            "n": len(ppls),
            "mean_ppl": round(avg_ppl, 2),
            "min_ppl": round(min_ppl, 2),
            "max_ppl": round(max_ppl, 2),
        })

    print(f"\n{'source':<15} {'n':>4} {'mean_ppl':>10} {'min_ppl':>10} {'max_ppl':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['source']:<15} {r['n']:>4} {r['mean_ppl']:>10} {r['min_ppl']:>10} {r['max_ppl']:>10}")

    out_path = (REPO_ROOT / args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "n", "mean_ppl", "min_ppl", "max_ppl"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
