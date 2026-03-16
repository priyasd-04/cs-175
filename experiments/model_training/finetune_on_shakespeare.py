# Author(s): Amish Kunal
"""Fine-tune the transformer monkey on Shakespeare.

Loads the base transformer + tokenizer from `models/transformer_judge.pt`,
then does a short language-model fine-tune on the tiny Shakespeare dataset.

Saves a new checkpoint (default: `models/oe_then_shakespeare.pt`) and prints
before/after generations + judge scores for a few prompts.

Run:
  python3 expirements/finetune_on_shakespeare.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import copy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifier.transformer_judge import TransformerJudge
from src.monkeys.TransformerMonkey import EOS_TOKEN, TransformerMonkey
from src.utils.load_datasets import load_shakespeare_dataset


def _load_judge_ckpt(path: str | Path, device: str):
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    tokenizer = ckpt["tokenizer"]
    cfg = ckpt["base_model_config"]

    base = TransformerMonkey(
        tokenizer.vocab_size,
        tokenizer=tokenizer,
        block_size=int(cfg["block_size"]),
        n_embd=int(cfg["n_embd"]),
        n_head=int(cfg["n_head"]),
        n_layer=int(cfg["n_layer"]),
        dropout=float(cfg.get("dropout", 0.2)),
    ).to(device)

    judge = TransformerJudge(base).to(device)
    judge.load_state_dict(ckpt["judge_state_dict"])
    judge.eval()

    # We'll fine-tune a separate copy of the generator so the judge stays fixed for scoring.
    model = copy.deepcopy(judge.transformer)
    model.eval()
    return model, tokenizer, cfg, judge


def _encode(tokenizer, text: str) -> list[int]:
    # skip texts with unseen characters
    try:
        ids = tokenizer.encode(text)
    except KeyError:
        return []
    return ids


@torch.no_grad()
def _judge_score_text(judge: TransformerJudge, tokenizer, text: str, device: str) -> float:
    ids = _encode(tokenizer, text)
    if not ids:
        return float("nan")
    block = int(judge.transformer.block_size)
    ids = ids[-block:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    return float(judge(x, lengths=lengths, pad_id=0).item())


@torch.no_grad()
def generate(model: TransformerMonkey, tokenizer, prompt: str, *, device: str, max_new_tokens: int) -> str:
    ids = _encode(tokenizer, prompt)
    if not ids:
        ids = [0]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8)
    return tokenizer.decode(y[0].tolist())


def finetune_lm(
    model: TransformerMonkey,
    tokenizer,
    *,
    train_text: str,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
    tune: str,
) -> None:
    data_ids = _encode(tokenizer, train_text)
    if not data_ids:
        raise RuntimeError("Failed to encode Shakespeare training text with current tokenizer.")
    data = torch.tensor(data_ids, dtype=torch.long, device=device)
    block = int(model.block_size)

    if tune == "lm_head":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("lm_head.")
    elif tune == "all":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError("tune must be one of: lm_head, all")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

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
            print(f"step {step:5d} | loss {loss.item():.4f}")

    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-ckpt", type=str, default="models/transformer_judge.pt")
    ap.add_argument("--out", type=str, default="models/oe_then_shakespeare.pt")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tune", choices=("lm_head", "all"), default="lm_head")
    ap.add_argument("--max-new", type=int, default=180)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading base model + judge checkpoint...")
    model, tokenizer, cfg, judge = _load_judge_ckpt(args.judge_ckpt, device=device)

    print("Loading Shakespeare dataset...")
    sp = list(load_shakespeare_dataset(0.0, 0.0))
    sp_text = EOS_TOKEN.join(str(x) for x in sp)

    prompts = [
        "To be, or not to be ",
        "The ",
        "Romeo, wherefore art thou ",
        "All the world's a stage ",
    ]

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    score_rows: list[dict] = []

    print("\nBefore fine-tune:")
    for p in prompts:
        out = generate(model, tokenizer, p, device=device, max_new_tokens=args.max_new)
        score = _judge_score_text(judge, tokenizer, out, device=device)
        print(f"\nPrompt: {p!r}")
        print(f"Score:  {score:.4f}")
        print(f"Text:   {out}")
        score_rows.append({"prompt": p.strip(), "stage": "before", "score": f"{score:.4f}", "text": out[:200]})

    print("\nFine-tuning on Shakespeare...")
    finetune_lm(
        model,
        tokenizer,
        train_text=sp_text,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        tune=args.tune,
    )

    print("\nAfter fine-tune:")
    for p in prompts:
        out = generate(model, tokenizer, p, device=device, max_new_tokens=args.max_new)
        score = _judge_score_text(judge, tokenizer, out, device=device)
        print(f"\nPrompt: {p!r}")
        print(f"Score:  {score:.4f}")
        print(f"Text:   {out}")
        score_rows.append({"prompt": p.strip(), "stage": "after", "score": f"{score:.4f}", "text": out[:200]})

    csv_path = results_dir / f"finetune_scores_{args.tune}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "stage", "score", "text"])
        writer.writeheader()
        writer.writerows(score_rows)
    print(f"\nSaved scores: {csv_path}")

    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base_model_config": cfg,
            "tokenizer": tokenizer,
            "model_state_dict": model.state_dict(),
        },
        out_path,
    )
    print(f"\nSaved fine-tuned model: {out_path}")


if __name__ == "__main__":
    main()

