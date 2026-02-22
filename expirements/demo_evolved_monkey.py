"""Demo the evolved transformer monkey.

This loads:
  - the transformer judge checkpoint (for base model weights + tokenizer)
  - the evolved lm_head weights (produced by monkeys/evolve_transformers.py)

Then it generates a few samples for slide-friendly prompts.

Run:
  python3 expirements/demo_evolved_monkey.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifier.transformer_judge import TransformerJudge
from monkeys.TransformerMonkey import TransformerMonkey


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
    return judge, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-ckpt", type=str, default="models/transformer_judge.pt")
    ap.add_argument("--lm-head", type=str, default="models/best_evolved_lm_head.pt")
    ap.add_argument("--max-new", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    judge, tokenizer = _load_judge_ckpt(args.judge_ckpt, device=device)

    lm_path = (REPO_ROOT / args.lm_head).resolve()
    if not lm_path.exists():
        raise FileNotFoundError(f"Missing evolved lm_head file: {lm_path}")

    lm = torch.load(lm_path, map_location=device)
    judge.transformer.lm_head.load_state_dict(lm["lm_head"])
    judge.transformer.eval()

    prompts = [
        "To be, or not to be ",
        "The ",
        "Romeo, wherefore art thou ",
        "All the world's a stage ",
    ]

    with torch.no_grad():
        for p in prompts:
            x = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
            y = judge.transformer.generate(x, max_new_tokens=args.max_new, temperature=float(args.temperature))
            out = tokenizer.decode(y[0].tolist())
            print(f"\nPrompt: {p!r}")
            print(f"Output:  {out}")


if __name__ == "__main__":
    main()

