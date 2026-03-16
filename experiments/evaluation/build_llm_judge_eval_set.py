# Author(s): Amish Kunal
"""Build a labeled eval set with 0-shot and few-shot prompts.

Run:
  python3 expirements/build_llm_judge_eval_set.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.monkeys.bigram import BigramModel
from src.monkeys.generator import random_monkey
from src.monkeys.TransformerMonkey import CharTokenizer, TransformerMonkey
from src.utils.load_datasets import load_old_english_dataset, load_shakespeare_dataset


@dataclass
class Example:
    source: str
    text: str
    label: int


def _load_transformer_from_judge_ckpt(path: Path, device: str) -> tuple[TransformerMonkey, CharTokenizer] | None:
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    tok = ckpt["tokenizer"]
    cfg = ckpt["base_model_config"]
    model = TransformerMonkey(
        tok.vocab_size,
        tokenizer=tok,
        block_size=int(cfg["block_size"]),
        n_embd=int(cfg["n_embd"]),
        n_head=int(cfg["n_head"]),
        n_layer=int(cfg["n_layer"]),
        dropout=float(cfg.get("dropout", 0.2)),
    ).to(device)
    model.eval()
    return model, tok


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
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [0]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8)
        out.append(tokenizer.decode(y[0].tolist()))
    return out


def _few_shot_prefix() -> str:
    return (
        "Task: Decide if text is Shakespeare-like.\n"
        "Output exactly one token: SHAKESPEARE or NOT_SHAKESPEARE.\n\n"
        "Example 1:\n"
        "Text: To be, or not to be: that is the question.\n"
        "Label: SHAKESPEARE\n\n"
        "Example 2:\n"
        "Text: xqzv pmlk !! zzz ttt qqq ???\n"
        "Label: NOT_SHAKESPEARE\n\n"
        "Example 3:\n"
        "Text: In this old tongue the kingly halls were sung with iron words.\n"
        "Label: NOT_SHAKESPEARE\n\n"
    )


def _zero_shot_prompt(text: str) -> str:
    return (
        "Task: Decide if text is Shakespeare-like.\n"
        "Output exactly one token: SHAKESPEARE or NOT_SHAKESPEARE.\n\n"
        f"Text:\n{text}\n\n"
        "Label:"
    )


def _few_shot_prompt(text: str) -> str:
    return _few_shot_prefix() + f"Text:\n{text}\n\nLabel:"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-source", type=int, default=40, help="How many examples per source.")
    ap.add_argument("--sample-len", type=int, default=300)
    ap.add_argument("--shakespeare-fit-lines", type=int, default=2000)
    ap.add_argument("--judge-ckpt", type=str, default="models/transformer_judge.pt")
    ap.add_argument("--transformer-prompt", type=str, default="To be, or not to be ")
    ap.add_argument("--transformer-new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-jsonl", type=str, default="results/llm_judge_eval_examples.jsonl")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = int(args.n_per_source)

    print("Loading datasets...")
    sp = list(load_shakespeare_dataset(0.0, 0.0, seed=args.seed))
    oe = list(load_old_english_dataset(0.0, 0.0, seed=args.seed))

    real_sp = [str(x) for x in sp[:n]]
    real_oe = [str(x) for x in oe[:n]]

    print("Training bigram baseline generator...")
    bigram = BigramModel(smoothing=0.5).fit([str(x) for x in sp[: args.shakespeare_fit_lines]])
    bigram_texts = [bigram.generate(args.sample_len, seed=args.seed + i) for i in range(n)]

    print("Generating random baseline text...")
    random_texts = [random_monkey(args.sample_len) for _ in range(n)]

    print("Generating transformer text (if checkpoint exists)...")
    ckpt_path = (REPO_ROOT / args.judge_ckpt).resolve()
    transformer_texts: list[str] = []
    loaded = _load_transformer_from_judge_ckpt(ckpt_path, device=device)
    if loaded is not None:
        tr_model, tr_tok = loaded
        transformer_texts = _gen_transformer_texts(
            tr_model,
            tr_tok,
            n=n,
            prompt=str(args.transformer_prompt),
            max_new_tokens=int(args.transformer_new),
            device=device,
        )
    else:
        print(f"Checkpoint not found at {ckpt_path}; skipping transformer-generated source.")

    examples: list[Example] = []
    examples.extend(Example("real_sp", t, 1) for t in real_sp)
    examples.extend(Example("real_oe", t, 0) for t in real_oe)
    examples.extend(Example("random", t, 0) for t in random_texts)
    examples.extend(Example("bigram", t, 0) for t in bigram_texts)
    if transformer_texts:
        examples.extend(Example("transformer", t, 0) for t in transformer_texts)

    out_path = (REPO_ROOT / args.out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            row = {
                "source": ex.source,
                "label": ex.label,
                "text": ex.text,
                "zero_shot_prompt": _zero_shot_prompt(ex.text),
                "few_shot_prompt": _few_shot_prompt(ex.text),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.source] = counts.get(ex.source, 0) + 1

    print(f"\nSaved: {out_path}")
    print("Counts by source:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")
    print(f"Total examples: {len(examples)}")


if __name__ == "__main__":
    main()

