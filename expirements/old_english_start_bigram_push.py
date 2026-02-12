"""Old English start to Shakespeare push (bigram).

Start a character-level bigram model from an Old English
corpus, then iteratively nudges it toward Shakespeare style by:

- sampling text from the bigram
- scoring samples with our Shakespeare-vs-noise judge
- updating the bigram on high-scoring samples

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from classifier.judge import Noise_Shakespeare_Classifier
from load_datasets import load_old_english_dataset
from monkeys.bigram import BigramModel


NOISE_FILE = "noise_dataset/noise.txt"
JUDGE_SEED = 13_625_442


def _extract_text(example: Any) -> str:
    """Extract a string from a dataset row with flexible schema"""
    if isinstance(example, str):
        return example
    if isinstance(example, dict):
        for key in (
            "old",
            "old_english",
            "oe",
            "source",
            "text",
            "translation",
            "modern",
            "modern_english",
            "en",
            "target",
        ):
            v = example.get(key)
            if isinstance(v, str) and v.strip():
                return v
        # last resort is join any non-empty string fields.
        vals = [v for v in example.values() if isinstance(v, str) and v.strip()]
        return " ".join(vals) if vals else str(example)
    return str(example)


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Nearest-rank percentile for a pre-sorted sequence."""
    if not sorted_vals:
        return float("nan")
    idx = int(p * (len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def _summarize(tag: str, scores: Sequence[float]) -> None:
    s = sorted(float(x) for x in scores)
    print(
        f"{tag:>10s} | n={len(s):4d} | mean={mean(s):.4f} | "
        f"p25={_percentile(s, 0.25):.4f} p50={_percentile(s, 0.50):.4f} "
        f"p75={_percentile(s, 0.75):.4f} | max={s[-1]:.4f}"
    )


def _iter_seed(base_seed: int, it: int, i: int) -> int:
    """seed generator"""
    return base_seed * 100_000 + it * 1_000 + i


def main() -> None:
    ap = argparse.ArgumentParser(description="Old English bigram → judge-guided Shakespeare push")
    ap.add_argument("--iterations", type=int, default=40, help="Number of update iterations")
    ap.add_argument("--sample-len", type=int, default=300, help="Characters per generated sample")
    ap.add_argument("--n-samples", type=int, default=200, help="Samples per iteration")
    ap.add_argument("--top-k", type=int, default=40, help="Top-K samples used in topk mode")
    ap.add_argument("--smoothing", type=float, default=0.5, help="Add-k smoothing for bigram sampling")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for deterministic sampling")
    ap.add_argument(
        "--mode",
        choices=("topk", "weighted"),
        default="topk",
        help="Update rule: topk = update on top-K samples; weighted = score-weighted update on all samples",
    )
    ap.add_argument(
        "--weight-scale",
        type=float,
        default=None,
        help=(
            "Only for --mode weighted. Scales the normalized weights so updates are stronger. "
            "If not set, defaults to --top-k (roughly comparable update size to topk mode)."
        ),
    )
    ap.add_argument(
        "--decay",
        type=float,
        default=None,
        help="Optional factor in (0,1]. If set, old counts decay each iteration before adding new ones.",
    )
    ap.add_argument(
        "--final-n",
        type=int,
        default=60,
        help="How many samples to draw at the end (for printing top examples)",
    )
    args = ap.parse_args()

    # 1) Load Old English dataset and fit starting bigram.
    oe_ds = load_old_english_dataset(0.0, 0.0, seed=args.seed)
    oe_texts = [_extract_text(row) for row in oe_ds]
    oe_texts = [t for t in oe_texts if t.strip()]
    print(f"Loaded Old English examples: {len(oe_texts)}")

    bigram = BigramModel(smoothing=args.smoothing).fit(oe_texts)

    # 2) Train judge.
    judge = Noise_Shakespeare_Classifier(seed=JUDGE_SEED)
    if not Path(NOISE_FILE).exists():
        raise FileNotFoundError(f"Expected noise file at '{NOISE_FILE}'.")
    judge.train(NOISE_FILE, show_report=False)

    # 3) Iteratively push bigram toward Shakespeare.
    print("\nIterating...")
    for it in range(args.iterations):
        samples = [bigram.generate(args.sample_len, seed=_iter_seed(args.seed, it, i)) for i in range(args.n_samples)]
        scores = judge.shakespeare_likeliness(samples)

        _summarize(f"iter{it}", scores)

        if args.mode == "topk":
            ranked = sorted(zip(scores, samples), reverse=True)
            best_texts = [t for _, t in ranked[: args.top_k]]
            bigram.partial_fit(best_texts, decay=args.decay)
        else:
            # Normalize weights, then scale so update magnitude is meaningful.
            total = float(sum(max(0.0, s) for s in scores)) or 1.0
            scale = float(args.weight_scale) if args.weight_scale is not None else float(args.top_k)
            weights = [scale * (max(0.0, s) / total) for s in scores]
            bigram.partial_fit(samples, weights=weights, decay=args.decay)

    # Show a few best samples at the end.
    final_samples = [bigram.generate(args.sample_len, seed=args.seed * 999_999 + i) for i in range(args.final_n)]
    final_scores = judge.shakespeare_likeliness(final_samples)
    top = sorted(zip(final_scores, final_samples), reverse=True)[:5]
    print("\nTop samples (end):")
    for p, t in top:
        print(f"p={p:.4f} | {t[:140]}...")


if __name__ == "__main__":
    main()

