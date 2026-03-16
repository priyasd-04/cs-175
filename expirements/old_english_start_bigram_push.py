"""Old English start to Shakespeare push (bigram).

Start a character-level bigram model from an Old English
corpus, then iteratively nudges it toward Shakespeare style by:

- sampling text from the bigram
- scoring samples with our Shakespeare-vs-noise judge
- updating the bigram on high-scoring samples

"""

from __future__ import annotations

import argparse
import binascii
import csv
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from classifier.judge import Noise_Shakespeare_Classifier
from utils.load_datasets import load_old_english_dataset, load_shakespeare_dataset
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


def _stats(scores: Sequence[float]) -> dict:
    s = sorted(float(x) for x in scores)
    return {
        "n": len(s),
        "mean": mean(s),
        "p25": _percentile(s, 0.25),
        "p50": _percentile(s, 0.50),
        "p75": _percentile(s, 0.75),
        "max": s[-1],
    }


def _summarize(tag: str, scores: Sequence[float]) -> dict:
    st = _stats(scores)
    print(
        f"{tag:>10s} | n={st['n']:4d} | mean={st['mean']:.4f} | "
        f"p25={st['p25']:.4f} p50={st['p50']:.4f} "
        f"p75={st['p75']:.4f} | max={st['max']:.4f}"
    )
    return st


def _iter_seed(base_seed: int, it: int, i: int) -> int:
    """seed generator"""
    return base_seed * 100_000 + it * 1_000 + i


def _char_ngrams(text: str, n: int) -> set[str]:
    text = text.strip()
    if n <= 0 or len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _signature_ngrams(ngrams: set[str], k: int) -> list[str]:
    """Pick k stable 'signature' ngrams (smallest crc32) for indexing."""
    if not ngrams or k <= 0:
        return []
    scored = [(binascii.crc32(ng.encode("utf-8")) & 0xFFFFFFFF, ng) for ng in ngrams]
    scored.sort(key=lambda x: x[0])
    return [ng for _, ng in scored[:k]]


class AntiCopyChecker:
    """Detect near-duplicate text vs reference lines using char n-grams.

    We use a "containment" score rather than plain Jaccard:
        containment(text, ref) = |ngrams(text) ∩ ngrams(ref)| / |ngrams(ref)|

    This is better when `text` is long (300 chars) and `ref` is short,
    because a copied line should have high containment even if Jaccard is low.
    """

    def __init__(
        self,
        reference_texts: Sequence[str],
        *,
        ngram: int = 5,
        signatures_per_text: int = 12,
        max_candidates: int = 2000,
    ) -> None:
        self.ngram = int(ngram)
        self.signatures_per_text = int(signatures_per_text)
        self.max_candidates = int(max_candidates)

        self._ref_ngrams: list[set[str]] = []
        self._sig_index: dict[str, list[int]] = {}

        for idx, raw in enumerate(reference_texts):
            ng = _char_ngrams(raw, self.ngram)
            self._ref_ngrams.append(ng)
            for sig in _signature_ngrams(ng, self.signatures_per_text):
                self._sig_index.setdefault(sig, []).append(idx)

    def max_similarity(self, text: str) -> float:
        """Maximum containment similarity to any reference line (0..1)."""
        text_ngrams = _char_ngrams(text, self.ngram)
        if not text_ngrams:
            return 0.0

        # Candidate pruning via signature index.
        candidates: set[int] = set()
        for sig in _signature_ngrams(text_ngrams, self.signatures_per_text):
            for idx in self._sig_index.get(sig, []):
                candidates.add(idx)
                if len(candidates) >= self.max_candidates:
                    break
            if len(candidates) >= self.max_candidates:
                break

        if not candidates:
            return 0.0

        best = 0.0
        for idx in candidates:
            ref = self._ref_ngrams[idx]
            if not ref:
                continue
            inter = len(text_ngrams & ref)
            if inter == 0:
                continue
            sim = inter / len(ref)
            if sim > best:
                best = sim
        return float(best)


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
        "--anti-copy",
        action="store_true",
        help="Drop/zero-weight samples that are near-duplicates of Shakespeare training lines",
    )
    ap.add_argument("--copy-ngram", type=int, default=5, help="n for character n-grams in anti-copy check")
    ap.add_argument(
        "--copy-threshold",
        type=float,
        default=0.80,
        help="If max containment similarity >= threshold, treat sample as too similar (0..1)",
    )
    ap.add_argument(
        "--copy-ref-lines",
        type=int,
        default=5000,
        help="How many Shakespeare lines to use as anti-copy reference",
    )
    ap.add_argument(
        "--copy-signatures",
        type=int,
        default=12,
        help="How many signature n-grams to index per reference/sample (speed/quality tradeoff)",
    )
    ap.add_argument(
        "--copy-max-candidates",
        type=int,
        default=2000,
        help="Max candidate reference lines checked per sample (caps runtime)",
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

    anti_copy: AntiCopyChecker | None = None
    if args.anti_copy:
        sp = load_shakespeare_dataset(0.0, 0.0, seed=args.seed)
        ref_lines = [str(x).strip() for x in sp[: args.copy_ref_lines] if str(x).strip()]
        anti_copy = AntiCopyChecker(
            ref_lines,
            ngram=args.copy_ngram,
            signatures_per_text=args.copy_signatures,
            max_candidates=args.copy_max_candidates,
        )
        print(
            f"Anti-copy enabled | ref_lines={len(ref_lines)} | ngram={args.copy_ngram} | "
            f"threshold={args.copy_threshold}"
        )

    # 3) Iteratively push bigram toward Shakespeare.
    results_dir = Path(REPO_ROOT) / "results"
    results_dir.mkdir(exist_ok=True)

    iteration_rows: list[dict] = []

    print("\nIterating...")
    for it in range(args.iterations):
        samples = [bigram.generate(args.sample_len, seed=_iter_seed(args.seed, it, i)) for i in range(args.n_samples)]
        scores = judge.shakespeare_likeliness(samples)

        st = _summarize(f"iter{it}", scores)
        iteration_rows.append({"iteration": it, **st})

        keep_mask: list[bool] = [True] * len(samples)
        if anti_copy is not None:
            sims = [anti_copy.max_similarity(t) for t in samples]
            threshold = float(args.copy_threshold)
            keep_mask = [s < threshold for s in sims]
            kept = sum(keep_mask)
            copy_rate = 1.0 - (kept / len(keep_mask))
            print(
                f"{'':>10s} | anti_copy copy_rate={copy_rate:.3f} | "
                f"mean_sim={mean(sims):.3f} | max_sim={max(sims):.3f}"
            )

        if args.mode == "topk":
            ranked = sorted(zip(scores, samples, keep_mask), key=lambda x: x[0], reverse=True)
            filtered = [(p, t) for p, t, keep in ranked if keep]
            if not filtered:
                filtered = [(p, t) for p, t, _ in ranked]
            best_texts = [t for _, t in filtered[: args.top_k]]
            bigram.partial_fit(best_texts, decay=args.decay)
        else:
            total = float(sum(max(0.0, s) for s in scores)) or 1.0
            scale = float(args.weight_scale) if args.weight_scale is not None else float(args.top_k)
            weights = [scale * (max(0.0, s) / total) for s in scores]
            if anti_copy is not None:
                weights = [w if keep else 0.0 for w, keep in zip(weights, keep_mask)]
            bigram.partial_fit(samples, weights=weights, decay=args.decay)

    # Show a few best samples at the end.
    final_samples = [bigram.generate(args.sample_len, seed=args.seed * 999_999 + i) for i in range(args.final_n)]
    final_scores = judge.shakespeare_likeliness(final_samples)
    top = sorted(zip(final_scores, final_samples), reverse=True)[:5]
    print("\nTop samples (end):")
    for p, t in top:
        print(f"p={p:.4f} | {t[:140]}...")

    # --- Save CSV ---
    suffix = f"{args.mode}"
    if args.anti_copy:
        suffix += "_anticopy"
    if args.decay is not None:
        suffix += f"_decay{args.decay}"
    csv_path = results_dir / f"bigram_push_{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "n", "mean", "p25", "p50", "p75", "max"])
        writer.writeheader()
        writer.writerows(iteration_rows)
    print(f"\nSaved iteration stats: {csv_path}")

    # --- Save plot ---
    iters = [r["iteration"] for r in iteration_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, [r["mean"] for r in iteration_rows], label="Mean", linewidth=2)
    ax.fill_between(
        iters,
        [r["p25"] for r in iteration_rows],
        [r["p75"] for r in iteration_rows],
        alpha=0.25,
        label="p25–p75",
    )
    ax.plot(iters, [r["max"] for r in iteration_rows], "--", label="Max", alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Judge Score (Shakespeare probability)")
    ax.set_title(f"Bigram Push: OE → Shakespeare ({args.mode})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = results_dir / f"bigram_push_{suffix}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()

