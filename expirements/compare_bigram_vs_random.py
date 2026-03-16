"""
Compare random monkeys vs. bigram-trained monkeys using the current "judge".
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from classifier.judge import Noise_Shakespeare_Classifier
from utils.load_datasets import load_shakespeare_dataset
from monkeys.bigram import BigramModel
from monkeys.generator import random_monkey


def _stats(probs: list[float]) -> dict:
    probs_sorted = sorted(probs)
    n = len(probs_sorted)
    q = lambda p: probs_sorted[int(p * (n - 1))]
    return {
        "n": n,
        "mean": mean(probs_sorted),
        "min": probs_sorted[0],
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "max": probs_sorted[-1],
    }


def summarize(name: str, probs: list[float]) -> dict:
    st = _stats(probs)
    print(f"\n== {name} ==")
    print(f"n={st['n']}")
    print(f"mean={st['mean']:.4f}")
    print(f"min={st['min']:.4f}  25%={st['p25']:.4f}  50%={st['p50']:.4f}  75%={st['p75']:.4f}  max={st['max']:.4f}")
    return st


def main() -> None:
    sample_len = 300
    n_samples = 60
    shakespeare_fit_lines = 2000

    sp = load_shakespeare_dataset(0.0, 0.0)

    bigram = BigramModel(smoothing=0.5).fit(sp[:shakespeare_fit_lines])

    judge = Noise_Shakespeare_Classifier()
    judge.train("noise_dataset/noise.txt", show_report=False)

    random_texts = [random_monkey(sample_len) for _ in range(n_samples)]
    bigram_texts = [bigram.generate(sample_len, seed=i) for i in range(n_samples)]

    random_probs = judge.shakespeare_likeliness(random_texts)
    bigram_probs = judge.shakespeare_likeliness(bigram_texts)

    random_st = summarize("Random monkey", random_probs)
    bigram_st = summarize("Bigram monkey (trained on Shakespeare)", bigram_probs)

    def top_k(texts: list[str], probs: list[float], k: int = 3):
        scored = sorted(zip(probs, texts), reverse=True)
        return scored[:k]

    print("\n== Example outputs (top by judge score) ==")
    print("\nRandom monkey:")
    for p, t in top_k(random_texts, random_probs):
        print(f"p={p:.4f} | {t[:120]}...")

    print("\nBigram monkey:")
    for p, t in top_k(bigram_texts, bigram_probs):
        print(f"p={p:.4f} | {t[:120]}...")

    # --- Save results ---
    results_dir = Path(REPO_ROOT) / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "bigram_vs_random.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "n", "mean", "min", "p25", "p50", "p75", "max"])
        writer.writeheader()
        writer.writerow({"source": "random", **random_st})
        writer.writerow({"source": "bigram", **bigram_st})
    print(f"\nSaved stats: {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(random_probs, bins=20, alpha=0.7, label="Random", color="#d62728")
    axes[0].hist(bigram_probs, bins=20, alpha=0.7, label="Bigram", color="#2ca02c")
    axes[0].set_xlabel("Judge Score (Shakespeare probability)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score Distribution: Random vs Bigram")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    sources = ["Random", "Bigram"]
    means = [random_st["mean"], bigram_st["mean"]]
    colors = ["#d62728", "#2ca02c"]
    axes[1].bar(sources, means, color=colors, width=0.5)
    axes[1].set_ylabel("Mean Judge Score")
    axes[1].set_title("Mean Score Comparison")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    plot_path = results_dir / "bigram_vs_random.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()

