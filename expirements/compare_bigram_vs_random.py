"""
Compare random monkeys vs. bigram-trained monkeys using the current "judge".
"""

from __future__ import annotations

import os
import sys
from statistics import mean

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from classifier.judge import Noise_Shakespeare_Classifier
from utils.load_datasets import load_shakespeare_dataset
from monkeys.bigram import BigramModel
from monkeys.generator import random_monkey


def summarize(name: str, probs: list[float]) -> None:
    probs_sorted = sorted(probs)
    n = len(probs_sorted)
    q = lambda p: probs_sorted[int(p * (n - 1))]

    print(f"\n== {name} ==")
    print(f"n={n}")
    print(f"mean={mean(probs_sorted):.4f}")
    print(f"min={probs_sorted[0]:.4f}  25%={q(0.25):.4f}  50%={q(0.50):.4f}  75%={q(0.75):.4f}  max={probs_sorted[-1]:.4f}")


def main() -> None:
    sample_len = 300
    n_samples = 60
    shakespeare_fit_lines = 2000

    # load Shakespeare lines (this uses HuggingFace datasets)
    sp = load_shakespeare_dataset(0.0, 0.0)

    # train bigram model
    bigram = BigramModel(smoothing=0.5).fit(sp[:shakespeare_fit_lines])

    # train judge (Shakespeare vs noise file already in repo)
    judge = Noise_Shakespeare_Classifier()
    judge.train("noise_dataset/noise.txt", show_report=False)

    # generate samples
    random_texts = [random_monkey(sample_len) for _ in range(n_samples)]
    bigram_texts = [bigram.generate(sample_len, seed=i) for i in range(n_samples)]

    # score samples
    random_probs = judge.shakespeare_likeliness(random_texts)
    bigram_probs = judge.shakespeare_likeliness(bigram_texts)

    # summaries
    summarize("Random monkey", random_probs)
    summarize("Bigram monkey (trained on Shakespeare)", bigram_probs)

    # few examples (top 3 by judge score from each)
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


if __name__ == "__main__":
    main()

