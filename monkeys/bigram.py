"""
Character-level bigram language model.
A "less random" monkey by learning what character usually
comes after what character from a corpus (e.g., Shakespeare).
"""

import random
from collections import Counter

from monkeys.generator import ALPHABET


def clean_text(text: str, alphabet: str = ALPHABET) -> str:
    text = text.lower()
    return "".join(ch for ch in text if ch in alphabet) #lowercase and drop characters not in alphabet


class BigramModel:
    def __init__(self, alphabet: str = ALPHABET, smoothing: float = 1.0):
        self.alphabet = alphabet
        self.smoothing = smoothing

        # counts[prev][next] = how often we saw prev -> next
        self.counts = {}  # prev_char (or "<s>") -> Counter(next_char)
        self.fitted = False

    def fit(self, texts):
        # build bigram counts from a list/iterable of strings
        self.counts = {}

        for raw in texts:
            text = clean_text(raw, self.alphabet)
            if not text:
                continue

            prev = "<s>"
            for ch in text:
                if prev not in self.counts:
                    self.counts[prev] = Counter()
                self.counts[prev][ch] += 1
                prev = ch

        self.fitted = True
        return self

    def _sample_next_char(self, prev, rng):
        counter = self.counts.get(prev, Counter())

        chars = list(self.alphabet)
        weights = [counter.get(ch, 0) + self.smoothing for ch in chars] # add-k smoothing so every char has some probability

        # if smoothing==0 and counter empty, fall back to uniform
        if sum(weights) == 0:
            weights = [1 for _ in chars]

        return rng.choices(chars, weights=weights, k=1)[0]

    def generate(self, length, seed_text="", seed=None):
        if not self.fitted:
            raise RuntimeError("Call fit(texts) before generate().")

        rng = random.Random(seed)

        prefix = clean_text(seed_text, self.alphabet)
        if len(prefix) >= length:
            return prefix[:length]

        out = list(prefix)
        prev = out[-1] if out else "<s>"

        while len(out) < length:
            nxt = self._sample_next_char(prev, rng)
            out.append(nxt)
            prev = nxt

        return "".join(out)

