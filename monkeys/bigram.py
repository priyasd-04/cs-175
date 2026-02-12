"""
Character-level bigram language model ("bigram monkey").

The model learns P(next_char | prev_char) from a corpus and can then sample
new strings that resemble the training distribution.
"""

from __future__ import annotations

import random
import unicodedata
from collections import Counter
from typing import Iterable, Optional, Sequence

from monkeys.generator import ALPHABET


START_TOKEN = "<s>"


def _fold_to_ascii(text: str) -> str:
    """Best-effort map non-ascii chars to ascii-ish equivalents.

    Useful for corpora containing Old English characters (þ/ð/æ, etc.)
    that are not in our `ALPHABET`.
    """
    # Common replacements
    replacements: dict[str, str] = {
        "Þ": "th",
        "þ": "th",
        "Ð": "th",
        "ð": "th",
        "Æ": "ae",
        "æ": "ae",
        "Ǣ": "ae",
        "ǣ": "ae",
        "Ǽ": "ae",
        "ǽ": "ae",
        "Œ": "oe",
        "œ": "oe",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Strip combining marks (accents) after unicode decomposition
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def clean_text(text: str, alphabet: str = ALPHABET) -> str:
    """Lowercase + keep only characters in `alphabet`."""
    text = _fold_to_ascii(text).lower()
    return "".join(ch for ch in text if ch in alphabet)


class BigramModel:
    """A smoothed character bigram model with sampling + incremental updates."""

    def __init__(self, alphabet: str = ALPHABET, smoothing: float = 1.0):
        self.alphabet = alphabet
        self.smoothing = smoothing

        # counts[prev][next] = how often we saw prev -> next
        # values are floats because we support weighted updates.
        self.counts: dict[str, Counter[str]] = {}
        self.fitted = False

    def fit(self, texts: Iterable[str]) -> BigramModel:
        """Reset counts and fit from scratch on an iterable of strings."""
        self.counts = {}
        for raw in texts:
            self._add_counts_from_text(raw, weight=1.0)

        self.fitted = True
        return self

    def partial_fit(
        self,
        texts: Iterable[str],
        weights: Optional[Sequence[float]] = None,
        *,
        decay: float | None = None,
    ) -> BigramModel:
        """Incrementally update bigram counts without resetting them.

        Args:
            texts: Iterable of strings to learn from.
            weights: Optional per-text weights (same order as `texts`).
                Higher weight => bigger count updates. Non-positive weights
                are ignored.
            decay: Optional factor in (0, 1]. If provided, existing counts are
                multiplied by `decay` before adding new counts.
        """
        if decay is not None:
            if not (0 < decay <= 1):
                raise ValueError("decay must be in (0, 1].")
            for prev, counter in self.counts.items():
                for ch in list(counter.keys()):
                    counter[ch] *= decay

        if weights is None:
            for raw in texts:
                self._add_counts_from_text(raw, weight=1.0)
        else:
            for raw, w in zip(texts, weights):
                w = float(w)
                if w > 0:
                    self._add_counts_from_text(raw, weight=w)

        self.fitted = True
        return self

    def _add_counts_from_text(self, raw: str, *, weight: float) -> None:
        text = clean_text(raw, self.alphabet)
        if not text:
            return

        prev = START_TOKEN
        for ch in text:
            if prev not in self.counts:
                self.counts[prev] = Counter()
            self.counts[prev][ch] += weight
            prev = ch

    def _sample_next_char(self, prev: str, rng: random.Random) -> str:
        counter = self.counts.get(prev, Counter())

        chars = list(self.alphabet)
        # Add-k smoothing so every char has some probability.
        weights = [counter.get(ch, 0.0) + self.smoothing for ch in chars]

        # If smoothing == 0 and counter empty, fall back to uniform.
        if sum(weights) == 0:
            weights = [1.0 for _ in chars]

        return rng.choices(chars, weights=weights, k=1)[0]

    def generate(self, length: int, seed_text: str = "", seed: int | None = None) -> str:
        """Sample a string of `length` characters."""
        if not self.fitted:
            raise RuntimeError("Call fit(texts) before generate().")

        rng = random.Random(seed)

        prefix = clean_text(seed_text, self.alphabet)
        if len(prefix) >= length:
            return prefix[:length]

        out = list(prefix)
        prev = out[-1] if out else START_TOKEN

        while len(out) < length:
            nxt = self._sample_next_char(prev, rng)
            out.append(nxt)
            prev = nxt

        return "".join(out)

