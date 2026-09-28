"""Novelty and compliance filtering against the reference AMP database.

Two distinct constraints from `verify_submission.py`:

1. The **full 50,000 library** must contain no sequence that appears *exactly* in
   `data/antibacterial.fasta`.
2. Every **top-100** sequence must satisfy
   `Levenshtein.ratio(seq, ref) <= 0.80` for *every* one of the 39,448 references.

Constraint 2 is the expensive one, so an earlier version of this module pruned
candidate references with a k-mer inverted index plus a length band. That was a
mistake and it showed up in testing: the k-mer index is a *heuristic* — two
sequences can exceed the threshold without sharing a k-mer — and the length band
was derived for a threshold of 0.80 while being applied at our tighter 0.72
working margin. The combination let a candidate through at ratio 0.783
(``KWKFKIKFHFHKKW`` vs the length-9 ``KKFKKFFKK``), which is still compliant but
defeats the point of having a margin.

The replacement keeps only an *exact*, loss-free bound. Since
``ratio(a, b) = 1 - indel(a, b) / (|a| + |b|)`` and ``indel(a, b) >= ||a| - |b||``,

    ratio(a, b) <= 2 * min(|a|, |b|) / (|a| + |b|).

Requiring that upper bound to exceed a threshold ``t`` gives an exact admissible
band for the reference length ``nb`` given the candidate length ``na``:

    t * na / (2 - t)  <  nb  <  na * (2 - t) / t.

References outside that band *cannot* reach ``t``, so skipping them loses
nothing. Everything inside is scored exhaustively. At t = 0.72 this prunes
little, but the full sweep costs about 2 s per 100 candidates against all 39,448
references, so exhaustive screening is affordable and correctness wins.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from Levenshtein import ratio as lev_ratio

from .constants import MAX_TOP_SIMILARITY


def length_band(na: int, threshold: float) -> tuple[float, float]:
    """Exact admissible reference-length band for a candidate of length `na`."""
    lo = threshold * na / (2.0 - threshold)
    hi = na * (2.0 - threshold) / threshold
    return lo, hi


class ReferenceIndex:
    """Exact-match set plus a length-bucketed index for similarity screening."""

    def __init__(self, references: list[str]):
        self.references = references
        self.exact = set(references)

        self._by_length: dict[int, list[str]] = defaultdict(list)
        for ref in references:
            self._by_length[len(ref)].append(ref)
        self._lengths = sorted(self._by_length)

    # --- constraint 1 -------------------------------------------------------
    def is_exact_match(self, sequence: str) -> bool:
        return sequence in self.exact

    def filter_exact(self, sequences: list[str]) -> list[str]:
        """Drop any sequence present verbatim in the reference database."""
        return [s for s in sequences if s not in self.exact]

    # --- constraint 2 -------------------------------------------------------
    def _candidate_refs(self, sequence: str, threshold: float):
        lo, hi = length_band(len(sequence), threshold)
        for n in self._lengths:
            if lo < n < hi:
                yield from self._by_length[n]

    def max_similarity(self, sequence: str, threshold: float = MAX_TOP_SIMILARITY) -> float:
        """Highest Levenshtein ratio against the reference database.

        Only references that could possibly reach `threshold` are scored; the
        bound is exact, so the returned value is correct whenever it matters
        (i.e. whenever it is at or above `threshold`). Values reported below the
        threshold are a valid lower bound.
        """
        best = 0.0
        for ref in self._candidate_refs(sequence, threshold):
            r = lev_ratio(sequence, ref)
            if r > best:
                best = r
                if best >= 1.0:
                    break
        return best

    def exceeds(self, sequence: str, threshold: float = MAX_TOP_SIMILARITY) -> bool:
        """True if any reference exceeds `threshold`. Exits at the first hit."""
        for ref in self._candidate_refs(sequence, threshold):
            if lev_ratio(sequence, ref) > threshold:
                return True
        return False

    def screen(self, sequences: list[str], threshold: float = MAX_TOP_SIMILARITY) -> np.ndarray:
        """Boolean mask: True where the sequence passes the similarity constraint."""
        return np.array([not self.exceeds(s, threshold) for s in sequences], dtype=bool)


def verify_top_similarity(
    top: list[str],
    references: list[str],
    threshold: float = MAX_TOP_SIMILARITY,
) -> list[tuple[str, str, float]]:
    """Exhaustive final check with no pruning at all, mirroring the organizers'
    `_veritfy_max_simularity` exactly.

    Deliberately does not use `ReferenceIndex`: this is the last line of defence
    before we write the file, so it must not share any pruning logic that could
    itself be wrong.

    Returns the offending (sequence, reference, ratio) triples; empty means the
    top list is compliant.
    """
    violations: list[tuple[str, str, float]] = []
    for seq in top:
        for ref in references:
            r = lev_ratio(seq, ref)
            if r > threshold:
                violations.append((seq, ref, r))
                break
    return violations
