"""Novelty and compliance filtering against the reference AMP database.

Two distinct constraints from `verify_submission.py`:

1. The **full 50,000 library** must contain no sequence that appears *exactly* in
   `data/antibacterial.fasta`.
2. Every **top-100** sequence must satisfy
   `Levenshtein.ratio(seq, ref) <= 0.80` for *every* one of the 39,448 references.

Constraint 2 is the expensive one. A naive sweep is 100 x 39,448 comparisons for
the final list, which is fine — but while *selecting* the top-100 we need to screen
tens of thousands of candidates, so this module adds two exact (loss-free) filters
that shrink the candidate set before any edit distance is computed.

Filter A - length bounds. `Levenshtein.ratio(a, b) = 1 - indel(a, b) / (|a| + |b|)`
and `indel(a, b) >= ||a| - |b||`, so
    ratio(a, b) <= (2 * min(|a|, |b|)) / (|a| + |b|).
Requiring that upper bound to exceed 0.8 gives `max < 1.5 * min`. References
outside that length band can never trip the threshold.

Filter B - k-mer blocking. Two sequences with ratio > t must share a long common
subsequence and therefore, in practice, at least one k-mer. We build an inverted
k-mer index over the references and only score references sharing a k-mer with the
candidate. `k=4` keeps the index small while discarding the overwhelming majority
of pairs. Filter B is a heuristic, so `verify_top_similarity` re-checks the final
list exhaustively with no blocking.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from Levenshtein import ratio as lev_ratio

from .constants import MAX_TOP_SIMILARITY


class ReferenceIndex:
    """Exact-match set plus a blocked index for similarity screening."""

    def __init__(self, references: list[str], kmer: int = 4):
        self.references = references
        self.kmer = kmer
        self.exact = set(references)

        self._by_length: dict[int, list[int]] = defaultdict(list)
        for i, ref in enumerate(references):
            self._by_length[len(ref)].append(i)

        self._kmer_index: dict[str, set[int]] = defaultdict(set)
        for i, ref in enumerate(references):
            for j in range(len(ref) - kmer + 1):
                self._kmer_index[ref[j : j + kmer]].add(i)

    # --- constraint 1 -------------------------------------------------------
    def is_exact_match(self, sequence: str) -> bool:
        return sequence in self.exact

    def filter_exact(self, sequences: list[str]) -> list[str]:
        """Drop any sequence present verbatim in the reference database."""
        return [s for s in sequences if s not in self.exact]

    # --- constraint 2 -------------------------------------------------------
    def _candidate_refs(self, sequence: str) -> set[int]:
        n = len(sequence)
        # Filter A: only lengths that could possibly exceed the threshold.
        lo, hi = int(np.ceil(n / 1.5)), int(np.floor(n * 1.5))
        allowed: set[int] = set()
        for length in range(lo, hi + 1):
            allowed.update(self._by_length.get(length, ()))
        if not allowed:
            return set()

        # Filter B: must share at least one k-mer.
        shared: set[int] = set()
        for j in range(len(sequence) - self.kmer + 1):
            shared.update(self._kmer_index.get(sequence[j : j + self.kmer], ()))
        return allowed & shared

    def max_similarity(self, sequence: str, *, exhaustive: bool = False) -> float:
        """Highest Levenshtein ratio against the reference database."""
        if exhaustive:
            pool: object = range(len(self.references))
        else:
            pool = self._candidate_refs(sequence)
        best = 0.0
        for i in pool:  # type: ignore[union-attr]
            r = lev_ratio(sequence, self.references[i])
            if r > best:
                best = r
                if best >= 1.0:
                    break
        return best

    def screen(self, sequences: list[str], threshold: float = MAX_TOP_SIMILARITY) -> np.ndarray:
        """Boolean mask: True where the sequence passes the similarity constraint."""
        return np.array(
            [self.max_similarity(s) <= threshold for s in sequences],
            dtype=bool,
        )


def verify_top_similarity(
    top: list[str],
    references: list[str],
    threshold: float = MAX_TOP_SIMILARITY,
) -> list[tuple[str, str, float]]:
    """Exhaustive final check, mirroring `_veritfy_max_simularity` exactly.

    Returns the list of offending (sequence, reference, ratio) triples; empty
    means the top list is compliant.
    """
    violations: list[tuple[str, str, float]] = []
    for seq in top:
        for ref in references:
            r = lev_ratio(seq, ref)
            if r > threshold:
                violations.append((seq, ref, r))
                break
    return violations
