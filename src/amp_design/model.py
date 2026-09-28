"""Baseline generative model: a smoothed variable-order Markov language model
over the 20-letter amino-acid alphabet, with an explicit length model.

This is the v0 generator. It is deliberately dependency-light (NumPy only) and
fully deterministic given a seed, so a compliant submission exists from day one.
A neural generator can be dropped in behind the same `sample()` interface without
touching the entry point.

Why a variable-order Markov model is a reasonable baseline here: AMP activity is
driven largely by composition and local amphipathic periodicity rather than by
long-range tertiary structure, so a model that captures k-mer statistics already
produces sequences whose charge / hydrophobic-moment distributions sit inside the
reference AMP cloud — which is exactly what seqme's ConformityScore rewards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import AMINO_ACIDS

N_LETTERS = len(AMINO_ACIDS)
BOS = N_LETTERS  # begin-of-sequence context symbol


class MarkovPeptideModel:
    """Katz-style backoff Markov model over amino-acid sequences.

    Attributes
    ----------
    order:
        Maximum context length.
    tables:
        `tables[k]` holds counts for contexts of length `k`, as a dict mapping a
        packed integer context to a length-20 count vector.
    length_pmf:
        Empirical distribution over sequence lengths, indexed from `min_length`.
    """

    def __init__(self, order: int, tables: list[dict[int, np.ndarray]], length_pmf: np.ndarray, min_length: int):
        self.order = order
        self.tables = tables
        self.length_pmf = length_pmf
        self.min_length = min_length

    # --- construction -------------------------------------------------------
    @classmethod
    def fit(
        cls,
        sequences: list[str],
        order: int = 4,
        weights: np.ndarray | None = None,
    ) -> "MarkovPeptideModel":
        """Fit from a corpus. `weights` optionally up-weights individual sequences."""
        idx_map = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        tables: list[dict[int, np.ndarray]] = [dict() for _ in range(order + 1)]

        if weights is None:
            weights = np.ones(len(sequences), dtype=np.float64)

        for seq, w in zip(sequences, weights):
            tokens = [BOS] * order + [idx_map[c] for c in seq]
            for pos in range(order, len(tokens)):
                nxt = tokens[pos]
                for k in range(order + 1):
                    ctx = tokens[pos - k : pos]
                    key = _pack(ctx)
                    table = tables[k]
                    vec = table.get(key)
                    if vec is None:
                        vec = np.zeros(N_LETTERS, dtype=np.float64)
                        table[key] = vec
                    vec[nxt] += w

        lengths = np.array([len(s) for s in sequences])
        min_length = int(lengths.min())
        counts = np.bincount(lengths - min_length, weights=weights)
        length_pmf = counts / counts.sum()

        return cls(order=order, tables=tables, length_pmf=length_pmf, min_length=min_length)

    # --- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload: dict[str, np.ndarray] = {
            "order": np.array([self.order]),
            "min_length": np.array([self.min_length]),
            "length_pmf": self.length_pmf,
        }
        for k, table in enumerate(self.tables):
            if table:
                keys = np.fromiter(table.keys(), dtype=np.int64, count=len(table))
                vals = np.stack([table[int(key)] for key in keys]).astype(np.float32)
            else:
                keys = np.zeros(0, dtype=np.int64)
                vals = np.zeros((0, N_LETTERS), dtype=np.float32)
            payload[f"keys_{k}"] = keys
            payload[f"vals_{k}"] = vals
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "MarkovPeptideModel":
        data = np.load(path)
        order = int(data["order"][0])
        tables = []
        for k in range(order + 1):
            keys = data[f"keys_{k}"]
            vals = data[f"vals_{k}"].astype(np.float64)
            tables.append({int(key): vals[i] for i, key in enumerate(keys)})
        return cls(
            order=order,
            tables=tables,
            length_pmf=data["length_pmf"],
            min_length=int(data["min_length"][0]),
        )

    # --- inference ----------------------------------------------------------
    def _next_distribution(self, context: list[int], alpha: float) -> np.ndarray:
        """Backoff distribution over the next residue.

        Interpolates from the longest available context down to the unigram,
        weighting each order by its observed count mass. `alpha` is a Dirichlet
        smoothing constant that keeps every residue reachable.
        """
        probs = np.full(N_LETTERS, alpha, dtype=np.float64)
        weight = 1.0
        for k in range(min(self.order, len(context)), -1, -1):
            vec = self.tables[k].get(_pack(context[len(context) - k :] if k else []))
            if vec is None:
                continue
            total = vec.sum()
            if total <= 0:
                continue
            probs += weight * vec
            # Longer contexts dominate; shorter ones fill in the tail.
            weight *= 0.15
        probs /= probs.sum()
        return probs

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        temperature: float = 1.0,
        alpha: float = 0.05,
        min_length: int = 8,
        max_length: int = 50,
        length_bias: np.ndarray | None = None,
    ) -> list[str]:
        """Draw `n` sequences. Deterministic given `rng`."""
        pmf = self.length_pmf if length_bias is None else length_bias
        offsets = np.arange(len(pmf)) + self.min_length
        keep = (offsets >= min_length) & (offsets <= max_length)
        pmf = np.where(keep, pmf, 0.0)
        pmf = pmf / pmf.sum()

        lengths = rng.choice(offsets, size=n, p=pmf)
        out: list[str] = []
        for target_len in lengths:
            context = [BOS] * self.order
            chars: list[str] = []
            for _ in range(int(target_len)):
                probs = self._next_distribution(context, alpha)
                if temperature != 1.0:
                    probs = probs ** (1.0 / temperature)
                    probs /= probs.sum()
                nxt = int(rng.choice(N_LETTERS, p=probs))
                chars.append(AMINO_ACIDS[nxt])
                context = context[1:] + [nxt]
            out.append("".join(chars))
        return out


def _pack(context: list[int]) -> int:
    """Pack a context of symbols in [0, 20] into a single integer key."""
    key = 1
    for symbol in context:
        key = key * 21 + (symbol + 1)
    return key
