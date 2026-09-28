"""Physicochemical descriptors, NumPy-only.

These reimplement the two descriptors that seqme's `ConformityScore` is configured
with in the organizers' own benchmarking notebook:

    ConformityScore(reference=AMPs,
                    predictors=[HydrophobicMoment(), Charge()],
                    kde_bandwidth="silverman")

seqme delegates both to `modlamp`. We reimplement them here so that inference has
no heavy dependency, and validate against modlamp in
`scripts/validate_descriptors.py` (dev extra).

`modlamp` reference behaviour being matched:
  * GlobalDescriptor.calculate_charge(ph=7.0, amide=False) — Henderson-Hasselbalch
    over Bjellqvist-style pKa values, including terminal groups.
  * PeptideDescriptor(scale="eisenberg").calculate_moment(window=11, angle=100,
    modality="mean") — sliding-window Eisenberg hydrophobic moment, normalised by
    window length.
"""

from __future__ import annotations

import numpy as np

from .constants import AMINO_ACIDS

# --- Eisenberg consensus hydrophobicity scale (modlamp "eisenberg") ---------
EISENBERG = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
    "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
    "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53,
    "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}

# --- Kyte-Doolittle, for GRAVY ---------------------------------------------
KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# --- pKa values used by modlamp's calculate_charge --------------------------
POS_PKS = {"Nterm": 9.38, "K": 10.67, "R": 12.10, "H": 6.04}
NEG_PKS = {"Cterm": 2.15, "D": 3.71, "E": 4.15, "C": 8.14, "Y": 10.10}

_AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
_EISENBERG_VEC = np.array([EISENBERG[aa] for aa in AMINO_ACIDS], dtype=np.float64)
_KD_VEC = np.array([KYTE_DOOLITTLE[aa] for aa in AMINO_ACIDS], dtype=np.float64)


def encode(sequences: list[str]) -> list[np.ndarray]:
    """Map each sequence to an int array of amino-acid indices."""
    return [np.fromiter((_AA_INDEX[c] for c in seq), dtype=np.int64, count=len(seq)) for seq in sequences]


def composition_matrix(sequences: list[str]) -> np.ndarray:
    """(n_sequences, 20) matrix of amino-acid counts."""
    out = np.zeros((len(sequences), len(AMINO_ACIDS)), dtype=np.float64)
    for row, idx in enumerate(encode(sequences)):
        np.add.at(out[row], idx, 1.0)
    return out


def net_charge(sequences: list[str], ph: float = 7.0, amide: bool = False) -> np.ndarray:
    """Net charge at a given pH, matching modlamp's `calculate_charge`."""
    counts = composition_matrix(sequences)
    n = len(sequences)
    charges = np.zeros(n, dtype=np.float64)

    # Positively ionisable side chains + N-terminus.
    for aa, pk in POS_PKS.items():
        if aa == "Nterm":
            c = np.ones(n, dtype=np.float64)
        else:
            c = counts[:, _AA_INDEX[aa]]
        charges += c * (1.0 / (1.0 + 10.0 ** (ph - pk)))

    # Negatively ionisable side chains + C-terminus.
    cterm_pk = 15.0 if amide else NEG_PKS["Cterm"]
    for aa, pk in NEG_PKS.items():
        if aa == "Cterm":
            c = np.ones(n, dtype=np.float64)
            pk = cterm_pk
        else:
            c = counts[:, _AA_INDEX[aa]]
        charges -= c * (1.0 / (1.0 + 10.0 ** (pk - ph)))

    return charges


def hydrophobic_moment(
    sequences: list[str],
    window: int = 11,
    angle: int = 100,
    modality: str = "mean",
) -> np.ndarray:
    """Eisenberg hydrophobic moment (amphiphilicity), matching modlamp.

    For every sliding window of length `w = min(window, len(seq))`, project the
    per-residue hydrophobicity onto a helical wheel of `angle` degrees per
    residue and take the vector magnitude, normalised by `w`. Aggregate windows
    by `modality` ("mean" or "max").
    """
    out = np.empty(len(sequences), dtype=np.float64)
    for i, idx in enumerate(encode(sequences)):
        h = _EISENBERG_VEC[idx]
        n = h.shape[0]
        w = min(window, n)
        rads = np.deg2rad(angle) * np.arange(w)
        cos_w, sin_w = np.cos(rads), np.sin(rads)

        # Vectorised sliding windows. Deliberately an elementwise multiply plus a
        # NumPy reduction rather than a matvec: BLAS chooses its reduction order
        # from CPU vectorisation and thread count, so `views @ cos_w` can differ
        # in the last ulps between machines. The competition regenerates our
        # library on their hardware and compares it to ours, so every float in
        # the generation path must be machine-independent.
        views = np.lib.stride_tricks.sliding_window_view(h, w)  # (n - w + 1, w)
        vcos = (views * cos_w).sum(axis=1)
        vsin = (views * sin_w).sum(axis=1)
        moments = np.sqrt(vcos**2 + vsin**2) / w

        out[i] = moments.mean() if modality == "mean" else moments.max()
    return out


def gravy(sequences: list[str]) -> np.ndarray:
    """Mean Kyte-Doolittle hydropathy."""
    return np.array([_KD_VEC[idx].mean() for idx in encode(sequences)], dtype=np.float64)


def mean_eisenberg(sequences: list[str]) -> np.ndarray:
    """Mean Eisenberg hydrophobicity (seqme's `Hydrophobicity(scale='eisenberg')`)."""
    return np.array([_EISENBERG_VEC[idx].mean() for idx in encode(sequences)], dtype=np.float64)


def aliphatic_index(sequences: list[str]) -> np.ndarray:
    """Ikai aliphatic index: relative volume of aliphatic side chains."""
    counts = composition_matrix(sequences)
    lengths = counts.sum(axis=1)
    a = counts[:, _AA_INDEX["A"]] / lengths
    v = counts[:, _AA_INDEX["V"]] / lengths
    i_ = counts[:, _AA_INDEX["I"]] / lengths
    l_ = counts[:, _AA_INDEX["L"]] / lengths
    return 100.0 * (a + 2.9 * v + 3.9 * (i_ + l_))


def isoelectric_point(sequences: list[str], lo: float = 0.0, hi: float = 14.0) -> np.ndarray:
    """pI by bisection on the net-charge curve."""
    lo_arr = np.full(len(sequences), lo)
    hi_arr = np.full(len(sequences), hi)
    for _ in range(60):
        mid = 0.5 * (lo_arr + hi_arr)
        q = _charge_at(sequences, mid)
        positive = q > 0
        lo_arr = np.where(positive, mid, lo_arr)
        hi_arr = np.where(positive, hi_arr, mid)
    return 0.5 * (lo_arr + hi_arr)


def _charge_at(sequences: list[str], ph: np.ndarray) -> np.ndarray:
    counts = composition_matrix(sequences)
    n = len(sequences)
    charges = np.zeros(n, dtype=np.float64)
    for aa, pk in POS_PKS.items():
        c = np.ones(n) if aa == "Nterm" else counts[:, _AA_INDEX[aa]]
        charges += c * (1.0 / (1.0 + 10.0 ** (ph - pk)))
    for aa, pk in NEG_PKS.items():
        c = np.ones(n) if aa == "Cterm" else counts[:, _AA_INDEX[aa]]
        charges -= c * (1.0 / (1.0 + 10.0 ** (pk - ph)))
    return charges


def descriptor_frame(sequences: list[str]) -> dict[str, np.ndarray]:
    """All descriptors at once, as a dict of arrays."""
    return {
        "length": np.array([len(s) for s in sequences], dtype=np.float64),
        "charge": net_charge(sequences),
        "amphiphilicity": hydrophobic_moment(sequences),
        "hydrophobicity": mean_eisenberg(sequences),
        "gravy": gravy(sequences),
        "aliphatic_index": aliphatic_index(sequences),
        "isoelectric_point": isoelectric_point(sequences),
    }
