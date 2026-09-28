"""Feature extraction for the activity / toxicity oracles.

Design choice: handcrafted physicochemical descriptors rather than protein-language-model
embeddings. Three independent lines of evidence support this for our setting:

* Pal et al. 2026 ("Coarse composition suffices") report that ten global
  physicochemical scalars recover 91% of full-feature performance on the 82k-peptide
  ESCAPE benchmark, and that predicted structure is unnecessary at inference.
* AMPBench-MT (2026) finds classical gradient-boosted regressors on handcrafted
  features (CatBoost, MAE 0.521) essentially level with the best protein-LM
  embeddings (ESM-C 300M, MAE 0.504) on pMIC regression.
* PathoMIC (2026) finds knowledge-guided features generalise better than purely
  data-driven representations to *unseen* species — which is our situation, since
  the competition panel contains strains absent from any public MIC table.

Descriptors are computed in NumPy with no external dependency, so the entry point
stays lightweight and deterministic.

Feature blocks
--------------
1. Amino-acid composition (20)
2. Global physicochemical scalars (13)
3. Grouped composition / transition / distribution — CTD (fixed property groupings)
4. Sliding-window extremes of hydrophobicity and charge density
5. Helical-wheel amphipathicity at both alpha (100 deg) and beta (180 deg) geometry
6. Terminal-region composition (first/last 5 residues)
"""

from __future__ import annotations

import numpy as np

from . import descriptors as D
from .constants import AMINO_ACIDS

# --- property groupings for CTD descriptors --------------------------------
# Standard three-way partitions used across the AMP prediction literature.
CTD_GROUPS: dict[str, tuple[str, str, str]] = {
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "vdw_volume": ("GASTPDC", "NVEQIL", "MHKFRYW"),
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "ss_propensity": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    "solvent_access": ("ALFCGIVW", "RKQEND", "MPSTHY"),
}

_AA_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
_EISENBERG = np.array([D.EISENBERG[a] for a in AMINO_ACIDS])
_CHARGE_VEC = np.array([1.0 if a in "KR" else (0.5 if a == "H" else (-1.0 if a in "DE" else 0.0))
                        for a in AMINO_ACIDS])


def _ctd_block(seq: str) -> np.ndarray:
    """Composition, transition and distribution for one property partition set."""
    out: list[float] = []
    n = len(seq)
    for groups in CTD_GROUPS.values():
        labels = np.empty(n, dtype=np.int8)
        for i, ch in enumerate(seq):
            for g, members in enumerate(groups):
                if ch in members:
                    labels[i] = g
                    break
            else:
                labels[i] = 1  # neutral fallback

        # Composition: fraction in each group.
        comp = np.bincount(labels, minlength=3) / n
        out.extend(comp.tolist())

        # Transition: frequency of adjacent group switches.
        if n > 1:
            pairs = labels[:-1] * 3 + labels[1:]
            counts = np.bincount(pairs, minlength=9).reshape(3, 3)
            trans = [
                (counts[0, 1] + counts[1, 0]) / (n - 1),
                (counts[0, 2] + counts[2, 0]) / (n - 1),
                (counts[1, 2] + counts[2, 1]) / (n - 1),
            ]
        else:
            trans = [0.0, 0.0, 0.0]
        out.extend(trans)

        # Distribution: relative position of the 25/50/75/100th percentile residue.
        for g in range(3):
            pos = np.flatnonzero(labels == g)
            if pos.size == 0:
                out.extend([0.0, 0.0, 0.0, 0.0])
            else:
                for q in (0.25, 0.5, 0.75, 1.0):
                    k = max(int(np.ceil(q * pos.size)) - 1, 0)
                    out.append((pos[k] + 1) / n)
    return np.asarray(out, dtype=np.float64)


def _window_block(seq: str, window: int = 7) -> np.ndarray:
    """Extremes of local hydrophobicity and charge density.

    Membrane insertion depends on a contiguous apolar face, and bacterial-surface
    binding on a contiguous cationic patch. Both are local, so global means miss
    them; these capture the strongest local stretch.
    """
    idx = np.fromiter((_AA_IDX[c] for c in seq), dtype=np.int64, count=len(seq))
    h = _EISENBERG[idx]
    q = _CHARGE_VEC[idx]
    w = min(window, len(seq))
    hv = np.lib.stride_tricks.sliding_window_view(h, w).mean(axis=1)
    qv = np.lib.stride_tricks.sliding_window_view(q, w).mean(axis=1)
    return np.array(
        [hv.max(), hv.min(), hv.std(), qv.max(), qv.min(), qv.std()],
        dtype=np.float64,
    )


def _terminal_block(seq: str, k: int = 5) -> np.ndarray:
    """Charge and hydrophobicity of the N- and C-terminal regions."""
    head, tail = seq[:k], seq[-k:]
    out = []
    for part in (head, tail):
        idx = np.fromiter((_AA_IDX[c] for c in part), dtype=np.int64, count=len(part))
        out.extend([_EISENBERG[idx].mean(), _CHARGE_VEC[idx].sum()])
    return np.asarray(out, dtype=np.float64)


# --- order-sensitive blocks ------------------------------------------------
# Reduced 7-letter alphabet (Murphy et al.) keeps dipeptide counts dense enough
# to estimate from short sequences: 49 features instead of 400.
REDUCED_ALPHABET = {
    **{c: 0 for c in "LVIMC"},   # aliphatic / hydrophobic
    **{c: 1 for c in "AG"},      # small
    **{c: 2 for c in "ST"},      # polar small
    **{c: 3 for c in "P"},       # proline (helix breaker)
    **{c: 4 for c in "FYW"},     # aromatic
    **{c: 5 for c in "EDNQ"},    # acidic / amide
    **{c: 6 for c in "KRH"},     # basic
}
_N_REDUCED = 7

_KD = np.array([D.KYTE_DOOLITTLE[a] for a in AMINO_ACIDS])
_VOLUME = np.array([88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7,
                    162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 227.8, 193.6])
_HELIX_PROP = np.array([1.42, 0.70, 1.01, 1.51, 1.13, 0.57, 1.00, 1.08, 1.16, 1.21,
                        1.45, 0.67, 0.57, 1.11, 0.98, 0.77, 0.83, 1.06, 1.08, 0.69])

_SCALES = [_EISENBERG, _KD, _CHARGE_VEC, (_VOLUME - _VOLUME.mean()) / _VOLUME.std(),
           (_HELIX_PROP - _HELIX_PROP.mean()) / _HELIX_PROP.std()]
_MAX_LAG = 8


def _autocorrelation_block(seq: str) -> np.ndarray:
    """Normalised Moreau-Broto autocorrelation of each property scale, lags 1..8.

    This is the block that separates a real AMP from a shuffled one. Amphipathic
    helices place hydrophobic residues roughly every 3-4 positions, so the lag-3
    and lag-4 terms carry the periodic signal that shuffling destroys while
    leaving composition untouched.
    """
    idx = np.fromiter((_AA_IDX[c] for c in seq), dtype=np.int64, count=len(seq))
    n = len(idx)
    out = np.zeros(len(_SCALES) * _MAX_LAG, dtype=np.float64)
    pos = 0
    for scale in _SCALES:
        v = scale[idx]
        for lag in range(1, _MAX_LAG + 1):
            out[pos] = float((v[:-lag] * v[lag:]).mean()) if n > lag else 0.0
            pos += 1
    return out


def _reduced_dipeptide_block(seq: str) -> np.ndarray:
    """Dipeptide frequencies over the 7-letter reduced alphabet."""
    codes = np.fromiter((REDUCED_ALPHABET[c] for c in seq), dtype=np.int64, count=len(seq))
    out = np.zeros(_N_REDUCED * _N_REDUCED, dtype=np.float64)
    if len(codes) > 1:
        pairs = codes[:-1] * _N_REDUCED + codes[1:]
        counts = np.bincount(pairs, minlength=_N_REDUCED * _N_REDUCED)
        out = counts / counts.sum()
    return out


def _amphipathic_block(seq: str) -> np.ndarray:
    """Longest contiguous stretch with sustained helical amphipathicity.

    Slides a window and records how long a run of windows keeps a hydrophobic
    moment above the sequence median — a proxy for the length of the
    membrane-inserting face, which matters more than the global average.
    """
    idx = np.fromiter((_AA_IDX[c] for c in seq), dtype=np.int64, count=len(seq))
    h = _EISENBERG[idx]
    w = min(9, len(seq))
    rads = np.deg2rad(100) * np.arange(w)
    views = np.lib.stride_tricks.sliding_window_view(h, w)
    # Elementwise rather than matvec, for machine-independent floats (see
    # amp_design.determinism).
    vcos = (views * np.cos(rads)).sum(axis=1)
    vsin = (views * np.sin(rads)).sum(axis=1)
    mom = np.sqrt(vcos**2 + vsin**2) / w
    if mom.size == 0:
        return np.zeros(3)
    thresh = np.median(mom)
    run = best = 0
    for m in mom:
        run = run + 1 if m >= thresh else 0
        best = max(best, run)
    return np.array([best / max(len(seq), 1), mom.max(), mom.mean()], dtype=np.float64)


def featurize(sequences: list[str]) -> np.ndarray:
    """Return the (n_sequences, n_features) design matrix."""
    if not sequences:
        return np.zeros((0, 0))

    lengths = np.array([len(s) for s in sequences], dtype=np.float64)
    comp = D.composition_matrix(sequences) / lengths[:, None]

    globals_block = np.stack(
        [
            lengths,
            D.net_charge(sequences),
            D.net_charge(sequences) / lengths,          # charge density
            D.hydrophobic_moment(sequences, modality="mean"),
            D.hydrophobic_moment(sequences, modality="max"),
            D.hydrophobic_moment(sequences, angle=180, modality="mean"),  # beta geometry
            D.mean_eisenberg(sequences),
            D.gravy(sequences),
            D.aliphatic_index(sequences),
            D.isoelectric_point(sequences),
            comp[:, [_AA_IDX[a] for a in "FWY"]].sum(axis=1),   # aromaticity
            comp[:, [_AA_IDX[a] for a in "KR"]].sum(axis=1),    # cationicity
            comp[:, [_AA_IDX[a] for a in "DE"]].sum(axis=1),    # anionicity
        ],
        axis=1,
    )

    ctd = np.stack([_ctd_block(s) for s in sequences])
    win = np.stack([_window_block(s) for s in sequences])
    term = np.stack([_terminal_block(s) for s in sequences])
    auto = np.stack([_autocorrelation_block(s) for s in sequences])
    dipep = np.stack([_reduced_dipeptide_block(s) for s in sequences])
    amph = np.stack([_amphipathic_block(s) for s in sequences])

    return np.concatenate(
        [comp, globals_block, ctd, win, term, auto, dipep, amph], axis=1
    )


def feature_names() -> list[str]:
    names = [f"comp_{a}" for a in AMINO_ACIDS]
    names += [
        "length", "charge", "charge_density",
        "hmoment_mean", "hmoment_max", "hmoment_beta",
        "eisenberg_mean", "gravy", "aliphatic_index", "pI",
        "aromaticity", "cationicity", "anionicity",
    ]
    for prop in CTD_GROUPS:
        names += [f"ctd_{prop}_C{g}" for g in range(3)]
        names += [f"ctd_{prop}_T{t}" for t in ("01", "02", "12")]
        for g in range(3):
            names += [f"ctd_{prop}_D{g}_{q}" for q in (25, 50, 75, 100)]
    names += ["win_h_max", "win_h_min", "win_h_std", "win_q_max", "win_q_min", "win_q_std"]
    names += ["nterm_h", "nterm_q", "cterm_h", "cterm_q"]
    for si in range(len(_SCALES)):
        names += [f"autocorr_s{si}_lag{lag}" for lag in range(1, _MAX_LAG + 1)]
    names += [f"dipep_{a}{b}" for a in range(_N_REDUCED) for b in range(_N_REDUCED)]
    names += ["amph_run_frac", "amph_max", "amph_mean"]
    return names
