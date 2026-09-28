"""Candidate scoring.

Two separate objectives, deliberately kept apart:

* `conformity_density` shapes the **50,000-sequence library**. It mirrors seqme's
  `ConformityScore`, which fits a Gaussian KDE over the reference AMPs'
  (amphiphilicity, charge) cloud and rewards generated points that sit in
  high-density regions. Because that metric scores *typicality*, not
  distribution-matching, the library wins by concentrating inside the reference
  mode while staying diverse in sequence space.

* `potency_prior` ranks the **top-100**, which is what actually goes to the wet lab.
  Here we want predicted MIC, not typicality.

The v0 `potency_prior` is a transparent, literature-derived physicochemical prior;
it is replaced by a trained oracle ensemble in `amp_design.oracles` once the models
are fitted. Keeping it as a fallback means the entry point never depends on a
checkpoint that might be missing.
"""

from __future__ import annotations

import numpy as np

from . import descriptors as D


class ConformityDensity:
    """Gaussian KDE over reference AMP (amphiphilicity, charge), Silverman bandwidth.

    Reimplements the density model inside seqme's `ConformityScore` so we can rank
    candidates by the exact quantity the competition measures, without importing
    scikit-learn at inference time.
    """

    def __init__(self, reference: list[str], subsample: int | None = 8000, seed: int = 0):
        amph = D.hydrophobic_moment(reference)
        chg = D.net_charge(reference)
        points = np.stack([amph, chg], axis=1)

        if subsample is not None and len(points) > subsample:
            rng = np.random.default_rng(seed)
            points = points[rng.choice(len(points), size=subsample, replace=False)]

        self.mean = points.mean(axis=0)
        self.std = points.std(axis=0)
        self.std[self.std == 0] = 1.0
        self.points = (points - self.mean) / self.std

        n, d = self.points.shape
        # Silverman's rule, as used by sklearn's KernelDensity(bandwidth="silverman").
        self.bandwidth = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))

    def log_density(self, sequences: list[str], block: int = 2048) -> np.ndarray:
        amph = D.hydrophobic_moment(sequences)
        chg = D.net_charge(sequences)
        query = (np.stack([amph, chg], axis=1) - self.mean) / self.std

        h2 = 2.0 * self.bandwidth**2
        ref_sq = (self.points**2).sum(axis=1)  # (n_ref,)
        out = np.empty(len(query), dtype=np.float64)

        for start in range(0, len(query), block):
            chunk = query[start : start + block]
            # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2, avoiding an (n, m, d) tensor.
            d2 = (chunk**2).sum(axis=1)[:, None] - 2.0 * (chunk @ self.points.T) + ref_sq[None, :]
            np.maximum(d2, 0.0, out=d2)
            m = -(d2 / h2)
            mx = m.max(axis=1, keepdims=True)
            out[start : start + block] = mx.squeeze(1) + np.log(np.exp(m - mx).sum(axis=1))
        return out


def potency_prior(sequences: list[str]) -> np.ndarray:
    """Physicochemical prior on antimicrobial potency, in [0, 1].

    Built from design rules that recur across the AMP literature for linear,
    membrane-active cationic peptides:

    * **Net charge +4 to +9.** Electrostatic attraction to anionic bacterial
      membranes (LPS / phosphatidylglycerol / cardiolipin) rises with charge, but
      beyond roughly +9 selectivity collapses and haemolysis rises.
    * **Hydrophobic moment high.** Amphipathicity — segregation of polar and
      apolar faces on a helical wheel — is the strongest single correlate of
      membrane disruption.
    * **Mean hydrophobicity moderate.** Too low and the peptide never inserts;
      too high and it aggregates, loses selectivity and lyses erythrocytes.
    * **Length 12-30.** Long enough to span or carpet the bilayer, short enough to
      synthesise cleanly and stay soluble.
    * **Low Cys / Met.** Free cysteines oxidise and dimerise; methionine oxidises.
      Both are liabilities under the competition's fixed synthesis workflow.
    """
    n = len(sequences)
    if n == 0:
        return np.zeros(0)

    charge = D.net_charge(sequences)
    amph = D.hydrophobic_moment(sequences)
    hyd = D.mean_eisenberg(sequences)
    lengths = np.array([len(s) for s in sequences], dtype=np.float64)
    comp = D.composition_matrix(sequences) / lengths[:, None]

    from .constants import AMINO_ACIDS

    idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    charge_score = _plateau(charge, 4.0, 9.0, width=2.5)
    amph_score = _sigmoid((amph - 0.45) / 0.12)
    hyd_score = _plateau(hyd, -0.10, 0.45, width=0.35)
    length_score = _plateau(lengths, 12.0, 30.0, width=6.0)

    cys_pen = np.clip(1.0 - 6.0 * comp[:, idx["C"]], 0.0, 1.0)
    met_pen = np.clip(1.0 - 4.0 * comp[:, idx["M"]], 0.0, 1.0)

    # Aromatic / Trp content helps membrane-interface anchoring in moderation.
    trp = comp[:, idx["W"]]
    trp_score = _plateau(trp, 0.02, 0.12, width=0.06)

    score = (
        0.30 * charge_score
        + 0.28 * amph_score
        + 0.16 * hyd_score
        + 0.10 * length_score
        + 0.08 * trp_score
        + 0.04 * cys_pen
        + 0.04 * met_pen
    )
    return np.clip(score, 0.0, 1.0)


def selectivity_prior(sequences: list[str]) -> np.ndarray:
    """Prior on the safety window (HC50 / MIC50), in [0, 1]; higher is safer.

    Haemolysis on human red blood cells tracks bulk hydrophobicity and aromatic
    content far more than net charge, because zwitterionic mammalian membranes
    lack the anionic headgroups that drive AMP selectivity. The practical design
    move is therefore to keep charge high while capping hydrophobicity.
    """
    hyd = D.mean_eisenberg(sequences)
    gravy = D.gravy(sequences)
    charge = D.net_charge(sequences)
    lengths = np.array([len(s) for s in sequences], dtype=np.float64)
    comp = D.composition_matrix(sequences) / lengths[:, None]

    from .constants import AMINO_ACIDS

    idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    bulky_aromatic = comp[:, idx["F"]] + comp[:, idx["W"]] + comp[:, idx["Y"]]
    aliphatic = comp[:, idx["L"]] + comp[:, idx["I"]] + comp[:, idx["V"]]

    hyd_pen = _sigmoid((0.55 - hyd) / 0.18)
    gravy_pen = _sigmoid((0.6 - gravy) / 0.5)
    aromatic_pen = _sigmoid((0.14 - bulky_aromatic) / 0.05)
    aliphatic_pen = _sigmoid((0.42 - aliphatic) / 0.10)
    charge_bonus = _sigmoid((charge - 3.0) / 2.0)

    return np.clip(
        0.30 * hyd_pen + 0.22 * gravy_pen + 0.20 * aromatic_pen + 0.16 * aliphatic_pen + 0.12 * charge_bonus,
        0.0,
        1.0,
    )


def synthesizability(sequences: list[str]) -> np.ndarray:
    """Solid-phase-synthesis feasibility prior, in [0, 1].

    The proposal scores "rate of sequences satisfying empirically derived
    synthesizability constraints". Known Fmoc-SPPS failure modes:
    long homopolymer runs, high beta-sheet propensity driving on-resin
    aggregation, multiple cysteines, Asn-Gly / Asp-Gly deamidation and
    aspartimide-prone motifs, and N-terminal glutamine cyclisation.
    """
    n = len(sequences)
    scores = np.ones(n, dtype=np.float64)

    for i, seq in enumerate(sequences):
        penalty = 0.0

        # Homopolymer runs.
        run, longest = 1, 1
        for a, b in zip(seq, seq[1:]):
            run = run + 1 if a == b else 1
            longest = max(longest, run)
        if longest >= 4:
            penalty += 0.12 * (longest - 3)

        # Cysteine count: >1 free Cys invites uncontrolled disulfide scrambling.
        n_cys = seq.count("C")
        if n_cys >= 2:
            penalty += 0.20 * (n_cys - 1)
        elif n_cys == 1:
            penalty += 0.05

        # Aspartimide-prone and deamidation-prone dipeptides.
        for motif in ("DG", "DS", "DN", "NG"):
            penalty += 0.05 * seq.count(motif)

        # N-terminal Gln cyclises to pyroglutamate.
        if seq.startswith("Q"):
            penalty += 0.08

        # Beta-branched / aggregation-prone stretches.
        window = 5
        for j in range(len(seq) - window + 1):
            chunk = seq[j : j + window]
            if sum(c in "VITYFW" for c in chunk) >= 4:
                penalty += 0.04

        scores[i] = max(0.0, 1.0 - penalty)

    return scores


# --- shaping helpers --------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _plateau(x: np.ndarray, lo: float, hi: float, width: float) -> np.ndarray:
    """1 inside [lo, hi], decaying smoothly outside over `width`."""
    below = _sigmoid((x - lo) / (width / 4.0))
    above = _sigmoid((hi - x) / (width / 4.0))
    return below * above
