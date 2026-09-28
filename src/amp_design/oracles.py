"""Trained oracles, loaded at inference to rank the top-100.

Three heads, fitted in `scripts/train_oracles.py`:

* `amp_probability`  — P(antimicrobial), gradient-boosted trees on 261 descriptors.
* `predicted_pmic`   — log10(MIC / uM) per species, conditioned on a species one-hot.
* `predicted_phc50`  — log10(HC50 / uM), censored at the 128 uM assay ceiling.

Everything downstream of these is shaped by how the competition actually scores
Phase 2:

* Team score per category is the **arithmetic mean over 25 peptides drawn
  uniformly at random from our top-100**. A single outstanding peptide is worth
  nothing; the whole list has to be good, and lowering variance matters as much
  as raising the mean.
* The potency threshold is **MIC <= 16 uM**, i.e. log10 <= 1.204, and Success
  Rate is the *fraction of strains* meeting it. So the objective is not minimum
  MIC — it is the probability of crossing a fixed threshold, across many strains.
  That makes expected success rate the right quantity to rank on, and it rewards
  breadth over depth.
* Safety Window is HC50/MIC50 with HC50 censored at 128 uM, so once a peptide is
  predicted non-haemolytic, further "safety" gains are unobtainable and only
  potency moves the score.

Gradient-boosted trees are exact piecewise-constant functions — leaf lookups and
float additions — so they are reproducible across machines, unlike anything
involving BLAS.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .features import featurize

# Panel species, mapped onto GRAMPA's naming. The competition tests 20 strains
# across these species; GRAMPA has no strain-level MIC for most, so we predict
# at species level and average.
PANEL_TO_GRAMPA = {
    "A. baumannii": 3,      # 3 strains on the panel (incl. 1 MDR)
    "E. cloacae": 1,
    "E. coli": 5,
    "K. pneumoniae": 2,
    "P. aeruginosa": 3,
    "S. typhimurium": 2,    # covers both S. enterica entries
    "B. subtilis": 1,
    "S. aureus": 2,
    "E. faecalis": 1,
    "E. faecium": 1,
}

GRAM_NEGATIVE = {"A. baumannii", "E. cloacae", "E. coli", "K. pneumoniae",
                 "P. aeruginosa", "S. typhimurium"}
GRAM_POSITIVE = {"B. subtilis", "S. aureus", "E. faecalis", "E. faecium"}

POTENCY_THRESHOLD_LOG10 = np.log10(16.0)   # MIC <= 16 uM
HC50_CEILING_LOG10 = np.log10(128.0)


class OracleEnsemble:
    def __init__(self, bundle: dict):
        self.clf = bundle["classifier"]["model"]
        mic = bundle["mic"]
        self.mic_model = mic["model"]
        self.mic_species = mic["species"]
        self.mic_n_features = mic["n_features"]
        self._sp_index = {s: i for i, s in enumerate(self.mic_species)}
        hemo = bundle.get("hemolysis") or {}
        self.hemo_model = hemo.get("model")

        self.panel = [s for s in PANEL_TO_GRAMPA if s in self._sp_index]

    @classmethod
    def load(cls, path: str | Path) -> "OracleEnsemble":
        with open(path, "rb") as fh:
            return cls(pickle.load(fh))

    # --- heads -------------------------------------------------------------
    def amp_probability(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def predicted_pmic(self, X: np.ndarray, species: str, chunk: int = 8192) -> np.ndarray:
        """log10(MIC / uM) for one species.

        Chunked because the species one-hot makes the design matrix wide (261
        descriptors plus ~680 species columns). Scoring a 50,000-sequence
        library in one block would allocate several hundred MB per species, ten
        times over; chunking keeps the peak bounded without changing the result.
        """
        n = X.shape[0]
        width = self.mic_n_features + len(self.mic_species)
        col = self.mic_n_features + self._sp_index[species]
        out = np.empty(n, dtype=np.float64)

        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            Z = np.zeros((stop - start, width), dtype=np.float64)
            Z[:, : self.mic_n_features] = X[start:stop, : self.mic_n_features]
            Z[:, col] = 1.0
            out[start:stop] = self.mic_model.predict(Z)
        return out

    def predicted_phc50(self, X: np.ndarray) -> np.ndarray:
        if self.hemo_model is None:
            return np.full(X.shape[0], HC50_CEILING_LOG10)
        return np.minimum(self.hemo_model.predict(X), HC50_CEILING_LOG10)

    # --- competition-shaped aggregates ------------------------------------
    def panel_profile(self, sequences: list[str]) -> dict[str, np.ndarray]:
        """Predict the full panel and derive the five category scores.

        Strain counts weight each species so that the predicted success rate
        approximates the competition's per-strain average rather than a
        per-species one.
        """
        X = featurize(sequences)
        n = len(sequences)

        pmic = {sp: self.predicted_pmic(X, sp) for sp in self.panel}
        weights = np.array([PANEL_TO_GRAMPA[sp] for sp in self.panel], dtype=np.float64)
        stack = np.stack([pmic[sp] for sp in self.panel], axis=1)   # (n, n_species)

        # Soft success indicator. A hard threshold discards the information that
        # a prediction sits just above or just below 16 uM, and our MIC MAE is
        # ~0.48 log units, so a sigmoid with that width is the honest form.
        margin = (POTENCY_THRESHOLD_LOG10 - stack) / 0.48
        success = 1.0 / (1.0 + np.exp(-np.clip(margin, -40, 40)))

        def weighted(mask_species):
            idx = [i for i, sp in enumerate(self.panel) if sp in mask_species]
            if not idx:
                return np.zeros(n)
            w = weights[idx]
            return (success[:, idx] * w).sum(axis=1) / w.sum()

        overall = (success * weights).sum(axis=1) / weights.sum()
        mic50 = np.median(stack, axis=1)
        phc50 = self.predicted_phc50(X)

        return {
            "amp_probability": self.amp_probability(X),
            "success_overall": overall,
            "success_gram_negative": weighted(GRAM_NEGATIVE),
            "success_gram_positive": weighted(GRAM_POSITIVE),
            "pmic50": mic50,
            "pmic_min": stack.min(axis=1),
            "pmic_spread": stack.std(axis=1),
            "phc50": phc50,
            # Safety window in log space: log10(HC50) - log10(MIC50).
            "safety_window": phc50 - mic50,
            "features": X,
        }

    def composite_rank_score(self, sequences: list[str], profile: dict | None = None) -> np.ndarray:
        """Single ranking score for the top-100.

        Weighting rationale: four of the five competition categories are
        activity-based (broad-spectrum, Gram-positive, Gram-negative, MDR) and
        one is selectivity-based, so MIC-derived quantities dominate. This also
        matches BATTLE-AMP (Szymczak et al. 2026, from the organizing lab),
        whose first conclusion is that models trained on MIC data outperform
        binary classifiers regardless of architecture.

        `amp_probability` enters as a **multiplicative gate, not an additive
        term**. Measured on GRAMPA: peptides with MIC <= 2 uM score 0.693 on our
        classifier while peptides with MIC >= 64 uM score 0.760 — i.e. among
        real AMPs the binary classifier is very slightly *anti*-correlated with
        potency, because both groups are database AMPs. Adding it linearly would
        reward the wrong thing. Its actual job is to catch sequences that are
        not AMP-like at all, where the MIC regressor extrapolates optimistically
        (it was fit only on peptides already believed active). A saturating gate
        does that: it penalises the implausible and is indifferent above ~0.45.
        """
        p = profile if profile is not None else self.panel_profile(sequences)

        # Rescale the safety window into [0, 1] over a plausible range
        # (0 to 2.5 log units, i.e. HC50/MIC50 from 1x to ~300x).
        sw = np.clip(p["safety_window"] / 2.5, 0.0, 1.0)
        # Penalise across-strain inconsistency: the random 25-peptide draw means
        # variance is a real cost, and a peptide potent against only one species
        # scores poorly on success rate anyway.
        consistency = 1.0 / (1.0 + p["pmic_spread"])

        merit = (
            0.52 * p["success_overall"]
            + 0.22 * sw
            + 0.14 * consistency
            + 0.12 * np.clip((1.5 - p["pmic50"]) / 2.5, 0.0, 1.0)
        )

        gate = 0.55 + 0.45 / (
            1.0 + np.exp(-np.clip((p["amp_probability"] - 0.30) / 0.08, -40, 40))
        )
        return merit * gate
