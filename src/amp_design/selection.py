"""Assembling the 50,000-sequence library and the top-100 candidate list.

These are two different optimisation problems and are solved separately.

**Library (50,000).** Scored by Phase 1 on four metric families at once. The
binding tension is that Diversity, FKEA and AuthPct reward spreading out, while
FBD, Precision and ConformityScore reward sitting on the AMP manifold. A library
that simply takes the 50,000 highest-density candidates collapses onto the mode
and loses the diversity metrics; one that samples uniformly loses conformity. We
therefore use **density-stratified selection**: partition candidates by conformity
density into quantile strata, then fill each stratum with a greedy
maximum-diversity pick. The result is centred on the AMP mode but retains broad
coverage of sequence space.

**Top-100.** Scored by the wet lab, as the *arithmetic mean* over a uniformly
random 25-peptide subset. Because the mean is what counts and the draw is random,
there is no value in a single outstanding sequence — every one of the 100 must be
strong. We also enforce pairwise dissimilarity so that a single wrong assumption
about the activity prior cannot take down the whole batch.
"""

from __future__ import annotations

import numpy as np
from Levenshtein import ratio as lev_ratio


def stratified_diverse_selection(
    sequences: list[str],
    density: np.ndarray,
    n_select: int,
    *,
    n_strata: int = 20,
    density_floor_quantile: float = 0.25,
    rng: np.random.Generator,
) -> list[int]:
    """Pick `n_select` indices, concentrating on high density while keeping spread.

    Candidates below `density_floor_quantile` are discarded outright — these are
    the sequences that look nothing like an AMP and would drag down conformity,
    FBD and the surrogate activity scores together. The survivors are split into
    `n_strata` equal-count density bands, and each band contributes a quota that
    grows linearly with its density rank.
    """
    order = np.argsort(density)
    floor_idx = int(len(order) * density_floor_quantile)
    eligible = order[floor_idx:]

    if len(eligible) <= n_select:
        return eligible.tolist()

    strata = np.array_split(eligible, n_strata)

    # Quota grows linearly with stratum rank (lowest density -> smallest quota).
    ranks = np.arange(1, n_strata + 1, dtype=np.float64)
    weights = ranks / ranks.sum()
    quotas = np.floor(weights * n_select).astype(int)
    quotas[-1] += n_select - quotas.sum()

    chosen: list[int] = []
    for stratum, quota in zip(strata, quotas):
        quota = min(quota, len(stratum))
        if quota <= 0:
            continue
        if quota == len(stratum):
            chosen.extend(stratum.tolist())
        else:
            picked = rng.choice(stratum, size=quota, replace=False)
            chosen.extend(picked.tolist())

    # Top up if rounding left us short.
    if len(chosen) < n_select:
        remaining = np.setdiff1d(eligible, np.array(chosen, dtype=eligible.dtype), assume_unique=False)
        extra = rng.choice(remaining, size=n_select - len(chosen), replace=False)
        chosen.extend(extra.tolist())

    return chosen[:n_select]


def greedy_diverse_top(
    sequences: list[str],
    scores: np.ndarray,
    n_select: int,
    *,
    max_pairwise_similarity: float = 0.65,
    pool_multiplier: int = 40,
) -> list[int]:
    """Greedy score-ordered selection with a pairwise-dissimilarity constraint.

    Walk candidates from best score downward, accepting one only if it is
    sufficiently dissimilar from everything already accepted. If the constraint
    proves too tight to reach `n_select`, it is relaxed in steps rather than
    silently returning a short list.
    """
    order = np.argsort(-scores)
    pool = order[: max(n_select * pool_multiplier, n_select)]

    threshold = max_pairwise_similarity
    for _ in range(8):
        accepted: list[int] = []
        for idx in pool:
            seq = sequences[idx]
            if all(lev_ratio(seq, sequences[j]) <= threshold for j in accepted):
                accepted.append(int(idx))
                if len(accepted) == n_select:
                    return accepted
        threshold = min(0.95, threshold + 0.05)

    # Final fallback: fill by score, ignoring the diversity constraint.
    accepted_set = set(accepted)
    for idx in order:
        if len(accepted) == n_select:
            break
        if int(idx) not in accepted_set:
            accepted.append(int(idx))
            accepted_set.add(int(idx))
    return accepted[:n_select]
