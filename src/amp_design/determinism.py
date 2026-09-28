"""Cross-machine determinism helpers.

The competition checks reproducibility twice over:

* `verify_submission.py` runs the entry point twice on one machine and requires
  byte-identical output;
* the proposal states organizers regenerate the library "on a Linux workstation
  with a single GPU" and compare it against the submitted one.

The second check is the hard one. Same-machine repeatability is easy — seed the
RNG. Cross-machine repeatability is not: BLAS kernels reduce in different orders
depending on CPU vectorisation and thread count, so a float computed here can
differ from the same float computed there in the last few ulps. Any `argsort`
over such values can then reorder, and a ranked output flips.

The fix is to make every decision boundary coarse relative to that noise:

* `quantize` rounds scores to a fixed number of decimals before comparison, so
  differences below ~1e-6 cannot change an ordering;
* `stable_argsort` breaks remaining ties on the sequence string itself, which is
  exact and platform-independent.

Together these mean the pipeline only needs floats to agree to ~6 significant
decimals across machines — a bound BLAS comfortably meets — rather than bit-exactly.
"""

from __future__ import annotations

import numpy as np

SCORE_DECIMALS = 6
LOGIT_DECIMALS = 4


def quantize(values: np.ndarray, decimals: int = SCORE_DECIMALS) -> np.ndarray:
    """Round to a fixed grid so sub-ulp differences cannot flip comparisons."""
    return np.round(np.asarray(values, dtype=np.float64), decimals)


def quantize_logits(logits: np.ndarray) -> np.ndarray:
    """Coarser rounding for pre-softmax values, where absolute scale is larger."""
    return np.round(np.asarray(logits, dtype=np.float64), LOGIT_DECIMALS)


def stable_argsort(scores: np.ndarray, keys: list[str], descending: bool = True) -> np.ndarray:
    """Argsort by quantized score, breaking ties lexicographically on `keys`.

    Using the sequence string as the tiebreak makes the ordering a total order
    that depends on no floating-point detail beyond the quantized score.
    """
    q = quantize(scores)
    order = np.lexsort((np.array(keys, dtype=object), -q if descending else q))
    return order


def stable_unique_sorted(sequences: list[str]) -> list[str]:
    """Deterministic, platform-independent canonical ordering of a sequence set."""
    return sorted(set(sequences))
