"""The generation pipeline: pool -> library -> top-100.

Split out from `generate.py` so the entry point stays a thin CLI and each stage
can be tested and swept independently.

Stage rationale, all of it measured rather than assumed (see RESULTS.md):

1. **Sampling.** The generator is pluggable. Measurement decided which one:
   a conditional transformer beats the backoff Markov model on every
   embedding-distribution metric (FBD 1.18 vs 2.56, MMD 2.90 vs 6.68,
   Recall 0.63 vs 0.50), which is most of what Phase 1 scores.

2. **Compliance filtering.** Drop invalid, duplicate, and any sequence present
   verbatim in the reference database. The last one is a hard requirement on the
   *whole* library, not just the top-100.

3. **Library selection.** Density-temperature sampling. seqme's ConformityScore
   is a typicality statistic in 2-D (amphiphilicity, charge) space — the
   reference AMPs score ~0.50 against themselves — so beating 0.5 means
   concentrating toward the descriptor mode. Crucially that is a *different*
   space from the ESM embedding space where FBD/Recall/FKEA live, so
   concentrating in the former need not collapse the latter. `tau` is the knob;
   its value comes from the sweep in `eval/selection_sweep.py`.

4. **Top-100 selection.** The full oracle panel over the whole library, then a
   similarity screen against the reference on the top shortlist, then greedy
   diverse selection. The panel runs on all 50,000 rather than a cheap
   pre-gate: featurising the library costs about two minutes and the
   gradient-boosted predictions are microseconds per row, so gating would only
   have risked discarding good candidates. The similarity screen stays on a
   shortlist because it is O(candidates x 39,448) edit distances.
"""

from __future__ import annotations

import numpy as np

from . import scoring
from .constants import (
    AMINO_ACID_SET,
    MAX_LENGTH,
    MIN_LENGTH,
    TOP_SIMILARITY_MARGIN,
)
from .determinism import quantize, stable_argsort
from .filters import ReferenceIndex, verify_top_similarity
from .selection import greedy_diverse_top


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardise, tolerating a degenerate (constant) input."""
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def is_valid(seq: str) -> bool:
    return MIN_LENGTH <= len(seq) <= MAX_LENGTH and set(seq) <= AMINO_ACID_SET


def collect_pool(
    sampler,
    index: ReferenceIndex,
    *,
    target: int,
    rng: np.random.Generator,
    max_rounds: int = 40,
    verbose: bool = True,
) -> list[str]:
    """Sample until `target` compliant, unique, non-reference sequences exist.

    `sampler(n, rng)` returns a list of sequences. Sampling continues in rounds
    from the same generator, so the loop is deterministic given the seed.
    """
    pool: list[str] = []
    seen: set[str] = set()
    batch = max(target // 3, 10_000)

    for r in range(1, max_rounds + 1):
        if len(pool) >= target:
            break
        for seq in sampler(batch, rng):
            if seq in seen or not is_valid(seq):
                continue
            if index.is_exact_match(seq):
                continue
            seen.add(seq)
            pool.append(seq)
        if verbose:
            print(f"  round {r}: pool = {len(pool):,} / {target:,}", flush=True)

    if len(pool) < target:
        raise RuntimeError(f"pool exhausted at {len(pool):,}, wanted {target:,}")
    return pool


def select_library(
    pool: list[str],
    density: scoring.ConformityDensity,
    *,
    n_select: int,
    tau: float,
    rng: np.random.Generator,
    verbose: bool = True,
) -> list[str]:
    """Density-temperature selection, then a canonical ordering.

    `tau = 0` is uniform (maximum embedding spread, generator-level conformity);
    larger `tau` concentrates toward the descriptor mode, buying ConformityScore
    at the cost of FBD/Recall.
    """
    if n_select > len(pool):
        raise ValueError(f"cannot select {n_select:,} from a pool of {len(pool):,}")

    if verbose:
        print(f"  scoring {len(pool):,} candidates by conformity density", flush=True)
    log_density = density.log_density(pool)

    if n_select == len(pool):
        # Nothing to select; `np.argpartition` would also reject kth == len.
        idx = np.arange(len(pool))
    elif tau == 0.0:
        idx = rng.choice(len(pool), size=n_select, replace=False)
    else:
        w = quantize(tau * log_density)
        w = w - w.max()
        # Gumbel top-k: exact weighted sampling without replacement in one pass.
        keys = w + rng.gumbel(size=len(w))
        idx = np.argpartition(-keys, n_select)[:n_select]

    library = [pool[i] for i in idx]
    # Canonical order: sorting the strings makes the FASTA byte-identical
    # regardless of the order selection happened to produce.
    library.sort()
    if verbose:
        sel = log_density[idx]
        print(f"  selected {len(library):,} | mean log-density "
              f"{sel.mean():.3f} (pool {log_density.mean():.3f})", flush=True)
    return library


def select_top(
    library: list[str],
    index: ReferenceIndex,
    oracles,
    *,
    top_k: int,
    envelope=None,
    envelope_pool: int = 3000,
    gate_size: int | None = None,
    shortlist_size: int = 2000,
    verbose: bool = True,
) -> tuple[list[str], dict]:
    """Rank the library for wet-lab performance under the competition's scoring.

    The team score in each category is the arithmetic mean over 25 peptides
    drawn uniformly at random from this list, so every entry must be strong and
    low variance matters as much as a high mean. We therefore rank on expected
    success rate rather than on minimum predicted MIC, and enforce pairwise
    dissimilarity so a single wrong modelling assumption cannot sink the batch.
    """
    # --- stage 1: full oracle panel over the whole library -----------------
    # An earlier version gated to 6,000 by a cheap physicochemical prior first.
    # That was a false economy: featurising 50,000 sequences costs about two
    # minutes and the gradient-boosted predictions are microseconds per row, so
    # the whole library can be scored directly. Gating risked discarding strong
    # candidates the crude prior happened to rank low, which matters because
    # this list is what actually goes to the wet lab.
    gated = library if gate_size is None else [
        library[i] for i in stable_argsort(
            0.5 * scoring.potency_prior(library)
            + 0.2 * scoring.selectivity_prior(library)
            + 0.3 * scoring.synthesizability(library),
            library,
        )[:min(gate_size, len(library))]
    ]

    # --- stage 0: restrict to the measured-efficacy envelope ---------------
    # The envelope is our most transfer-robust signal, and it rests on 1,279
    # measured peptides rather than the 46 used to compare rankers. Using it as
    # a *filter* rather than one term in a sum is what actually moves the
    # selected profile onto the measured-active region: blended equally, the
    # oracle terms dragged amphiphilicity back to 0.34 against a 0.49-0.74
    # target. Robust signal defines the candidate set; the weaker learned
    # signals order within it.
    if envelope is not None and envelope_pool < len(gated):
        env_all = envelope.log_density(gated)
        keep = stable_argsort(env_all, gated)[:envelope_pool]
        gated = [gated[i] for i in keep]
        if verbose:
            print(f"  efficacy envelope: kept top {len(gated):,} of "
                  f"{len(library):,} by measured-active density", flush=True)

    if verbose:
        print(f"  running oracle panel on {len(gated):,} sequences "
              f"(10 species x MIC, HC50, AMP classifier)", flush=True)
    profile = oracles.panel_profile(gated)
    composite = oracles.composite_rank_score(gated, profile)

    # --- blend three signals with independent rationales --------------------
    # Ranked against 46 peptides with MICs measured on the real competition
    # panel (external, zero training overlap), the individual signals score
    # Spearman 0.241 (composite), 0.247 (envelope) and 0.406 (safety window),
    # and the equal-weight blend 0.470. Weights are deliberately *equal*
    # z-scores rather than fitted: n = 46 with many candidate combinations
    # tested is far too little to fit weights without overfitting, and the
    # standard error on any one of those correlations is about 0.15.
    #
    # Why blend rather than pick the best:
    #   * envelope  - population profile of peptides that measurably worked
    #                 broadly; makes no per-peptide claim, so it is the signal
    #                 least exposed to the distribution shift that sank our
    #                 MIC regressor externally (0.611 internal -> 0.062).
    #   * composite - MIC-trained, which BATTLE-AMP finds beats binary
    #                 classifiers regardless of architecture.
    #   * safety    - the selectivity axis, and our strongest single external
    #                 correlate. It is already inside `composite` at weight
    #                 0.22; including it again deliberately up-weights it.
    if envelope is not None:
        env_density = envelope.log_density(gated)
        blended = _zscore(env_density) + _zscore(composite) + _zscore(profile["safety_window"])
        if verbose:
            print(f"  blended ranking: envelope + composite + safety window", flush=True)
    else:
        blended = _zscore(composite)

    # Synthesizability is a gate on being testable at all: a peptide that fails
    # QC is never retested and becomes a dead slot in the 25-peptide draw.
    gated_synth = scoring.synthesizability(gated)
    blended = blended * (0.5 + 0.5 * gated_synth)

    short_idx = stable_argsort(blended, gated)[:min(shortlist_size, len(gated))]
    composite = blended

    # --- stage 2: similarity screen against the reference ------------------
    if verbose:
        print(f"  similarity-screening {len(short_idx):,} candidates", flush=True)
    passing = [int(i) for i in short_idx
               if not index.exceeds(gated[int(i)], TOP_SIMILARITY_MARGIN)]
    if len(passing) < top_k:
        raise RuntimeError(f"only {len(passing)} passed the similarity filter; need {top_k}")

    sub_seqs = [gated[i] for i in passing]
    sub_scores = composite[passing]

    # --- stage 3: greedy diverse selection ---------------------------------
    picked = greedy_diverse_top(sub_seqs, sub_scores, n_select=top_k)
    top = [sub_seqs[i] for i in picked]

    violations = verify_top_similarity(top, index.references)
    if violations:
        raise RuntimeError(f"top list violates the similarity constraint: {violations[:3]}")

    stats = {
        "mean_envelope_logdensity": float(
            envelope.log_density(top).mean()) if envelope is not None else float("nan"),
        "mean_success_overall": float(profile["success_overall"][passing][picked].mean()),
        "mean_amp_probability": float(profile["amp_probability"][passing][picked].mean()),
        "mean_pmic50": float(profile["pmic50"][passing][picked].mean()),
        "mean_safety_window": float(profile["safety_window"][passing][picked].mean()),
        "mean_synthesizability": float(gated_synth[passing][picked].mean()),
    }
    return top, stats
