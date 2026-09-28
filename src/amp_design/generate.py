"""Entry point: `uv run generate`.

Writes `generate/library.fasta` (50,000 sequences) and `generate/top.fasta`
(100 sequences). Output is byte-identical across runs for a fixed seed, which the
organizers' compliance script checks by running this twice and diffing the bytes.

Determinism notes
-----------------
* All randomness flows through a single `numpy.random.default_rng(seed)`.
* The default device is CPU. Even when a neural generator is installed, we do not
  sample on GPU for the official entry point: non-deterministic CUDA kernels can
  make two runs differ, and the organizers run on their own GPU workstation.
* FASTA is written with explicit "\\n" newlines so the bytes do not depend on the
  host platform.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import scoring
from .constants import (
    AMINO_ACID_SET,
    DEFAULT_SEED,
    LIBRARY_FILENAME,
    LIBRARY_SIZE,
    MAX_LENGTH,
    MIN_LENGTH,
    TOP_FILENAME,
    TOP_SIMILARITY_MARGIN,
    TOP_SIZE,
)
from .fasta import read_sequences, write_fasta
from .filters import ReferenceIndex, verify_top_similarity
from .model import MarkovPeptideModel
from .paths import find_resource
from .selection import greedy_diverse_top, stratified_diverse_selection

REFERENCE_FASTA = "data/antibacterial.fasta"
CHECKPOINT = "checkpoint/markov.npz"


def is_valid(seq: str) -> bool:
    return MIN_LENGTH <= len(seq) <= MAX_LENGTH and set(seq) <= AMINO_ACID_SET


def build_library(
    model: MarkovPeptideModel,
    index: ReferenceIndex,
    density: scoring.ConformityDensity,
    *,
    n_sequences: int,
    rng: np.random.Generator,
    oversample: float,
    verbose: bool = True,
) -> list[str]:
    """Sample, filter for compliance, then select a density-stratified subset."""
    target_pool = int(n_sequences * oversample)
    pool: list[str] = []
    seen: set[str] = set()

    # Sample in rounds until the compliant pool is large enough. Each round draws
    # from the same generator, so the whole loop stays deterministic.
    rounds = 0
    while len(pool) < target_pool and rounds < 60:
        rounds += 1
        batch = model.sample(
            n=max(target_pool // 4, 20_000),
            rng=rng,
            temperature=1.0,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
        )
        for seq in batch:
            if seq in seen or not is_valid(seq):
                continue
            if index.is_exact_match(seq):  # constraint 1: no verbatim reuse
                continue
            seen.add(seq)
            pool.append(seq)
        if verbose:
            print(f"  round {rounds}: pool = {len(pool):,} / {target_pool:,}", flush=True)

    if len(pool) < n_sequences:
        raise RuntimeError(
            f"Only produced {len(pool)} compliant sequences, need {n_sequences}."
        )

    if verbose:
        print(f"  scoring {len(pool):,} candidates by conformity density", flush=True)
    log_density = density.log_density(pool)

    chosen = stratified_diverse_selection(
        pool, log_density, n_select=n_sequences, rng=rng
    )
    library = [pool[i] for i in chosen]

    # Stable, seed-determined order.
    library.sort()
    return library


def select_top(
    library: list[str],
    index: ReferenceIndex,
    *,
    top_k: int,
    verbose: bool = True,
) -> list[str]:
    """Rank the library for wet-lab performance and enforce the 80% identity rule."""
    potency = scoring.potency_prior(library)
    selectivity = scoring.selectivity_prior(library)
    synth = scoring.synthesizability(library)

    # Weighted toward potency: four of the five competition categories are
    # activity-based and only one is selectivity-based. Synthesizability is a
    # gate, not a goal — a peptide that fails QC is simply never tested.
    composite = 0.55 * potency + 0.25 * selectivity + 0.20 * synth

    # Screen the strongest candidates against the reference database first, so
    # the expensive similarity check runs on a shortlist rather than 50,000.
    shortlist_size = min(len(library), max(top_k * 60, 3000))
    shortlist_idx = np.argsort(-composite)[:shortlist_size]

    if verbose:
        print(f"  similarity-screening {shortlist_size:,} shortlisted candidates", flush=True)

    passing_idx = [
        int(i)
        for i in shortlist_idx
        if index.max_similarity(library[int(i)]) <= TOP_SIMILARITY_MARGIN
    ]
    if len(passing_idx) < top_k:
        raise RuntimeError(
            f"Only {len(passing_idx)} candidates pass the similarity filter; need {top_k}."
        )

    sub_sequences = [library[i] for i in passing_idx]
    sub_scores = composite[passing_idx]
    picked_local = greedy_diverse_top(sub_sequences, sub_scores, n_select=top_k)
    top = [sub_sequences[i] for i in picked_local]

    # Exhaustive final check, exactly as the organizers run it.
    violations = verify_top_similarity(top, index.references)
    if violations:
        raise RuntimeError(f"Top list violates the similarity constraint: {violations[:3]}")

    return top


def main() -> None:
    entry_point = Path(sys.argv[0]).stem

    parser = argparse.ArgumentParser(description="Generate an AMP Challenge submission.")
    parser.add_argument("--n-sequences", type=int, default=LIBRARY_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--oversample", type=float, default=4.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    rng = np.random.default_rng(args.seed)

    reference_path = find_resource(REFERENCE_FASTA)
    checkpoint_path = find_resource(CHECKPOINT)

    if verbose:
        print(f"[1/5] Loading reference database: {reference_path}", flush=True)
    references = read_sequences(reference_path)
    index = ReferenceIndex(references)

    if verbose:
        print(f"[2/5] Loading model checkpoint: {checkpoint_path}", flush=True)
    model = MarkovPeptideModel.load(checkpoint_path)
    density = scoring.ConformityDensity(references, seed=args.seed)

    if verbose:
        print(f"[3/5] Building library of {args.n_sequences:,}", flush=True)
    library = build_library(
        model,
        index,
        density,
        n_sequences=args.n_sequences,
        rng=rng,
        oversample=args.oversample,
        verbose=verbose,
    )

    if verbose:
        print(f"[4/5] Selecting top {args.top_k}", flush=True)
    top = select_top(library, index, top_k=args.top_k, verbose=verbose)

    out_dir = Path(entry_point)
    write_fasta(library, out_dir / LIBRARY_FILENAME)
    write_fasta(top, out_dir / TOP_FILENAME)

    if verbose:
        print(f"[5/5] Wrote {len(library):,} -> {out_dir / LIBRARY_FILENAME}", flush=True)
        print(f"      Wrote {len(top)} -> {out_dir / TOP_FILENAME}", flush=True)


if __name__ == "__main__":
    main()
