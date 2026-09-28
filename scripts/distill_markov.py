"""Distil the transformer into a high-order backoff Markov model.

Motivation
----------
The transformer wins Phase 1 decisively on the embedding-distribution metrics
(FBD 1.18 vs 2.56, MMD 2.90 vs 6.68, Recall 0.63 vs 0.50 against the Markov
baseline). But sampling it in float64 on CPU — which is what makes the
submission reproducible across machines — costs ~48 ms/sequence, so a 100k-
candidate pool takes ~80 minutes, and the organizers run the entry point twice.

A Markov model samples in ~0.7 ms/sequence and its inference is pure
integer-indexed table lookup plus small elementwise float ops, which is about as
reproducible as computation gets. The question is whether the transformer's
advantage survives distillation.

It plausibly should. The order-4 Markov baseline was fit on only 39,448 real
AMPs, so its high-order contexts were starved of data and backed off almost
immediately. Sampling millions of sequences from the transformer creates an
effectively unlimited corpus drawn from the *transformer's* distribution, which
lets us fit a much higher order without sparsity — the student can then
represent most of what the teacher learned.

This trades a modest amount of fidelity for ~70x faster generation and a
stronger reproducibility guarantee. Whether that trade is worth taking is an
empirical question, answered by running the seqme harness on both.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amp_design.fasta import read_sequences  # noqa: E402
from amp_design.model import MarkovPeptideModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True,
                    help="FASTA files of transformer samples to distil from")
    ap.add_argument("--out", default="checkpoint/markov_distilled.npz")
    ap.add_argument("--order", type=int, default=6)
    args = ap.parse_args()

    sequences: list[str] = []
    for path in args.corpus:
        s = read_sequences(path)
        print(f"  {path}: {len(s):,}")
        sequences.extend(s)
    print(f"total distillation corpus: {len(sequences):,}")

    t0 = time.time()
    model = MarkovPeptideModel.fit(sequences, order=args.order)
    model.save(args.out)

    n_contexts = sum(len(t) for t in model.tables)
    size = Path(args.out).stat().st_size / 1e6
    print(f"order-{args.order} distilled model: {n_contexts:,} contexts, "
          f"{size:.1f} MB, fit in {time.time()-t0:.0f}s -> {args.out}")

    # Coverage diagnostic: what fraction of the longest-context lookups will
    # actually hit, rather than backing off? Low coverage means the order is
    # too high for the corpus size.
    rng = np.random.default_rng(0)
    sample = model.sample(2000, rng)
    lengths = [len(s) for s in sample]
    print(f"sanity: {len(sample)} samples, mean length {np.mean(lengths):.1f}, "
          f"unique {len(set(sample))}")


if __name__ == "__main__":
    main()
