"""Fit the baseline Markov generator on the reference AMP corpus.

Run:  uv run --extra dev python scripts/train_markov.py

The reference file `data/antibacterial.fasta` is the competition's own exclusion
list, and its headers carry activity annotations. We up-weight sequences annotated
as antibacterial and present in multiple source databases, on the reasoning that
multiply-curated entries are better-evidenced actives.

Note the generator never emits a verbatim training sequence into the submission:
`generate.py` filters every exact match before selection, which is also what the
compliance script requires.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amp_design.fasta import read_fasta  # noqa: E402
from amp_design.model import MarkovPeptideModel  # noqa: E402


def parse_header(header: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in ("len", "charge", "disulfide", "dbs", "activity"):
        m = re.search(rf"{key}=([^\s]+)", header)
        if m:
            fields[key] = m.group(1)
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/antibacterial.fasta")
    parser.add_argument("--out", default="checkpoint/markov.npz")
    parser.add_argument("--order", type=int, default=4)
    args = parser.parse_args()

    headers, sequences = read_fasta(args.reference)
    print(f"Loaded {len(sequences):,} reference sequences")

    weights = np.ones(len(sequences), dtype=np.float64)
    for i, header in enumerate(headers):
        fields = parse_header(header)
        n_dbs = len(fields.get("dbs", "").split("|")) if fields.get("dbs") else 1
        activities = fields.get("activity", "").split("|")

        # Multiply-curated entries carry more evidential weight.
        weights[i] *= 1.0 + 0.10 * min(n_dbs, 10)
        # Antibacterial annotation is the competition's actual target.
        if "antibacterial" in activities:
            weights[i] *= 1.6
        # Haemolytic/cytotoxic annotations are liabilities for the selectivity
        # category, so down-weight them.
        if any("toxic" in a or "hemolytic" in a for a in activities):
            weights[i] *= 0.5

    print(f"Weight range: {weights.min():.2f} - {weights.max():.2f}")

    model = MarkovPeptideModel.fit(sequences, order=args.order, weights=weights)
    model.save(args.out)

    size_mb = Path(args.out).stat().st_size / 1e6
    n_contexts = sum(len(t) for t in model.tables)
    print(f"Saved order-{args.order} model: {n_contexts:,} contexts, {size_mb:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
