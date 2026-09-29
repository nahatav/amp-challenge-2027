"""Build a large peptide corpus for pretraining the generator.

Why pretrain at all. The current generator is 4.8M parameters trained on 41,412
AMPs, and its train/val gap (1.61 vs 1.72) says it is already at the edge of
what that much data supports — so simply making it bigger would overfit, not
improve. The established fix is to pretrain on a broad peptide corpus and then
fine-tune on AMPs: PepBERT pretrains on 1.97-19.2M peptide sequences, and a
comparable baseline uses 1.7M UniProt sequences of at most 50 residues.
Hyformer's peptide setup pretrains on 3.5M general peptides plus 1M AMPs before
fine-tuning on 4,547 peptides with measured MIC.

The corpus is built from SwissProt (reviewed, ~570k proteins):

  * every protein already 8-50 residues — genuine short peptides, the closest
    natural match to our target length range;
  * sliding windows from longer proteins, which supplies the bulk. Windows are
    strided rather than exhaustive so no single large protein dominates.

Only the 20 canonical amino acids are kept, matching the competition alphabet.

Run:  uv run --extra dev python scripts/build_pretrain_corpus.py
"""

from __future__ import annotations

import argparse
import gzip
import random
from pathlib import Path

VALID = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 8, 50


def iter_fasta_gz(path: Path):
    header, parts = None, []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield "".join(parts)
                header, parts = line, []
            else:
                parts.append(line.upper())
    if header is not None:
        yield "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swissprot", default="../data/swissprot.fasta.gz")
    ap.add_argument("--out", default="../data/pretrain_peptides.txt")
    ap.add_argument("--target", type=int, default=2_500_000)
    ap.add_argument("--stride", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    out: list[str] = []
    n_proteins = n_short = 0

    for seq in iter_fasta_gz(Path(args.swissprot)):
        n_proteins += 1
        if not set(seq) <= VALID:
            # Drop sequences containing X/B/Z/U/O rather than repairing them.
            seq = "".join(c for c in seq if c in VALID)
            if len(seq) < MIN_LEN:
                continue

        if MIN_LEN <= len(seq) <= MAX_LEN:
            n_short += 1
            if seq not in seen:
                seen.add(seq)
                out.append(seq)
            continue

        # Strided windows of a randomised length inside the target range.
        for start in range(0, len(seq) - MIN_LEN, args.stride):
            L = rng.randint(MIN_LEN, MAX_LEN)
            frag = seq[start : start + L]
            if len(frag) < MIN_LEN:
                continue
            if frag not in seen:
                seen.add(frag)
                out.append(frag)
        if len(out) >= args.target:
            break

    rng.shuffle(out)
    out = out[: args.target]
    Path(args.out).write_text("\n".join(out), encoding="utf-8")

    lens = [len(s) for s in out]
    print(f"proteins scanned      : {n_proteins:,}")
    print(f"natural short peptides: {n_short:,}")
    print(f"corpus written        : {len(out):,} unique peptides -> {args.out}")
    print(f"length min/median/max : {min(lens)} / {sorted(lens)[len(lens)//2]} / {max(lens)}")


if __name__ == "__main__":
    main()
