"""Local mirror of the organizers' `verify_submission.py`, minus the git clone.

Runs every compliance check against the files already in `generate/`, so we can
iterate without pushing to GitHub on each change. The real verifier additionally
clones the repo, runs `uv sync`, and executes the entry point twice; use
`--check-reproducibility` here to reproduce that part locally.

Run:  uv run python scripts/selfcheck.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Levenshtein import ratio as lev_ratio  # noqa: E402

from amp_design.constants import (  # noqa: E402
    AMINO_ACID_SET,
    LIBRARY_SIZE,
    MAX_LENGTH,
    MAX_TOP_SIMILARITY,
    MIN_LENGTH,
    TOP_SIZE,
)
from amp_design.fasta import read_fasta  # noqa: E402
from amp_design.filters import ReferenceIndex  # noqa: E402

FAIL = "FAIL"
PASS = "ok"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status:>4}] {label}{(' — ' + detail) if detail else ''}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="generate")
    parser.add_argument("--reference", default="data/antibacterial.fasta")
    parser.add_argument("--check-reproducibility", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    library_path = out_dir / "library.fasta"
    top_path = out_dir / "top.fasta"

    ok = True
    print("Library")
    headers, library = read_fasta(library_path)
    ok &= check("exactly 50,000 records", len(library) == LIBRARY_SIZE, f"{len(library):,}")
    ok &= check("all headers non-empty", all(h.strip() for h in headers))
    bad_alpha = [s for s in library if not set(s) <= AMINO_ACID_SET]
    ok &= check("alphabet is the 20 canonical AAs", not bad_alpha, f"{len(bad_alpha)} bad")
    bad_len = [s for s in library if not (MIN_LENGTH <= len(s) <= MAX_LENGTH)]
    ok &= check("lengths within 8–50", not bad_len, f"{len(bad_len)} bad")
    ok &= check("no duplicate sequences", len(set(library)) == len(library),
                f"{len(library) - len(set(library))} dupes")

    print("Reference overlap")
    _, references = read_fasta(args.reference)
    ref_set = set(references)
    overlap = set(library) & ref_set
    ok &= check("zero exact matches vs antibacterial.fasta", not overlap, f"{len(overlap)} overlapping")

    print("Top list")
    _, top = read_fasta(top_path)
    ok &= check("exactly 100 records", len(top) == TOP_SIZE, f"{len(top)}")
    ok &= check("no duplicates", len(set(top)) == len(top))
    lib_set = set(library)
    missing = [s for s in top if s not in lib_set]
    ok &= check("every entry is in the library", not missing, f"{len(missing)} missing")

    print("Similarity (exhaustive, 100 x 39,448)")
    worst = 0.0
    worst_pair = ("", "")
    violations = 0
    for seq in top:
        for ref in references:
            r = lev_ratio(seq, ref)
            if r > worst:
                worst, worst_pair = r, (seq, ref)
            if r > MAX_TOP_SIMILARITY:
                violations += 1
                break
    ok &= check(
        f"max Levenshtein ratio <= {MAX_TOP_SIMILARITY}",
        violations == 0,
        f"worst = {worst:.3f} ({violations} violating)",
    )
    print(f"         worst pair: {worst_pair[0]}")
    print(f"                  vs {worst_pair[1]}")

    if args.check_reproducibility:
        print("Reproducibility")
        before = (library_path.read_bytes(), top_path.read_bytes())
        subprocess.run([sys.executable, "-m", "amp_design.generate", "--quiet"], check=True)
        after = (library_path.read_bytes(), top_path.read_bytes())
        ok &= check("library byte-identical across runs", before[0] == after[0])
        ok &= check("top byte-identical across runs", before[1] == after[1])

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
