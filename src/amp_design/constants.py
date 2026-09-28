"""Competition constants and hard constraints.

Every value here is taken directly from the AMP Challenge 2027 compliance script
(`scripts/verify_submission.py` in szczurek-lab/amp-challenge-2027). Do not relax
any of them.
"""

from __future__ import annotations

# --- Alphabet and length (verify_submission.py) -----------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_SET = frozenset(AMINO_ACIDS)
MIN_LENGTH = 8
MAX_LENGTH = 50

# --- Submission shape -------------------------------------------------------
LIBRARY_SIZE = 50_000
TOP_SIZE = 100

# --- Novelty constraints ----------------------------------------------------
# The FULL library must have zero exact matches against data/antibacterial.fasta.
# The top-100 must additionally satisfy Levenshtein.ratio(seq, ref) <= 0.80
# against EVERY reference sequence.
MAX_TOP_SIMILARITY = 0.80

# A safety margin below the hard threshold, so that a slightly different
# similarity definition on the organizers' side (the proposal mentions an
# MMseqs2-based 80% identity filter, which is not identical to Levenshtein
# indel ratio) cannot disqualify a candidate.
TOP_SIMILARITY_MARGIN = 0.72

# --- Output layout ----------------------------------------------------------
# The verifier reads `<entry_point>/library.fasta` and `<entry_point>/top.fasta`,
# where <entry_point> is the stem of argv[0], i.e. "generate".
LIBRARY_FILENAME = "library.fasta"
TOP_FILENAME = "top.fasta"

# --- Reproducibility --------------------------------------------------------
DEFAULT_SEED = 42
