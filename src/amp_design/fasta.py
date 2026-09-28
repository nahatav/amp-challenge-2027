"""Minimal, dependency-free FASTA I/O.

Deliberately mirrors the parser in the organizers' `verify_submission.py` so that
what we validate locally is exactly what they will parse.
"""

from __future__ import annotations

from pathlib import Path


def read_fasta(path: str | Path) -> tuple[list[str], list[str]]:
    """Return (headers, sequences). Sequences are upper-cased and joined."""
    headers: list[str] = []
    sequences: list[str] = []
    header: str | None = None
    parts: list[str] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    headers.append(header)
                    sequences.append("".join(parts))
                header, parts = line[1:], []
            else:
                parts.append(line.upper())

    if header is not None:
        headers.append(header)
        sequences.append("".join(parts))

    return headers, sequences


def read_sequences(path: str | Path) -> list[str]:
    """Return just the sequences."""
    return read_fasta(path)[1]


def write_fasta(sequences: list[str], path: str | Path, prefix: str = "seq") -> None:
    """Write sequences with 1-based `>{prefix}{i}` headers.

    Uses an explicit "\\n" newline so the file is byte-identical on Windows and
    Linux — the reproducibility check compares raw bytes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for i, seq in enumerate(sequences, start=1):
            handle.write(f">{prefix}{i}\n{seq}\n")
