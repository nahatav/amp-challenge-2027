"""Locate repo-relative resources robustly.

The organizers run `uv run --no-sync generate` with the repository root as the
working directory, so `./data` and `./checkpoint` resolve. We additionally walk
up from this file so the entry point still works when invoked from elsewhere
(e.g. during local development or from a test).
"""

from __future__ import annotations

from pathlib import Path


def _candidate_roots() -> list[Path]:
    roots = [Path.cwd()]
    here = Path(__file__).resolve()
    roots.extend(here.parents)
    return roots


def find_resource(relative: str) -> Path:
    """Return the first existing path matching `relative` searching cwd then upwards."""
    rel = Path(relative)
    for root in _candidate_roots():
        candidate = root / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate '{relative}'. Searched {Path.cwd()} and the parents of "
        f"{Path(__file__).resolve()}."
    )


def repo_root() -> Path:
    """Directory containing pyproject.toml."""
    for root in _candidate_roots():
        if (root / "pyproject.toml").exists():
            return root
    return Path.cwd()
