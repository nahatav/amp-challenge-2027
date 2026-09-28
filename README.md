# AMP Challenge 2027 — submission

De novo generative design of linear antimicrobial peptides for the
[AMP Challenge](https://szczurek-lab.github.io/amp-challenge-website/)
(NeurIPS 2026 Competition Track).

## Reproduce the submission

```bash
uv sync
uv run generate
```

This writes:

- `generate/library.fasta` — 50,000 designed peptides
- `generate/top.fasta` — the 100 ranked candidates for wet-lab validation

Both files are byte-identical across runs for the default seed (42), on CPU.

## Verify compliance locally

```bash
uv run python scripts/selfcheck.py --check-reproducibility
```

This mirrors every check in the organizers' `verify_submission.py`: library size,
alphabet, length bounds, duplicates, top-list membership, zero exact overlap with
`data/antibacterial.fasta`, the exhaustive ≤0.80 Levenshtein-ratio constraint on
the top-100, and byte-level reproducibility.

## Layout

```
src/amp_design/
  constants.py    competition constraints, taken from verify_submission.py
  fasta.py        FASTA I/O matching the organizers' parser byte-for-byte
  descriptors.py  NumPy reimplementation of the modlamp descriptors seqme uses
  model.py        generative model (backoff Markov baseline)
  filters.py      novelty + 80%-identity screening with exact blocking filters
  scoring.py      conformity density, potency / selectivity / synthesizability
  selection.py    density-stratified library assembly, diverse top-100 selection
  generate.py     entry point
scripts/
  train_markov.py      fit the baseline generator
  selfcheck.py         local compliance check
checkpoint/       model weights (committed)
data/             reference AMP database (from the organizers' starter kit)
```

## Design rationale

The two deliverables are optimised separately because they are scored by different
things.

**The 50,000-sequence library** is scored in Phase 1 across four metric families:
surrogate activity prediction, sequence-level statistics (uniqueness, diversity,
novelty), distributional similarity in ESM embedding space (FBD, MMD,
precision/recall), and property-distribution conformity. Diversity, FKEA and
authenticity reward spreading out; FBD, precision and conformity reward sitting on
the AMP manifold. We resolve that tension with density-stratified selection —
discard the bottom quartile by conformity density, then fill density-ranked strata
with quotas that grow toward the mode, so the library is centred on the AMP
distribution while retaining coverage of sequence space.

**The top-100** is what gets synthesised. Twenty-five of the hundred are drawn
uniformly at random and the team score is the arithmetic *mean* over that draw, so
a single outstanding peptide is worth nothing — every one of the hundred has to be
strong, and the variance matters as much as the mean. Candidates are ranked by a
composite of potency, selectivity and synthesizability priors, then filtered for
pairwise dissimilarity so that one wrong modelling assumption cannot take down the
whole batch.

## Training data

- `data/antibacterial.fasta` — 39,448 antibacterial peptides, supplied in the
  competition starter kit, derived from the MarLys AMP database (an aggregation of
  AMPDB, APD, BaAMPs, CAMP, CancerPPD, DBAASP, DRAMP, InverPep, SATPdb and dbAMP).
  All sequences are 8–50 residues over the 20 canonical amino acids.

No proprietary or non-public data is used. No sequence from the reference database
is reproduced verbatim in the submission; exact matches are filtered before
selection.

## AI-assistant disclosure

Per the competition's use-of-AI-assistants rule, this submission was prepared with
assistance from a large language model (Claude). The human author remains
responsible for all submitted content.

## License

MIT — see `LICENSE`.
