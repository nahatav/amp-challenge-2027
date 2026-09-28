# Submission checklist

**Deadline: 1 October 2026, AOE.** Kaggle shows "2 days to go" as of 28 Sept.

---

## Blockers only you can clear

### 1. Institutional email — CLEARED
The NeurIPS proposal §2.3 requires an institutional address and explicitly
rejects gmail/hotmail/yahoo for primary registration. **Use
`vnahata@ucsd.edu`** — a `.edu` student address, which the proposal accepts on
the same basis as faculty. Register the Kaggle team under that address, not the
gmail one.

### 2. Create and push the GitHub repository
The repo must be **public** for co-authorship eligibility, with an OSI-permissive
license (MIT is already in place).

```bash
cd C:\Users\Valmi\amp-challenge\submission
gh repo create amp-challenge-2027-submission --public --source=. --push
```

Then grant read access to `@RasmusML` and `@szymczakpau` (required even for
public repos per the minimum requirements).

### 3. Submit on Kaggle
https://www.kaggle.com/competitions/amp-challenge — "Join Hackathon", then submit
per their form. You'll need:
- the abstract (draft below),
- the GitHub URL,
- the 50,000 library and top-100 (both committed in `generate/`),
- a training-data disclosure (see README "Training data").

### 4. Ask the organizers two questions
Open an issue at https://github.com/szczurek-lab/amp-challenge-2027/issues:

- **Top-50 or top-100?** The website FAQ says the 25 tested peptides are drawn
  from the *top 50*; the NeurIPS proposal and the Kaggle page both say the
  *top 100*. This materially changes how we order the list — if it's the top 50,
  ranks 51–100 are pure insurance and the first 50 should carry all the strongest
  candidates.
- **Which identity check governs?** `verify_submission.py` uses
  `Levenshtein.ratio` against `data/antibacterial.fasta` (39,448 sequences); the
  proposal describes MMseqs2 pairwise identity against the full MarLys database
  (~102k). We satisfy the stricter with margin (worst observed 0.714 vs the 0.80
  limit), but worth confirming.

---

## What's done

- [x] Full competition spec extracted, including the NeurIPS proposal PDF that
      ships inside the starter kit (`COMPETITION_SPEC.md`)
- [x] Literature review, 50 references, each tied to a design decision
      (`LITERATURE_REVIEW.md`)
- [x] Compliant baseline submission, byte-identical across runs, banked in git
- [x] Transformer generator trained and measured against the Markov baseline
- [x] Oracle ensemble: AMP classifier, per-species MIC, HC50
- [x] Library-selection rule calibrated against a measured Pareto curve
- [x] Determinism hardening for cross-machine reproduction
- [x] `scripts/selfcheck.py` reproduces every organizer compliance check locally

## Before you submit — run this

```bash
cd C:\Users\Valmi\amp-challenge\submission
uv sync
uv run generate
uv run python scripts/selfcheck.py --check-reproducibility
```

Expect "ALL CHECKS PASSED". Then commit `generate/library.fasta` and
`generate/top.fasta`.

Optionally verify exactly as the organizers will, against a pushed repo:

```bash
uv run python scripts/verify_submission.py https://github.com/<you>/<repo>
```

(Copy that script from `../research/amp-challenge-2027/scripts/`.)

---

## Draft abstract

> **De novo antimicrobial peptide design with a conditional peptide transformer
> and a reproducibility-hardened selection pipeline**
>
> We generate candidate peptides with a 4.8M-parameter conditional
> autoregressive transformer trained on the competition's antibacterial
> reference set augmented with the non-overlapping portion of DRAMP. Four
> control tokens — length, net charge, amphiphilicity and an activity bucket
> derived from an adversarially-trained classifier — are independently dropped
> during training to enable classifier-free guidance.
>
> The 50,000-sequence library and the top-100 list are optimised separately,
> because Phase 1 and Phase 2 score different things. The library is selected by
> density-temperature sampling against the reference AMPs' (amphiphilicity,
> charge) kernel density; the temperature was calibrated on a measured Pareto
> curve between seqme's ConformityScore and its embedding-distribution metrics,
> after we found that a naive density-stratified rule bought conformity by
> degrading FBD, FKEA and Recall. The top-100 is ranked by expected success rate
> — P(MIC ≤ 16 µM) averaged over the 20-strain panel — using gradient-boosted
> oracles trained on GRAMPA (MIC, MAE 0.48 log units) and HemoPI2 (HC50), with
> synthesizability entering as a gate rather than an objective, and a pairwise
> dissimilarity constraint to reduce the variance of the random 25-peptide draw.
>
> All splits are homology-grouped rather than random. The generation path
> contains no BLAS calls and samples in float64 with quantized logits, so the
> library reproduces byte-identically across machines, not merely across runs.
>
> AI-assistant disclosure: prepared with assistance from a large language model
> (Claude); the human author is responsible for all submitted content.

---

## Honest caveats to keep in mind

- Every wet-lab number we report is **our own oracles' prediction**, not an
  independent estimate. MIC MAE is 0.48 log units — roughly a 3× uncertainty
  band on any individual peptide. The oracles are used for ranking, which is
  what they're good for.
- We have not reproduced AMPlify / amPEPpy / AMPredictor / MBC-Attention
  exactly; the seqme third-party plugin branches referenced in the organizers'
  notebook are not public. Our surrogate is a stand-in.
- The Phase-1 aggregation weights are deliberately held out until the close of
  Phase 1, so no strategy can be tuned to them directly. We aimed for balanced
  strength across all four metric families rather than maximising any one.
- With 19 teams currently registered and a cap of 20 advancing, the most likely
  outcome is that every compliant team reaches the wet lab — which makes the
  top-100 quality (Phase 2) matter at least as much as the Phase-1 ranking.
