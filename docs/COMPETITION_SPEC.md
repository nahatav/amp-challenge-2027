# AMP Challenge 2026/27 — Authoritative Spec Notes

Sources: competition website, Kaggle (Overview/Rules), `szczurek-lab/amp-challenge-2027`
(incl. `AMPChallenge_NeurIPSCompetition.pdf`, the NeurIPS proposal), `szczurek-lab/seqme`.

## Hard deadlines
- **Submission: 1 October 2026, AOE.**
- Oct 2026: Phase 1 computational evaluation; qualification decisions.
- Dec 6–12 2026: NeurIPS Competition Track; winners + advancing teams announced.
- Oct 2026–Feb 2027 synthesis; Mar–Apr 2027 MIC/HC50 assays; late 2027 paper.

## BLOCKER: registration email
Proposal §2.3: "Each team must register with an **institutional email address**
(university, research institute, or company domain). ... Free webmail accounts
(gmail, hotmail, yahoo, and equivalents) are **not accepted for primary registration**."
→ A gmail address cannot be the primary team registration.

## Compliance (exact, from `scripts/verify_submission.py`)
The verifier clones the repo, `uv sync`, runs `uv run --no-sync generate` (NO ARGS,
cwd = repo root), then checks:

1. `generate/library.fasta` — **exactly 50,000** records.
2. Every record: non-empty header; alphabet ⊆ `ACDEFGHIKLMNPQRSTVWY`; `8 <= len <= 50`;
   **no duplicate sequences**.
3. `generate/top.fasta` — **exactly 100**, each a **member of the library**, no duplicates.
4. `_verify_no_overlap`: **the FULL 50,000 library must have ZERO exact matches**
   against `data/antibacterial.fasta`. (Not just the top-100. Set intersection.)
5. `_veritfy_max_simularity`: for every top-100 sequence and **every** reference,
   `Levenshtein.ratio(seq, ref) <= 0.80` (strictly `> 0.8` raises).
   NB: `Levenshtein.ratio` is *indel* similarity = `1 - indel_dist/(len(a)+len(b))`.
6. **Reproducibility**: runs `generate` twice; `library.fasta` and `top.fasta` must be
   **byte-identical**.

Output dir is `Path(sys.argv[0]).stem` → `generate/`.

## Reference database
`data/antibacterial.fasta`: **39,448 unique** sequences, all already 8–50 aa,
alphabet exactly the 20 canonical AAs. Median length 18, mean 18.7.
Headers are MarLys IDs (`MLAMP…`) with `len=`, `charge=`, `disulfide=`,
`dbs=AMPDB|APD|BaAMPs|CAMP|CancerPPD|DBAASP|DRAMP|InverPep|SATPdb|dbAMP`,
`activity=antibacterial|antimicrobial|anticancer|anti-biofilm|…`.
→ It is both the exclusion list **and** a labelled training set.
Full MarLys AMP DB (~102k peptides, 13 source DBs), CC-0: doi 10.17632/w4hb5grjwb.3

Proposal also mentions an 80% identity filter computed with **MMseqs2** against MarLys —
a second, different check from the Levenshtein one in the script. Satisfy the stricter.

## Phase 1 scoring — four metric families (proposal §1.5)
1. **Surrogate activity prediction** — aggregated over multiple independent AMP
   classifiers / MIC regressors, e.g. AMPredictor, MBC-Attention, DeepAMP,
   "chosen to reduce any single model's bias".
2. **Sequence-level** — Uniqueness, internal Diversity, Novelty vs known AMPs
   (alignment-based, normalized bit-scores), clustering-based coverage.
3. **Distributional similarity in embedding space** — FBD, MMD, Precision/Recall,
   against **two** reference sets (curated known AMPs **and** a generic peptide set),
   using **ESM2 and ESM-C** embeddings.
4. **Property distribution** — ConformityScore of **charge** and **amphiphilicity**
   to known AMPs; rate satisfying empirical **synthesizability** constraints.

Plus exact-match screen vs MarLys to quantify rediscovery.

> "The aggregation weights, tie-breaking criteria, and curated reference-set
> composition ... are **held out until the close of Phase 1**, to prevent direct
> optimization against the ranking function."
→ Do not overfit one metric. Be strong on all four families simultaneously.

## The actual seqme recipe (docs/tutorials/benchmark_peptides.ipynb)
```
Count, Uniqueness, Diversity(k=5), Novelty(reference=AMP-data),
FKEA(embedder=esm2-embed, bandwidth=1.0), FBD(UniProt), FBD(AMPs),
Fold(Precision(n_neighbors=3)), Fold(Recall(n_neighbors=3)),
ID(amPEPpy), ID(AMPlify),
AuthPct(train_set=AMP-data, embedder=esm2-embed),
ConformityScore(reference=AMPs, predictors=[amphiphilicity, charge],
                kde_bandwidth="silverman")
```
Embedder in the tutorial: `ESM2 t6_8M`, batch 512. Descriptors come from **modlamp**
(`GlobalDescriptor.calculate_charge(ph=7.0)`, `PeptideDescriptor` hydrophobic moment,
eisenberg scale, window 11, angle 100, modality "mean").

### Metric mechanics that matter
- **Novelty** = fraction of sequences *not exactly present* in the reference set.
  Trivially 1.0 if we never copy. Free point.
- **Diversity** = mean over sequences of mean normalized Levenshtein distance
  (`edit / max(len_a, len_b)`) to k=5 random others. Random strings maximize it;
  realism pulls the other way.
- **ConformityScore** = fit Gaussian KDE (silverman) on the reference's
  [amphiphilicity, charge] 2-D descriptor cloud with 5-fold CV; score = fraction of
  held-out reference points whose log-density is ≤ each generated point's log-density.
  **It rewards sitting at the MODE of the AMP descriptor distribution, not matching its
  spread.** A library concentrated in the high-density core scores ~0.9+; 0.5 = median
  typicality. This is the single most directly exploitable metric.
- **AuthPct** = fraction of generated whose NN-distance to the training set exceeds that
  training point's own NN-distance. **Penalizes near-copying.** Direct tension with FBD.
- **FBD** (minimize) computed against *both* UniProt (generic) and AMPs — so we must be
  AMP-like, and the UniProt FBD is a contrast, not a target to minimize blindly.
- **FKEA** = Vendi/RKE diversity in embedding space, reference-free; rewards many
  distinct modes. Guards against mode collapse.
- **Precision/Recall** wrapped in `Fold` with `strict=True` (equal sample counts).

### The core tension to engineer around
Diversity + FKEA + AuthPct + Novelty push *away* from the AMP manifold;
FBD + Precision + ConformityScore + surrogate activity push *toward* it.
Winning library = **high-density in the 2-D descriptor space, broadly spread in
ESM2 embedding space, zero exact reuse, no near-copies.**

## Two-objective split (important)
- The **50,000 library** is what Phase 1 scores → optimize realism/diversity/novelty/
  conformity.
- The **top-100** is what gets synthesized → optimize predicted potency + selectivity
  (HC50/MIC50), subject to the ≤80% identity constraint.
These are different objectives. Do not use one ranking for both.

Phase 2: **25 peptides drawn uniformly at random** from the top-100 (proposal §1.5
and Kaggle both say top-100; the website FAQ says "top 50" — contradiction, ask).
Team score = **arithmetic mean over those 25**. → The whole top-100 must be strong;
one hero sequence is worthless. Optimize the *mean*, and minimize variance.

## Scoring definitions (Phase 2)
- MIC ceiling 64 µM; HC50 ceiling 128 µM.
- Potency threshold **MIC ≤ 16 µM** (tighter than the 32 µM used in earlier work).
- Success rate = fraction of strains in the panel meeting the threshold.
- SW = HC50 / MIC50 across all 20 strains.
- Categories: Broad-spectrum, Gram+, Gram−, MDR ESKAPE (8 isolates), Optimal Selectivity.

## Baselines to beat
HydrAMP (cond. VAE, Szymczak et al. Nat Commun 2023) and AMP-Diffusion (Torres et al.,
Cell Biomaterials 2025). Organizers publish their 50k libraries + seqme metrics as the
target. Both are excluded from rankings (organizer-affiliated).

## Rules worth remembering
- Only **generative** methods permitted.
- Top-100 must be a subset of the library.
- One entry per model; genuinely distinct methods may each submit; no team-size limit.
- 72-hour window to fix failed compliance checks (proposal §2.2) — the Kaggle page's
  flat "will be disqualified" is contradicted by the proposal and the website.
- Post-submission integrity checks: pairwise library-overlap and method-abstract
  similarity across teams (anti-collusion).
- **LLM/AI-assistant use is permitted but must be disclosed** in method documentation.
- Fewer than 20 qualifying teams → unused capacity redistributed as **extra peptides
  per team**. As of 2026-09-28 Kaggle shows 19 teams / 23 participants.

## Co-authorship (Full Requirements)
Public repo on the template, OSI-permissive license, fixed default seed giving identical
output across runs, full training-data disclosure (any non-public data must be released
permissively). Minimum-only teams get their data but no authorship.
