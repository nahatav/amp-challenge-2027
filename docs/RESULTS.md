# Measurements

Everything here was measured locally, mostly with the organizers' own `seqme`
package (`eval/evaluate.py`), against the real reference sets. Arrows mark the
metric's own objective.

---

## 1. Oracles (homology-grouped splits, not random splits)

Random row splits inflate AMP model scores badly (Sidorczuk et al. 2022;
AMPBench-MT 2026), so all splits below group sequences by shared k-mer
signature so near-duplicates cannot straddle train/test.

| Head | Data | Metric | Result |
|---|---|---|---|
| AMP classifier | 39,448 AMPs vs 47,542 adversarial negatives | AUROC | 0.841 |
| | | AUPRC | 0.824 |
| | | MCC | 0.506 |
| MIC regressor v1 | GRAMPA, 29,418 (sequence, species) pairs | MAE / R² / ρ | 0.480 / 0.354 / 0.588 |
| **MIC regressor v2** | **+ BATTLE-AMP, 38,483 pairs over 9,023 seqs** | **MAE / R² / ρ** | **0.485 / 0.383 / 0.611** |
| HC50 regressor | HemoPI2, 1,908 peptides | MAE / R² / ρ | 0.267 / 0.318 / 0.581 |

Adding the organizing lab's curated DBAASP tables from
`battleamp-snakemake/data/activity/` lifted the MIC regressor's unique-sequence
count from 5,796 to 9,023 and Spearman from 0.588 to **0.611**, with R² from
0.354 to 0.383. MAE is flat (0.480 → 0.485), which is the right trade: the
oracle is used for *ranking*, so rank correlation is the operative metric.

Per-species Spearman, v1 → v2 (panel species): *P. aeruginosa* 0.57 → **0.63**,
*E. faecalis* 0.50 → **0.60**, *E. cloacae* 0.65 → **0.70**, *S. enterica*
0.60 → **0.65**, *A. baumannii* 0.63 → **0.65**, *E. coli* 0.64 → 0.66,
*S. aureus* 0.58 → 0.58. Two regress on small test sets: *B. subtilis*
0.62 → 0.53 and *E. faecium* 0.71 → 0.53 (n = 34, noisy).

A unit trap worth recording: those tables have a `unit` column that mixes µM and
µg/ml, but it describes the *original* `concentration` field — the derived
`MIC`/`activity` columns are already normalised to µg/ml. Branching on `unit`
would double-convert most rows. Verified against a worked example (RRWWWWRRW,
MW 1574 Da: concentration 4, unit µM, activity 6.29156 = 4 × 1574/1000).

For context, AMPBench-MT's best reported pMIC regressor is ESM-C 300M at
MAE 0.504 / ρ 0.562 / R² 0.286, with CatBoost on handcrafted features at
MAE 0.521. Ours is MAE 0.480 / ρ 0.588 / R² 0.354 — competitive, though on a
different dataset and split, so not a like-for-like comparison.

Per-species MIC (Spearman ρ): *E. faecium* 0.71, *K. pneumoniae* 0.66,
*E. cloacae* 0.65, *E. coli* 0.64, *A. baumannii* 0.63, *B. subtilis* 0.62,
*S. enterica* 0.60, *S. aureus* 0.58, *P. aeruginosa* 0.57, *E. faecalis* 0.50.

### Classifier false-positive rate by negative source
The number that matters is the shuffled-AMP FPR: those keep composition exactly
and destroy only order, so they are the adversarial case OmegAMP was built
around.

| Negative source | FPR |
|---|---|
| UniProt short peptides | 0.032 |
| Uniform random | 0.039 |
| Mutated AMPs (5 substitutions) | 0.285 |
| **Shuffled AMPs** | **0.225** |

How much of that 22.5% is genuine error? The literature cuts both ways, but
mostly against my first reading. Benchmarking work on shuffled versus designed
peptides reports that shuffled variants generally *lose* antibacterial activity
while keeping composition and bulk physicochemical properties, and that
predictors fail badly on them (accuracies below 30%). Against that, Hayouka's
random peptide mixtures — defined-composition, random-sequence Lys/Phe peptides
— are potently antimicrobial, so composition alone can suffice in that
particular regime (short, binary composition, tested as a mixture).

Net: most of the 22.5% is real classifier error rather than label noise, and
sequence *order* carries real signal. That motivated the activity-conditioning
experiment in §5 — which we then measured and switched off, because BATTLE-AMP
(from the organizing lab) reports that *most* published models fail this same
composition-matched test, so the gap is field-wide rather than ours alone, and
closing it cost more FBD/Recall than it was worth.

---

## 2. The negative-data-bias trap, reproduced

Training the *same* features and model against UniProt negatives only — the
conventional setup used by AMPlify, amPEPpy and AMPScanner — gives AUROC 0.985,
which looks excellent. Scoring held-out sequence classes with it shows what it
actually learned:

| Sequence set | Adversarial classifier | Conventional classifier |
|---|---|---|
| Reference AMPs | 0.735 | 0.982 |
| Shuffled AMPs | 0.339 | 0.961 |
| **Uniform random** | 0.180 | **0.888** |
| UniProt | 0.067 | 0.074 |
| Our transformer pool | 0.314 | 0.903 |
| Markov baseline | 0.310 | 0.954 |

A conventional AMP classifier rates *uniform random peptides* at 0.89. It is
detecting "not UniProt-like", not "antimicrobial". This is exactly Sidorczuk et
al.'s finding, reproduced end to end.

**Consequence for us.** On AMPlify/amPEPpy-style surrogates our library already
scores ~0.90, so the "surrogate activity prediction" metric family is not a
weak point. The organizers do say their aggregation score was tuned against
"synthetic decoys", so the harder adversarial signal may also be in there — but
BATTLE-AMP's own finding is that most models cannot make that distinction,
which caps how much it can discriminate between submissions.

---

## 3. Generator comparison (seqme, n = 3,000 per group, ESM2 t6_8M)

### 3a. First round — transformer vs Markov

| Library | Div(5)↑ | Novelty↑ | FKEA↑ | FBD AMPs↓ | MMD↓ | Prec↑ | Recall↑ | Auth↑ | Conformity↑ |
|---|---|---|---|---|---|---|---|---|---|
| **Transformer v1** | 0.848 | 1.000 | **474** | **1.18** | **2.90** | 0.843 | **0.626** | 0.759 | 0.469 |
| Markov (raw) | 0.845 | 1.000 | 207 | 2.56 | 6.68 | 0.861 | 0.504 | 0.737 | 0.582 |
| Markov + stratified selection | 0.849 | 1.000 | 164 | 3.04 | 8.38 | 0.859 | 0.415 | 0.731 | 0.773 |
| *AMPs (reference)* | *0.855* | *0.000* | *788* | *0.069* | *0.027* | *0.883* | *0.884* | *0.550* | *0.496* |
| *AMPs (shuffled)* | *0.857* | *0.999* | *601* | *0.78* | *2.26* | *0.844* | *0.754* | *0.680* | *0.509* |
| *UniProt* | *0.814* | *0.946* | *138* | *3.40* | *13.3* | *0.802* | *0.432* | *0.603* | *0.364* |
| *Uniform random* | *0.867* | *1.000* | *188* | *3.89* | *13.9* | *0.733* | *0.333* | *0.809* | *0.471* |

### 3b. Second round — retrained model (val perplexity 13.2 → 5.53)

v1 stopped early and was underfit. Retraining to convergence (103 epochs, early
stop, DRAMP added to the corpus) produced a large, uniform improvement:

| Library | Div(5)↑ | FKEA↑ | FBD↓ | MMD↓ | Prec↑ | Recall↑ | Auth↑ | Conformity↑ |
|---|---|---|---|---|---|---|---|---|
| **Transformer v2 (null activity)** | 0.863 | **779** | **0.901** | **2.82** | 0.763 | **0.766** | 0.778 | **0.512** |
| v2 + activity bucket 5 | 0.867 | 773 | 1.038 | 3.66 | 0.754 | 0.739 | 0.783 | 0.516 |
| v2 + activity bucket 7 | 0.851 | **1082** | 1.114 | 2.90 | 0.723 | 0.710 | 0.797 | 0.508 |
| Transformer v1 | 0.850 | 511 | 1.191 | 2.99 | 0.813 | 0.612 | 0.769 | 0.468 |
| *AMPs (reference)* | *0.854* | *790* | *0.084* | *0.041* | *0.884* | *0.876* | *0.577* | *0.503* |
| *AMPs (shuffled)* | *0.856* | *625* | *0.820* | *2.37* | *0.835* | *0.760* | *0.690* | *0.512* |

v2 essentially **matches the reference on FKEA** (779 vs 790), lifts Recall from
0.61 to 0.77 (reference 0.88), cuts FBD to 0.90, and reaches Conformity 0.512 —
above both the reference (0.503) and uniform random (0.486) with no selection at
all. The one regression is Precision (0.813 → 0.763); density-based selection
recovers some of that, since it favours typical sequences.

**Activity conditioning: measured, then switched off.** Steering the activity
axis raises the adversarial classifier's score (0.369 → 0.415 at bucket 7) and
pushes FKEA above the reference, but costs FBD (0.90 → 1.11), Recall
(0.77 → 0.71), Precision (0.76 → 0.72), composition fidelity (L1 0.063 → 0.208)
and doubles exact-match memorisation (11 → 21 per 8,000). Not worth it. The
better route to predicted activity turned out to be simply **training the model
properly** — v2's null-conditioned amp_prob (0.369) already exceeds v1's
best-conditioned value.

**Two findings decided the architecture.**

1. The transformer is 2–3× closer to the AMP manifold than the Markov model on
   every embedding-distribution metric. Generator question settled.
2. The density-stratified selection I wrote first was *counterproductive*: it
   bought Conformity (0.58 → 0.77) by wrecking FBD (2.56 → 3.04), FKEA
   (207 → 164) and Recall (0.50 → 0.42). Replaced (§4).

**Note on `ConformityScore`.** Uniform random sequences score 0.471 against real
AMPs' 0.496 — the metric barely separates them, because a random 20-letter
composition lands near the AMP charge/amphiphilicity mode by accident. It is a
weak discriminator and unlikely to carry much aggregation weight, which argues
against over-optimising it.

---

## 4. Library selection: the Conformity / FBD trade-off

Selection rule: sample without replacement with probability ∝ `density ** tau`,
using the Gumbel top-k trick. Pool = 149,955 transformer samples, select 40,000.

| τ | Conformity↑ | FBD↓ | MMD↓ | FKEA↑ | Precision↑ | Recall↑ | Auth↑ |
|---|---|---|---|---|---|---|---|
| 0 | 0.464 | **1.27** | **3.24** | **491** | 0.828 | **0.614** | 0.758 |
| **1** | **0.610** | 1.70 | 5.36 | 356 | 0.839 | 0.578 | 0.748 |
| 3 | 0.709 | 2.06 | 6.57 | 299 | 0.846 | 0.516 | 0.752 |
| 8 | 0.759 | 2.36 | 7.52 | 269 | 0.848 | 0.473 | 0.736 |

Monotone trade-off, as expected: Conformity lives in 2-D descriptor space,
FBD/FKEA/Recall live in ESM embedding space, and concentrating in the former
costs some spread in the latter.

**Chose τ = 1.0.** It lifts Conformity clearly above both the reference (0.496)
and uniform random (0.471) — so we are not exposed on that metric — while giving
up little FBD. τ = 0 leaves Conformity *below* random, which would look bad on a
"maximize" metric; τ ≥ 3 pays far too much FBD/Recall for a metric that barely
discriminates.

---

## 5. Conditioning experiments

### Hand-picked "targeted" descriptor buckets — failed
Steering to charge bucket 4–6 (i.e. +6 to +10) on the theory that "more cationic
is more AMP-like":

| Setting | Charge | Conformity | FBD | MMD | Recall |
|---|---|---|---|---|---|
| empirical | 2.81 | 0.469 | 1.18 | 2.90 | 0.626 |
| targeted, guidance 1 | 6.98 | 0.234 | 2.68 | 14.4 | 0.366 |
| targeted, guidance 2 | 9.75 | 0.075 | 8.15 | 57.6 | 0.208 |

Worse on everything, Conformity included. The AMP charge mode is near **+3**;
steering to +7 moved the library *off* the mode, not onto it. Lesson: sample the
descriptor conditioning from the reference's own empirical joint distribution
and apply concentration later, by selection, where it can be tuned against a
measurement.

### Sampling temperature
| T | Charge | amp_prob | frac > 0.5 |
|---|---|---|---|
| 0.70 | 3.94 | 0.367 | 0.252 |
| 0.85 | 3.35 | 0.348 | 0.210 |
| 1.00 | 2.90 | 0.331 | 0.176 |
| 1.15 | 2.56 | 0.318 | 0.155 |

Temperature moves predicted activity only slightly (0.33 → 0.37). Not enough to
close the gap to real AMPs' 0.735, so the limit is the model, not the sampler —
which motivated adding an **activity conditioning axis** (a fourth control
token, bucketed by the adversarial classifier's score on each training peptide)
and retraining.

---

## 6. Runtime

The submission samples in NumPy float64 on CPU for cross-machine reproducibility
(see `src/amp_design/neural.py`), which is much slower than GPU sampling.
Optimisations applied, measured at batch 512:

| Version | ms / sequence |
|---|---|
| Full recompute each decode step | ~1200 |
| + KV cache | 102 |
| + elementwise single-token attention, lazy cache compaction | **48** |

The elementwise attention rewrite mattered twice over: NumPy dispatches
`(B,H,1,d) @ (B,H,d,L)` as B·H separate tiny BLAS calls — 8,192 per layer per
step at B=1024 — and writing it as a multiply-and-sum both removes that overhead
and removes BLAS from the attention path, which helps the determinism argument.

Verified: OpenBLAS reaches 37 GFLOPS float64 on large matmuls, so the bottleneck
was call overhead and cache copying, not the library.

---

## 7. Compliance (baseline submission, already banked in git)

All checks in the organizers' `verify_submission.py`, reproduced locally:

- 50,000 records, all unique, all 8–50 residues, canonical alphabet only ✓
- Zero exact overlap with `data/antibacterial.fasta` ✓
- top-100 all present in the library, no duplicates ✓
- Max Levenshtein ratio to any of the 39,448 references: **0.714** (limit 0.80) ✓
- Two consecutive runs byte-identical ✓

---

## 8. Predicted wet-lab profile of the top-100 (transformer pipeline)

| Quantity | Value |
|---|---|
| Mean predicted success rate across the 20-strain panel | 0.73 |
| Mean predicted MIC50 | ~5.2 µM (threshold is 16 µM) |
| Mean predicted safety window (log10 HC50/MIC50) | 1.20 (~16×) |
| Mean synthesizability score | 0.92 |

These are our own oracles' predictions, not independent estimates — MIC MAE is
0.48 log units, so individual values carry roughly a 3× uncertainty band. They
are useful for *ranking* candidates, which is all they are used for.

---

## 8b. Does the top-100 ranking actually track potency?

The composite rank score is only worth anything if it orders real peptides by
real measured MIC. Scored against GRAMPA, held out by MIC band:

| GRAMPA MIC band | n | Composite | Predicted success rate |
|---|---|---|---|
| ≤ 2 µM | 500 | **0.651** | 0.809 |
| 2–16 µM | 500 | 0.520 | 0.617 |
| 16–64 µM | 500 | 0.360 | 0.401 |
| ≥ 64 µM | 500 | **0.242** | 0.218 |

Monotone across all four bands. Spearman correlation between the composite and
measured −log10(MIC) over 3,000 GRAMPA peptides is **0.784** — but those
peptides were in the MIC regressor's training set, so treat that as optimistic;
the honest held-out figure is the regressor's own ρ = 0.588 under
homology-grouped splitting.

### A flaw this measurement caught
The first version of the composite added `amp_probability` linearly at weight
0.20. Scoring the bands showed the binary classifier gives **0.693 to peptides
with MIC ≤ 2 µM and 0.760 to peptides with MIC ≥ 64 µM** — among real AMPs it
is slightly *anti*-correlated with potency, because both groups are database
AMPs and the classifier only asks "is this an AMP at all".

This is exactly BATTLE-AMP's first conclusion (Szymczak et al. 2026, from the
organizing lab): MIC-trained models beat binary classifiers regardless of
architecture. The classifier was rewarding the wrong thing.

Fixed by making it a **saturating multiplicative gate** rather than an additive
term — it now penalises sequences that are not plausibly AMPs at all (where the
MIC regressor extrapolates optimistically, having been fit only on actives) and
is indifferent above ~0.45. The band separation above is post-fix.

---

## 9. A real defect the compliance check caught

The first full 50,000-sequence run passed every organizer check, but the
exhaustive similarity sweep reported a worst-case Levenshtein ratio of **0.783**
against the reference database — compliant (the limit is 0.80) but far above the
0.72 margin the pipeline is supposed to enforce. So the margin was silently not
being applied.

Cause: `ReferenceIndex` pruned candidate references with a length band derived
for a threshold of 0.80, while the screen was being run at 0.72. The offending
pair was `KWKFKIKFHFHKKW` (length 14) against `KKFKKFFKK` (length 9); the
hardcoded band admitted reference lengths [10, 21] and so never scored the
length-9 entry. A second, worse problem sat alongside it: a k-mer inverted index
was also pruning references, and that is a *heuristic* — two sequences can
exceed the threshold without sharing a k-mer — so it could drop true violations
at any threshold.

Fix: derive the band from the threshold, exactly. Since
`ratio(a,b) = 1 - indel(a,b)/(|a|+|b|)` and `indel(a,b) >= ||a|-|b||`,

    ratio(a, b) <= 2·min(|a|,|b|) / (|a|+|b|),

so for threshold `t` only references with

    t·na/(2−t) < nb < na·(2−t)/t

can possibly reach it. At t = 0.72 and na = 14 that is (7.88, 24.89), which
admits the length-9 reference. The k-mer index was removed entirely: exhaustive
screening of a 2,000-candidate shortlist against all 39,448 references takes
39 s, which is affordable, and correctness beats cleverness here.

Effect: **7 of the 100 selected peptides** would be rejected under the corrected
screen, so this was not a cosmetic issue — it changed the submitted list.

`verify_top_similarity` deliberately does no pruning at all and does not share
code with `ReferenceIndex`, so the final pre-write check cannot inherit a bug in
the pruning logic. That separation is what surfaced this.
