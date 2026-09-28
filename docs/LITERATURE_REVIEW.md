# Generative AI for Antimicrobial Peptide Design — Literature Review

Compiled for the AMP Challenge (NeurIPS 2026 Competition Track), 28 Sept 2026.
Every section ends with **→ Decision**: what we actually did with the finding.

---

## 0. Why this review is shaped the way it is

The competition scores two different artefacts with two different objectives:

| Artefact | Scored by | Objective |
|---|---|---|
| 50,000-sequence library | Phase 1, `seqme` + surrogate oracles | realism, diversity, novelty, property conformity |
| top-100 list | Phase 2, wet lab, **mean over a random 25** | MIC ≤ 16 µM across 20 strains; HC50/MIC50 |

So the review is organised around (1) what generates realistic-but-novel peptide
libraries, (2) what predicts potency and toxicity well enough to rank 100
candidates, and (3) what the assay and synthesis pipeline will actually do to us.

---

## 1. The organizers' own stack (highest-signal prior art)

The organizing committee has published the generator, the evaluator and the
benchmark. Reading their code is worth more than reading anyone's paper.

### 1.1 seqme — the Phase 1 evaluator
Møller-Larsen, Izdebski, Olszewski, Gawade, Kmicikiewicz, Zarzecki, Szczurek.
*seqme: a Python library for evaluating biological sequence design.*
Bioinformatics Advances 6(1) vbag212, 2026. arXiv:2511.04239. BSD-3.

Three metric families — sequence-based, embedding-based, property-based.
The implementations that matter, read from source:

- **`Novelty`** = fraction of sequences *not exactly present* in the reference.
  Exact string match only. Free 1.0 if we never copy.
- **`Diversity(k)`** = mean over sequences of mean normalised Levenshtein
  distance (`edit / max(len_a, len_b)`) to `k` random others.
- **`ConformityScore`** (after Frey et al., walk-jump sampling) = Gaussian KDE
  (Silverman bandwidth) fit on the reference's descriptor cloud under 5-fold CV;
  score = fraction of held-out reference points whose log-density is ≤ each
  generated point's. **This rewards sitting at the *mode*, not matching the
  spread.** Configured in their notebook with `[HydrophobicMoment(), Charge()]`.
- **`AuthPct`** (Alaa et al. 2022) = fraction of generated sequences whose
  nearest-training-neighbour distance exceeds that neighbour's own nearest-
  training distance. **Explicitly penalises near-copying.**
- **`FBD`** = Fréchet distance between embedding Gaussians. Minimise.
- **`FKEA`** (Vendi / RKE via random Fourier features) = reference-free
  diversity; counts effective modes in embedding space.
- **`Precision`/`Recall`** (Kynkäänniemi et al. 2019) with k-NN manifolds.

The tutorial `benchmark_peptides.ipynb` pins the exact recipe, embedder
(`ESM2 t6_8M`), and descriptor backend (`modlamp`).

**→ Decision.** We reimplemented `ConformityScore`'s KDE geometry in NumPy and
rank library candidates by it directly; we reimplemented modlamp's `Charge` and
`HydrophobicMoment` exactly so our internal score is the score. We also run the
real `seqme` offline as ground truth (`eval/evaluate.py`).

### 1.2 OmegAMP — the state of the art, from the same lab
Soares, Hetzel, Szymczak, Torres, Sommer, de la Fuente-Nunez, Theis, Günnemann,
Szczurek. *OmegAMP: Targeted AMP Discovery through Biologically Informed
Generation.* arXiv:2504.17247; ICLR 2026.

- 55-dimensional amino-acid embedding from physicochemical scales:
  Wimley–White hydrophobicity, isoelectric point, Levitt secondary-structure
  propensity, transmembrane propensity, and the Average Amino Acid Selectivity
  Index.
- 1D U-Net denoiser with linear attention; conditioning
  `(1_AMP, |s|, Charge, Hydrophobicity)` injected at every layer; self-
  conditioning and condition-annealed sampling.
- Generative training set: **only 5,149 non-redundant AMPs** (DRAMP, dbAMP, APD).
- Filtering cascade of **XGBoost** classifiers (general, species-specific,
  strain-specific) on global descriptors, composition, and exponential moving
  averages of amino-acid scales.
- **Synthetic negatives**: purely random, **shuffled AMPs** (composition kept,
  order destroyed), **mutated AMPs** (5 substitutions), added/deleted variants.
  This is the false-positive-rate story.
- **In vitro: 24/25 peptides (96%) active**; 80% at ≤4 µg/ml, >90% at 8 µg/ml
  vs <40% for baselines; 92% efficacy against MDR strains at 8 µg/ml.

**→ Decision.** This is the single most important paper for us. We copied the
adversarial negative-set construction wholesale into our classifier training,
and we use gradient-boosted trees for the same reason they do (exactness,
small-data robustness). It also sets the bar: any Phase-2 story that does not
approach ~90% hit rate is not competitive.

### 1.3 HydrAMP and Hyformer — the declared baselines
- Szymczak et al., *Discovering highly potent antimicrobial peptides with deep
  generative model HydrAMP*, Nat Commun 14:1453, 2023 — conditional VAE
  disentangling peptide representation from the antimicrobial condition.
- Izdebski, Olszewski, Gawade, Koras, Korkmaz, Rauscher, Tomczak, Szczurek,
  *Synergistic Benefits of Joint Molecule Generation and Property Prediction*
  (Hyformer), arXiv:2504.16559, TMLR 2026. One transformer backbone with a task
  token switching between causal and bidirectional masking; loss
  `ℓ_LM + µ·ℓ_MLM + η·ℓ_PRED`. Peptide setup: pretrain 3.5M general peptides +
  1M AMPs, fine-tune 4,547 peptides with *E. coli* MIC, 39 descriptors from
  `peptidy`, condition on MIC ≤ 2 µM, **best-of-K** conditional sampling.

Hyformer's reported oracle hit rates are a concrete target to beat:
**AMPlify 0.80, amPEPpy 0.72, HydrAMP_MIC 0.19, Fitness 0.80** (HydrAMP 0.87).

**→ Decision.** These numbers are our Phase-1 scoreboard proxy. Best-of-K
sampling is cheap and effective; our selection stage is a generalisation of it.

### 1.4 AMP-Diffusion — the other declared baseline
Torres, Chen, Wan, Chatterjee, de la Fuente-Nunez, *Generative latent diffusion
language modeling yields anti-infective synthetic peptides*, Cell Biomaterials
1(9), 2025. Diffusion over protein-language-model embeddings; generated peptides
match experimentally validated AMPs on pseudo-perplexity and residue diversity.

### 1.5 BATTLE-AMP — the organizers benchmarking the surrogates themselves
Szymczak P., Bukała A., Zarzecki W. et al. *BATTLE-AMP: Benchmarking
Antimicrobial Peptide Predictors.* bioRxiv, June 2026.

Same lab, same people running this competition, published three months before
the deadline. It evaluates AMP predictors against **experimentally measured
MICs** across clinically relevant species and strains: a survey of 48 published
methods found **fewer than 25% reproducible**, and 10 model families were
benchmarked using experimental data plus molecular dynamics.

Its four stated conclusions, each of which bears directly on our design:

1. **Models trained on MIC data outperform binary classifiers, regardless of
   architecture.**
2. **The best model depends on the target pathogen**, so model selection must be
   guided by the biological question.
3. **Most models cannot distinguish active peptides from inactive sequences with
   identical amino acid composition.**
4. **Activity cliffs remain unresolved** by both ML and MD — a stated limit of
   current computational methods.

**→ Decision.** Three consequences, and this paper retroactively justifies
choices we had made on other grounds.

- (1) says rank the top-100 on a **MIC regressor**, not on a binary AMP
  classifier. That is what `oracles.composite_rank_score` does: the classifier
  enters only as a sanity gate, at weight 0.20, against the MIC-derived success
  rate at 0.40.
- (2) says **condition MIC prediction on species**. We predict across 10 panel
  species separately and weight by the number of strains each contributes to the
  20-strain panel.
- (3) recalibrates how worried to be about our library scoring like shuffled
  AMPs on our adversarial classifier (0.31 vs real AMPs' 0.74). That failure
  mode is field-wide, and it is the organizers' own finding — so the surrogate
  ensemble in Phase 1 is unlikely to separate us from real AMPs on that axis
  either. It is a reason not to spend more of the FBD/Recall budget chasing it,
  which is exactly what the activity-conditioning experiment concluded
  independently.
- (4) is a caution on our own numbers: an oracle with MAE 0.48 log units cannot
  resolve activity cliffs, so per-peptide predictions carry roughly a 3x band
  and are only trustworthy for ranking.


### 1.6 The surrogate roster, from BATTLE-AMP's model registry

`szczurek-lab/battleamp-snakemake/models/registry.yaml` enumerates the predictor
families the organizing lab integrated. This is almost certainly the pool the
Phase 1 "surrogate activity prediction" family draws from — the proposal names
AMPredictor, MBC-Attention and DeepAMP, all of which appear here.

**Classifiers**

| Model | Architecture | Length range |
|---|---|---|
| AMPscanner v2 | CNN + LSTM | 10–200 |
| AmPEPpy | random forest | unbounded |
| AMPlify | LSTM + attention | 1–199 |
| AMPred-MFA | LSTM + CNN + attention | 3– |
| HydrAMP (AMP, MIC variants) | LSTM | **1–25** |
| SenseXAMP (classifier) | ESM + attention | **6–25** |
| AMPredictor | GCN + ESM | 1–65 |

**Regressors**

| Model | Architecture | Length range | Variants |
|---|---|---|---|
| APEX | attention + RNN | 1–52 | per-species: E. coli, S. aureus, K. pneumoniae, A. baumannii, P. aeruginosa, min |
| Deep-AMP | CNN + LSTM | **1–49** | CNN/LSTM × Gram+/Gram− |
| MBC-Attention | CNN + attention | 5–60 | — |
| SenseXAMP (regressor) | ESM + attention | **6–25** | S. aureus, E. coli |

Two models were **surveyed and excluded**, which is itself informative about how
brittle this field's tools are: MolE ("classified all benchmark sequences as
non-AMP") and sAMP-VGG16 ("classified virtually all benchmark sequences as AMP").

**→ Decision, and a check we would not otherwise have run.** Several of these
models have hard upper length limits — 25 residues for HydrAMP and SenseXAMP,
49 for Deep-AMP. A library skewed long would be scoreable by fewer surrogates.
We checked: **84.7% of our library is ≤25 residues** (the reference AMP set is
82.8%) and **100% is ≤49** (the reference itself has 0.5% at exactly 50, which
Deep-AMP could not score). So we already sit slightly inside the reference on
this axis and no correction was needed — but it was worth verifying rather than
assuming, and it is a cheap way for a long-skewed library to lose points.

The per-species APEX and Gram-split Deep-AMP variants also corroborate our
decision to predict MIC per panel species and weight by strain counts, rather
than predicting a single scalar "activity".


---

## 2. Benchmarks, and how AMP models get fooled

### 2.1 Negative-set bias — the field's biggest methodological trap
Sidorczuk et al., *Benchmarks in antimicrobial peptide prediction are biased due
to the selection of negative data*, Briefings in Bioinformatics 23(5) bbac343, 2022.

660 models × 12 architectures × 11 negative-sampling schemes. Models gain **>10
AUC points** when train and test negatives are sampled the same way. Performance
correlates negatively with the amino-acid-composition gap (ρ = −0.53) and length
gap (ρ = −0.44) between positives and negatives. n-gram features (AmpGram) were
the most robust across negative sets. ~70% of AMP tools are unreproducible.

**→ Decision.** Our classifier is never evaluated against a single negative
family. We report false-positive rate *per negative source* (UniProt, shuffled,
mutated, random) and treat the shuffled-FPR as the number that matters.

### 2.2 AMPBench-MT — homology-controlled, multi-endpoint
arXiv:2607.25518, 2026. Seven endpoints; MMseqs2 clustering at 30% identity for
splits; 161 endpoint-specific evaluations.

Headline numbers we care about:
- **pMIC regression**: ESM-C 300M best (MAE **0.504**, Spearman **0.562**,
  R² 0.286); CatBoost on handcrafted features essentially level (MAE 0.521).
- Low-toxicity classification: ProtT5-XL AUROC 0.826, MCC 0.534.
- **Selectivity regression R² ≈ 0.077** — the therapeutic index is barely
  predictable from sequence with current data.
- Binary AMP classification success (Qwen3-4B QLoRA, MCC 0.862) **does not
  transfer** to any downstream endpoint.

**→ Decision.** Three consequences. (a) Handcrafted features are not a
compromise — they are competitive, and they are exactly reproducible, which PLM
embeddings are not. (b) We do not chase binary-classification metrics. (c) We
do not trust a selectivity model to carry the Optimal Selectivity category;
we combine predicted HC50 with predicted MIC instead.

### 2.3 Composition may be almost all of it
Pal, Kumar, Solanki, Pareek, Singh, Singla, *Coarse composition suffices:
tabular in-context learning for multi-activity AMP profiling*, arXiv:2608.30337.
330 descriptors + TabPFN; **ten global physicochemical scalars recover 91% of
full-feature performance**; mAP-5 77.8% on ESCAPE (82,359 peptides) vs 72.1%
previous best; +11.2 points below 30% sequence identity. Predicted structure
unnecessary at inference.

### 2.4 Cross-species generalisation
PathoMIC (arXiv:2608.26228, 2026): knowledge-guided features beat purely
data-driven representations on **held-out species** — our exact situation, since
the competition panel includes strains with no public MIC data.

### 2.5 Other benchmarks
- ESCAPE / *A Standardized Benchmark for Multilabel AMP Classification*,
  arXiv:2511.04814, NeurIPS 2025 — 80k+ peptides, 27 repositories, multilabel
  hierarchy (antibacterial/antifungal/antiviral/antiparasitic).
- PepBenchmark, arXiv:2604.10531.

**→ Decision.** We built a 261-feature handcrafted design matrix
(composition, 13 global scalars, CTD, sliding-window extremes, terminal-region
composition, Moreau–Broto autocorrelation, reduced-alphabet dipeptides,
amphipathic-run statistics) and fit gradient-boosted trees with **k-mer-grouped
splits**, not random splits.

---

## 3. Protein language models and short peptides

- **ESM-2** (Lin et al. 2022) and **ESM-C** (EvolutionaryScale, 2024; 300M/600M/6B,
  MIT) — ESM-C 300M matches ESM-2 650M at much lower cost. Phase 1 uses both.
- **But**: only **0.06% of UniRef100 is shorter than 50 residues**. General
  protein LMs are badly matched to peptides; this motivated PepBERT, pLM4PEP,
  PeptideBERT. For benchmarks of a few hundred short peptides, larger embeddings
  add dimensionality without generalisation.

**→ Decision.** Two things follow. First, our oracle does not need ESM. Second —
and more usefully — since Phase-1's FBD/precision/recall run on ESM embeddings
of 8–50-mers, those embeddings are likely dominated by coarse length and
composition signal, which a well-matched composition distribution can track.

---

## 4. What actually makes a peptide antimicrobial (design rules)

### 4.1 Charge
Activity rises with net charge from +1 to +6 (2–12.8× MIC improvement), then
selectivity collapses. Bacterial membranes are anionic (phosphatidylglycerol,
cardiolipin, LPS); mammalian membranes are zwitterionic (PC/PE) plus cholesterol.

### 4.2 Hydrophobicity and the therapeutic window
Both antibacterial activity and haemolysis are driven by hydrophobic bulk, so
potency and toxicity are intrinsically coupled. There is an optimum, not a
monotone: an optimal-hydrophobicity window exists per organism class
(Gram-negative optimum distinct from Gram-positive/yeast). Excess hydrophobicity
and amphipathicity drive aggregation, serum binding and haemolysis.

### 4.3 Amphipathicity and helicity
Hydrophobic moment (Eisenberg, 100°/residue helical wheel) is the strongest
single sequence correlate of membrane disruption. But **high helicity correlates
with haemolysis too**; substituting the nonpolar face with cationic residues to
break perfect amphipathicity reduces haemolysis while retaining activity. Higher
self-association in solution ⇒ weaker antimicrobial, stronger haemolytic.

### 4.4 Length
eCAP series (12–48 residues): maximum **selectivity at 24 residues**, maximum
**activity at 12** (WR12). Specific activity peaks below 20 residues. 12–30 is
the practical sweet spot for potency, selectivity and synthesis cost.

### 4.5 Residue identity
- **Arg vs Lys**: the guanidinium group H-bonds to phosphate more strongly than
  the ammonium group, enables bidentate lipid interactions and membrane
  translocation. Activity generally rises with Arg content. But Lys-rich
  natural AMPs may be Lys-rich *for selectivity*.
- **Trp**: partitions to the bilayer interface; adjacent Trp residues confer
  greater activity.
- **The AMP lexicon** (Comms Biol 2021, exhaustive Arg/Trp peptides ≤7 residues):
  a single activity peak at **~40% arginine**; shortest actives are 4–5 residues.
- **Pro**: helix breaker; central Pro is essential to indolicidin's potency and
  its removal cuts both activity and haemolysis.

### 4.6 Composition vs order — the evidence cuts both ways
Hayouka's **random peptide mixtures** — stochastic-sequence, defined-composition
Lys/Phe binary mixtures — are potently antimicrobial and anti-biofilm, and
constrain resistance evolution in *P. aeruginosa*. That is direct evidence that
composition plus length carries much of AMP activity.

Against it: benchmarking work using **shuffled synthetic peptides as controls**
(Cardoso et al., Front. Microbiol. 2020; and the shuffled-peptide predictor
benchmark, J. Theor. Biol. 2017) finds that shuffled variants — identical
composition, randomised order — generally *lose* antibacterial activity, and
that predictors collapse below 30% accuracy on them. Forward feature selection
in that literature identifies **sequence order** alongside charge as critical.

**→ Decision.** We first read our classifier's ~22% false-positive rate on
shuffled AMPs as partly irreducible label noise. On the weight of evidence that
was too generous: most of it is real classifier error, and residue order carries
genuine signal. That reading is what motivated adding an **activity conditioning
axis** to the generator rather than accepting the gap. Our own measurement
agrees: our generated library matches the reference amino-acid composition
closely (L1 distance 0.058 vs a 0.013 subsample noise floor) yet still scores
like shuffled AMPs on the adversarial classifier — so the deficit is in order,
not composition.

### 4.7 Non-membrane mechanisms
Proline-rich AMPs (Bac7, apidaecin, ARV-1502) act non-lytically on the 70S
ribosome and DnaK, entering via the SbmA/BacA transporter. Powerful but
**Gram-negative-restricted** (SbmA is absent in Gram-positives) and
stereospecific.

**→ Decision.** Four of five competition categories reward breadth across a
20-strain panel spanning both Gram classes. We therefore target membrane-active
cationic amphipathic peptides, not PrAMPs.

---

## 5. Toxicity, haemolysis and selectivity

- **HAPPENN** (Sci Rep 2020): 10-fold CV accuracy 85.66%, MCC 0.71; beats
  HemoPI and HemoPred.
- **HemoPI2** (Comms Biol 2025): 1,924 peptides with experimental HC50;
  hybrid RF + motif AUROC 0.921; HC50 regression Pearson R 0.739, R² 0.543.
- **AmpLyze** (arXiv:2507.08162, 2025): deep HC50 regression.
- **ToxinPred 3.0**: 5,518 toxic / 5,518 non-toxic; extra-trees on compositional
  features AUROC 0.95 / MCC 0.78; ESM2-t33 AUC 0.93; hybrid + MERCI motifs
  AUROC 0.98.
- Haemolysis, cytotoxicity and systemic in vivo toxicity correlate (Sci Rep 2020).

**→ Decision.** We trained an HC50 head on HemoPI2 (1,926 usable peptides,
target log10 HC50 censored at the competition's 128 µM ceiling). Because HC50 is
censored, the Safety Window HC50/MIC50 saturates from above — so beyond
non-haemolytic, **only lowering MIC improves selectivity**. Our selectivity
score reflects that asymmetry rather than optimising haemolysis alone.

---

## 6. Synthesis feasibility — an under-modelled term that costs real points

The FAQ is explicit: sequences that fail synthesis, are insoluble, or fail
identity/purity QC **are not retested**. A peptide that cannot be made is a dead
slot in our 25-peptide draw. Phase 1 separately scores "rate of sequences
satisfying empirically derived synthesizability constraints".

Pesciullesi et al., *Amino acid composition drives aggregation during peptide
synthesis*, Nature Chemistry 2026 (ChemRxiv 2025; code `rxn4chemistry/AI4Aggregation`):
539 peptides, ensemble of 100 XGBoost classifiers, ~60% accuracy regardless of
architecture (ESM-2, BERT, time-series, classical ML all tie — the ceiling is the
data, not the model). A plain 20-dim composition vector reaches 59.5%.
SHAP attribution:

- **Drive aggregation**: Ser, Ile, Val, Thr, Gln, Leu (aliphatic/small polar,
  β-sheet packing).
- **Reduce aggregation**: Phe, Asp, Tyr, Arg, Cys, His, **Pro**.
- Critical aggregation occurs at residues 2–12 from the resin anchor.

Flow-SPPS work (PMC10966953): aggregation cuts coupling efficiency 30–40% vs
10–15%; depends on sequence and resin loading, not on coupling reagents.
Classical liabilities remain: multiple free Cys (disulfide scrambling),
Asp-Gly/Asp-Ser aspartimide, Asn-Gly deamidation, N-terminal Gln → pyroglutamate.

**→ Decision.** Our `synthesizability()` was **rewritten** against this paper —
the first draft penalised aromatics (F/Y/W), which is backwards: Phe and Tyr are
*protective*. It is now a composition-weighted aggregation sum plus explicit
chemical liabilities plus homopolymer and length penalties.

---

## 7. Databases

| Resource | Scale | Use here |
|---|---|---|
| **MarLys / MLAMP** (CC-0, doi 10.17632/w4hb5grjwb.3) | ~102k from 13 DBs | the competition's reference; `antibacterial.fasta` (39,448) is its 8–50aa antibacterial subset |
| **GRAMPA** (Witten & Witten) | 6,760 peptides, 51,345 MICs | our MIC regressor |
| **DBAASP v3** | >15,700 entries, MIC + haemolysis + cytotoxicity | LLAMP's source; API returned 403 for us |
| **DRAMP 4.0** | 30,260 entries (11,612 general) | overlaps MarLys |
| **dbAMP 3.0**, **APD6**, **Peptipedia 2.0** | — | aggregated in MarLys |
| **AMPSphere** | 863,498 predicted AMPs from 63,410 metagenomes | pretraining corpus (endpoints 404'd) |
| **HemoPI2** | 1,926 with experimental HC50 | our haemolysis head |
| **ESCAPE** | 82,359 peptides, multilabel | benchmark reference |

---

## 8. Generative model landscape

**VAE**: HydrAMP, PepCVAE, CLaSS, LSSAMP, PepVAE, CVAE-based (Bioinformatics 2025).
**GAN**: AMPGAN, AMPGAN v2, Multi-CGAN, PandoraGAN.
**Diffusion**: AMP-Diffusion, OmegAMP, Diff-AMP, ProT-Diff, MMCD (multi-modal
contrastive), CPL-Diff (ESM-2 latents, mask-controlled length), AMPGen
(evolutionary-information-reserved), knowledge-aware prompt diffusion for
pathogen-specific AMPs, multi-guided latent diffusion for non-haemolytic AMPs.
**Autoregressive / LM**: Hyformer, AMPTrans-LSTM, MoFormer, ProtGPT2 and ProGen2
fine-tuned / prefix-tuned / LoRA'd for AMPs, LLM-based peptide antibiotic design.
**RL / optimisation**: ApexAmphion (6.4B protein LM + RL against a composite MIC +
physicochemical reward), RL with LoRA for targeted AMP design (BiB 2025),
HMAMP (hypervolume-driven multi-objective), MODAN, AMPEMO, MOQA, QMO,
MAC-AMP (multi-agent LLM peer review; *E. coli* activity 0.943 vs 0.831 next best).
**Mining** (not generation, but sets the realism bar): APEX / molecular
de-extinction (10.3M encrypted peptides mined, 37,176 predicted broad-spectrum),
APEX 1.1 on 233 archaeal proteomes (12,623 archaeasins), AMPSphere, ApexOracle
(zero-shot to unseen strains via genomic features).

### 8.1 Wet-lab hit rates achieved by published pipelines (the bar)

| Method | Approach | Synthesised | Hit rate |
|---|---|---|---|
| OmegAMP (2025) | conditional diffusion + XGBoost cascade | 25 | **24/25 = 96%** |
| AMP-Designer (Sci. Adv. 2025) | LLM + prompt tuning | 18 | 94.4% |
| ARCADIAMP (Nat. Commun. 2026) | diffusion, 1M generated -> 76 passed filters | 10 | 8/10 (MIC <= 32 µg/mL) |
| AMPSphere (Cell 2024) | metagenome mining | 100 (earlier: 50) | 54% |
| CLaSS (Nat. Biomed. Eng. 2021) | controlled latent sampling + MD | 20 | 2 highly potent |
| CVAE-BIO (Brief. Bioinform. 2026) | CVAE + random forest | — | 18.5% at MIC <= 10 µg/mL |

Two lessons. First, heavy filtering is normal: ARCADIAMP generated 10^6 and kept
76 (0.008%). Second, generation plus a good classifier cascade substantially
beats mining (96% vs 54%), which is the regime this competition sits in.

### 8.2 A caution on classifier-guided generation
FBGAN-style feedback — score generated sequences with a classifier, retrain the
discriminator on the high scorers — is known to amplify classifier bias, because
the classifier only ever sees the generator's own output, and it does not
constrain potency, haemolysis or mechanism.

**→ Decision.** Our activity conditioning is deliberately *not* that loop. The
conditioning bucket is the adversarial classifier's score on each **real
training peptide**, so it encodes "which database AMPs are most canonically
AMP-like". The classifier never scores a generated sequence during training.

**Discrete diffusion theory** worth noting: D3PM, SEDD, MDLM/MD4/RADD showed the
absorbing-state (masked) formulation collapses to per-position cross-entropy;
EvoDiff applies this to proteins. Continuous diffusion now scales competitively.

**→ Decision.** We use a conditional autoregressive transformer with
classifier-free guidance. Rationale: on a 39,448-sequence corpus of ≤50-mers,
diffusion's advantage (parallel refinement of long sequences) does not apply,
while an AR model is far cheaper to make numerically reproducible — which, as
§9 explains, is the binding constraint.

---

## 9. The constraint nobody writes papers about: exact reproducibility

Co-authorship requires "a fixed default random seed such that running the
generation script twice produces identical output", and the proposal says
organizers regenerate the library **on their own Linux workstation** and compare
it to ours. Any ranked or sampled output is a discrete function of floats, so it
only reproduces if those floats agree. BLAS reduction order depends on CPU
vectorisation width and thread count.

- float32 matmul disagreement ≈ 1e-6 relative → ~1e-5 absolute on logits of
  magnitude 10 — the same order as a sampling decision boundary.
- float64 disagreement ≈ 1e-14.

**→ Decision.** Three hardening measures, all implemented:
1. The generation path contains **no BLAS calls** — sliding-window hydrophobic
   moments and the 2-D conformity KDE are written elementwise.
2. Neural sampling runs in **float64 on CPU** with logits quantized to 1e-6,
   leaving ~8 orders of magnitude of margin; all randomness is NumPy PCG64,
   which is bit-identical across platforms by specification.
3. Every ranking uses a quantized score with a **lexicographic tiebreak on the
   sequence string**, so sub-ulp differences cannot reorder anything.

Gradient-boosted trees were chosen partly for this reason: they are exact
piecewise-constant functions, unlike anything involving matrix multiplication.

---

## 10. Assay realities that shape the top-100

- MIC by CLSI broth microdilution; MIC ceiling 64 µM, HC50 ceiling 128 µM;
  potency threshold **16 µM** (tighter than the 32 µM common in prior work).
- Many AMPs lose activity at physiological salt; assays in standard media can
  overstate in vivo potency. Highly cationic peptides are the most salt-sensitive.
- Serum proteases degrade L-peptides; D-enantiomers are excluded here
  (linear, canonical, free termini only).
- All peptides synthesised by one vendor (AAPPtec) under one workflow, blinded,
  randomised order.

**→ Decision.** We cannot use D-amino acids, amidation, or stapling to buy
stability — everyone faces the same constraint. What we *can* control is
choosing candidates that are synthesisable and whose predicted activity does not
depend on extreme cationicity (the most salt-fragile regime).

---

## 11. Open questions we are still carrying

1. **Top-50 vs top-100.** The website FAQ says the 25 tested peptides are drawn
   from the *top 50*; the NeurIPS proposal and Kaggle both say the *top 100*.
   Materially changes ranking strategy. Needs an organizer answer.
2. **Aggregation weights are held out** until the close of Phase 1 — deliberately,
   to prevent optimisation against the ranking function. Implies a balanced
   strategy across all four metric families rather than maximising any one.
3. **Two different 80% identity checks**: `verify_submission.py` uses
   `Levenshtein.ratio` against `antibacterial.fasta`; the proposal describes
   MMseqs2 pairwise identity against MarLys. We satisfy the stricter with margin.

---

## Reference list

Ordered roughly as cited.

1. Møller-Larsen R. et al. *seqme.* Bioinformatics Advances 6(1):vbag212, 2026. arXiv:2511.04239.
2. Soares D. et al. *OmegAMP.* arXiv:2504.17247, ICLR 2026.
3. Szymczak P. et al. *AI-Driven Antimicrobial Peptide Discovery: Mining and Generation.* Acc. Chem. Res. 58(12):1831–1846, 2025.
4. Szymczak P. et al. *HydrAMP.* Nat. Commun. 14:1453, 2023.
5. Izdebski A. et al. *Synergistic Benefits of Joint Molecule Generation and Property Prediction* (Hyformer). arXiv:2504.16559, TMLR 2026.
6. Torres M.D.T. et al. *Generative latent diffusion language modeling yields anti-infective synthetic peptides.* Cell Biomaterials 1(9), 2025.
7. Sidorczuk K. et al. *Benchmarks in AMP prediction are biased due to the selection of negative data.* Brief. Bioinform. 23(5):bbac343, 2022.
8. *AMPBench-MT: A Homology-Controlled Benchmark.* arXiv:2607.25518, 2026.
9. Pal A. et al. *Coarse composition suffices.* arXiv:2608.30337, 2026.
10. *PathoMIC: A Benchmark for Cross-Species AMP Activity Prediction.* arXiv:2608.26228, 2026.
11. *ESCAPE: A Standardized Benchmark for Multilabel AMP Classification.* arXiv:2511.04814, NeurIPS 2025.
12. Witten J., Witten Z. *Deep learning regression model for AMP design* (GRAMPA). bioRxiv 692681, 2019.
13. Lin Z. et al. *ESM-2.* bioRxiv, 2022. / EvolutionaryScale, *ESM Cambrian*, 2024.
14. Frey N. et al. *Protein Discovery With Discrete Walk-Jump Sampling.* arXiv:2306.12360, 2023.
15. Alaa A. et al. *How Faithful is your Synthetic Data?* arXiv:2102.08921, 2022.
16. Kynkäänniemi T. et al. *Improved precision and recall metric.* NeurIPS 2019.
17. Friedman D., Dieng A.B. *The Vendi Score.* arXiv:2210.02410, 2023.
18. Ospanov A. et al. *Towards a Scalable Reference-Free Evaluation of Generative Models* (FKEA). arXiv:2407.02961, 2024.
19. Pesciullesi G. et al. *Amino acid composition drives aggregation during peptide synthesis.* Nature Chemistry, 2026.
20. *A robust data analytical method to investigate sequence dependence in flow-based peptide synthesis.* PMC10966953, 2024.
21. Chaudhary K. et al. *HemoPI2 / Prediction of hemolytic peptides and their hemolytic concentration.* Commun. Biol., 2025.
22. Timmons P.B., Hewage C.M. *HAPPENN.* Sci. Rep. 10:10869, 2020.
23. Rathore A.S. et al. *ToxinPred 3.0.* Comput. Biol. Med. 179:108926, 2024.
24. *AmpLyze.* arXiv:2507.08162, 2025.
25. Santos-Júnior C.D. et al. *Discovery of antimicrobial peptides in the global microbiome with machine learning* (AMPSphere). Cell, 2024.
26. Wan F. et al. *Deep-learning-enabled antibiotic discovery through molecular de-extinction* (APEX). Nat. Biomed. Eng. 8:854–871, 2024.
27. *Deep learning reveals antibiotics in the archaeal proteome* (APEX 1.1). Nat. Microbiol., 2025.
28. *Predicting and generating antibiotics against future pathogens with ApexOracle.* arXiv:2507.07862, 2025.
29. *A deep reinforcement learning platform for antibiotic discovery* (ApexAmphion). arXiv:2509.18153, 2025.
30. Yan J. et al. *MBC-Attention.* mSystems 8(4):e00345-23, 2023.
31. Dong R. et al. *Exploring the repository of de novo designed bifunctional AMPs through deep learning* (AMPredictor). eLife 13:RP97330, 2025.
32. Pandi A. et al. *Cell-free biosynthesis combined with deep learning.* Nat. Commun. 14:7197, 2023.
33. *LLAMP: AI-guided discovery and optimization of AMPs through species-aware language model.* PMC12271573, 2025.
34. Jin S. et al. *AMPGen.* Commun. Biol. 8:839, 2025.
35. Luo et al. *CPL-Diff.* Advanced Science, 2025.
36. *Controllable generation of predicted non-hemolytic AMPs by multi-guided latent diffusion.* J. Cheminform., 2026.
37. Zhou G. et al. *MAC-AMP.* arXiv:2602.14926, ICLR 2026.
38. *HMAMP: Hypervolume-Driven Multi-Objective AMP Design.* arXiv:2405.00753.
39. Park J. et al. *Reinforcement learning with low-rank adaptation for targeted AMP design.* Brief. Bioinform. 26(6):bbaf641, 2025.
40. Blower R.J. et al. / eCAP length series; *Rational Design of α-Helical AMPs with Enhanced Activities and Specificity/Therapeutic Index*, JBC.
41. *The lexicon of antimicrobial peptides: a complete set of arginine and tryptophan sequences.* Commun. Biol. 4:605, 2021.
42. Hayouka Z. et al. *Antimicrobial random peptide cocktails.* Chem. Commun. 55:2007, 2019; and related RPM work.
43. Chen Y. et al. *Role of helicity of α-helical AMPs to improve specificity.* Protein Cell 5(8):631, 2014.
44. Cutrona K.J. et al. *Role of arginine and lysine in the antimicrobial mechanism of histone-derived AMPs.* FEBS Lett., 2015.
45. Selsted M.E. et al. *Indolicidin* mechanism; *Tryptophan- and arginine-rich AMPs.* 2006.
46. Graf M., Wilson D.N. et al. *Proline-rich AMPs target the ribosome.* NAR 44(5):2439, 2016.
47. Steinegger M., Söding J. *MMseqs2.* Nat. Biotechnol. 35:1026, 2017.
48. Hollmann N. et al. *TabPFN / Accurate predictions on small data with a tabular foundation model.* Nature, 2025; TabPFN-2.5, arXiv:2511.08667.
49. Ho J., Salimans T. *Classifier-Free Diffusion Guidance.* arXiv:2207.12598.
50. Austin J. et al. *D3PM*; Sahoo S. et al. *MDLM*, NeurIPS 2024; Alamdari S. et al. *EvoDiff*.
