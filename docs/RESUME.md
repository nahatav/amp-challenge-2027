# Resume point — 28 Sept 2026, ~17:30 EDT

Paused for a wifi outage. Nothing network-dependent is running.

---

## STOPPED — nothing is running

All processes terminated cleanly for laptop shutdown. Verified: no python, no
git, GPU at 0%.

The generation run (PID 28340) was stopped part-way through round 2 of 4. It
writes its output only at the very end, so **the committed `generate/` files are
untouched and still valid**: 50,000 library sequences + 100 top candidates,
all compliance checks passing.

Working tree is clean at `8f992a7`.

### To restart the run

```bash
cd C:\Users\Valmi\amp-challenge\submission
uv run generate
```

Takes roughly 4 hours (4 sampling rounds, ~65 min each). Nothing is lost by the
interruption except wall-clock — the library is deterministic, so a fresh run
reproduces exactly what the interrupted one would have produced.

**What that run is for:** the committed top-100 was selected with the *buggy*
similarity screen (worst ratio 0.783 against a 0.72 intended margin). It is
still compliant — the hard limit is 0.80 — but 7 of its 100 entries would be
rejected under the corrected screen. The re-run applies the fix. **So the
current submission is valid and submittable as-is; the re-run improves it.**

### Then verify

```bash
uv run python scripts/selfcheck.py --check-reproducibility
```

Expect "ALL CHECKS PASSED" with worst similarity <= 0.72.

---

## Unfinished: APEX-pathogen weights

**Why it matters.** Our oracle ranks curated-database peptides well
(ρ 0.611 held-out on GRAMPA) but **failed an external test**: on the 46
AMP-Diffusion peptides measured on the real competition panel
(`data/baselines/experimental_mic.csv`, zero overlap with our training data),
Spearman vs measured activity was only **0.062** for predicted success rate and
**0.241** for the composite. APEX-pathogen was trained by the lab that will run
our assays, on 11 of our 20 strains at strain level, and produced the
AMP-Diffusion paper's reported MICs. If it ranks those 46 where ours cannot, it
should become the top-100 ranker.

**State.** Code is ready at `research/apex/`:
`apex_score.py` (written, standalone, no biopython needed), plus upstream
`APEX_models.py`, `utils.py`, `aaindex1.csv`, `LICENSE`.
**Missing: the 8 weight files (~230 MB) in `research/apex/APEX_pathogen_models/`.**

Two download routes, both failed today:
- GitHub LFS via `szczurek-lab/ampdiffusion-starter-kit` — throttled to ~10 KB/s
  (`research/apex/lfs_manifest.json` holds resolved oids; hrefs expire, re-resolve)
- GitLab `machine-biology-group-public/apex-pathogen` — reached 94 MB then
  disconnected. **This route was much faster; retry it first:**

```bash
cd C:\Users\Valmi\amp-challenge\research
git clone --depth 1 https://gitlab.com/machine-biology-group-public/apex-pathogen.git apex-gitlab
cp apex-gitlab/APEX_pathogen_models/* apex/APEX_pathogen_models/
```

Then the decisive experiment:
```bash
cd C:\Users\Valmi\amp-challenge\research\apex
# score the 46 wet-lab peptides and correlate with measured MIC
```
(Validation snippet is in the conversation; rebuild from
`data/baselines/experimental_mic.csv` — 506 rows, 46 peptides x 11 strains.)

Note `torch.load` needs `weights_only=False` — the checkpoints are pickled
`nn.Module` objects and torch 2.11 defaults to True.

---

## Next steps, in priority order

1. **Verify the finished run** (automatic, or the selfcheck command above).
   Expect worst similarity to drop from 0.783 to <= 0.72 after the filter fix.
2. **Download APEX** and run the 46-peptide validation. This decides whether we
   re-rank the top-100.
3. **Benchmark against the official baseline.** `data/baselines/hydramp_library.fasta`
   is HydrAMP's real 50,000-sequence library from the starter kit. Run
   `eval/evaluate.py --extra "HydrAMP=../data/baselines/hydramp_library.fasta"`
   to place us directly against a declared baseline on seqme metrics.
4. **Push the repo and submit** — see `SUBMISSION_CHECKLIST.md`.
   Register with **vnahata@ucsd.edu** (gmail is rejected by the rules).

---

## Deadline

**1 October 2026, AOE.** Comfortable margin.
