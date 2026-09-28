"""Train the activity / potency oracles used to rank the top-100.

Two models:

1. **AMP classifier** — is this sequence antimicrobial at all?
   Positives: the 39,448 antibacterial reference peptides.
   Negatives: a deliberately adversarial mixture, following OmegAMP
   (Soares et al. 2025), whose classifier is trained against
     * UniProt short peptides without antimicrobial annotation,
     * **shuffled AMPs** — identical amino-acid composition, destroyed order,
     * **mutated AMPs** — 5 random substitutions,
     * uniformly random sequences.
   The shuffled negatives matter most: they force the model to key on residue
   *arrangement* (amphipathic periodicity) rather than composition alone. Since
   the competition's surrogate ensemble likely includes classifiers trained the
   same way, a generator that only matches composition would be scored as a
   false positive. Our hydrophobic-moment and CTD-transition features are
   order-sensitive precisely so this signal is learnable.

2. **MIC regressor** — how potent, per species?
   GRAMPA (Witten & Witten): 51,345 MIC measurements over 6,760 peptides.
   Target is log10(MIC / micromolar). Species enters as a one-hot covariate, as
   in LLAMP and PathoMIC, both of which find species conditioning necessary.

Homology-aware splitting: AMPBench-MT and Sidorczuk et al. both show that random
row splits inflate AMP model scores badly. We split by a cheap sequence-identity
clustering so near-duplicates cannot straddle train and test.

Run:  uv run --extra dev python scripts/train_oracles.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amp_design.constants import AMINO_ACIDS, MAX_LENGTH, MIN_LENGTH  # noqa: E402
from amp_design.fasta import read_sequences  # noqa: E402
from amp_design.features import featurize  # noqa: E402

# Species in the competition panel, mapped to how GRAMPA names them.
PANEL_SPECIES = [
    "E. coli", "S. aureus", "P. aeruginosa", "K. pneumoniae",
    "A. baumannii", "E. faecalis", "E. faecium", "B. subtilis",
    "S. typhimurium", "E. cloacae",
]


# --------------------------------------------------------------------------
# Negative-set construction
# --------------------------------------------------------------------------
def make_negatives(amps: list[str], uniprot: list[str], rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Return (sequences, source_labels)."""
    seqs: list[str] = []
    src: list[str] = []

    for s in uniprot:
        if MIN_LENGTH <= len(s) <= MAX_LENGTH and set(s) <= set(AMINO_ACIDS):
            seqs.append(s)
            src.append("uniprot")

    # Shuffled AMPs: same composition, destroyed order.
    picks = rng.choice(len(amps), size=min(20_000, len(amps)), replace=False)
    for i in picks:
        chars = list(amps[i])
        rng.shuffle(chars)
        seqs.append("".join(chars))
        src.append("shuffled")

    # Mutated AMPs: 5 random substitutions.
    picks = rng.choice(len(amps), size=min(10_000, len(amps)), replace=False)
    alphabet = np.array(list(AMINO_ACIDS))
    for i in picks:
        chars = list(amps[i])
        n_mut = min(5, len(chars))
        pos = rng.choice(len(chars), size=n_mut, replace=False)
        for p in pos:
            chars[p] = str(rng.choice(alphabet))
        seqs.append("".join(chars))
        src.append("mutated")

    # Uniformly random sequences with an AMP-like length profile.
    lengths = rng.choice([len(a) for a in amps], size=10_000)
    for L in lengths:
        seqs.append("".join(rng.choice(alphabet, size=int(L))))
        src.append("random")

    return seqs, src


# --------------------------------------------------------------------------
# Cheap homology-aware grouping
# --------------------------------------------------------------------------
def cluster_ids(sequences: list[str], k: int = 5) -> np.ndarray:
    """Assign a group id by shared k-mer signature (a poor man's MMseqs2).

    Sequences sharing their lexicographically smallest k-mer land in the same
    group. This is far weaker than real clustering, but it is enough to stop
    exact-and-near duplicates from being split across the train/test boundary,
    which is the failure mode the benchmark papers warn about.
    """
    groups: dict[str, int] = {}
    out = np.empty(len(sequences), dtype=np.int64)
    for i, s in enumerate(sequences):
        kmers = [s[j : j + k] for j in range(max(len(s) - k + 1, 1))]
        sig = min(kmers) if kmers else s
        if sig not in groups:
            groups[sig] = len(groups)
        out[i] = groups[sig]
    return out


def grouped_split(groups: np.ndarray, test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_test = int(len(uniq) * test_frac)
    test_groups = set(uniq[:n_test].tolist())
    mask = np.array([g in test_groups for g in groups])
    return ~mask, mask


# --------------------------------------------------------------------------
def train_classifier(args, rng) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

    amps = read_sequences(args.reference)
    uniprot = read_sequences(args.uniprot)
    print(f"  positives: {len(amps):,}")

    negs, neg_src = make_negatives(amps, uniprot, rng)
    print(f"  negatives: {len(negs):,} " + str({s: neg_src.count(s) for s in set(neg_src)}))

    sequences = amps + negs
    y = np.concatenate([np.ones(len(amps)), np.zeros(len(negs))])

    print("  featurizing...")
    X = featurize(sequences)
    print(f"  design matrix: {X.shape}")

    groups = cluster_ids(sequences)
    tr, te = grouped_split(groups, test_frac=0.2, rng=rng)
    print(f"  train {tr.sum():,} / test {te.sum():,} (grouped by k-mer signature)")

    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
        l2_regularization=1.0, random_state=args.seed,
    )
    clf.fit(X[tr], y[tr])

    p = clf.predict_proba(X[te])[:, 1]
    metrics = {
        "auroc": float(roc_auc_score(y[te], p)),
        "auprc": float(average_precision_score(y[te], p)),
        "mcc": float(matthews_corrcoef(y[te], (p > 0.5).astype(int))),
    }
    print(f"  AMP classifier: AUROC {metrics['auroc']:.4f}  AUPRC {metrics['auprc']:.4f}  MCC {metrics['mcc']:.4f}")

    # Per-negative-source false-positive rate: the number that matters most is
    # the FPR on shuffled AMPs, since that is the adversarial case.
    neg_src_arr = np.array(["positive"] * len(amps) + neg_src)
    for s in sorted(set(neg_src)):
        m = te & (neg_src_arr == s)
        if m.sum():
            fpr = float((p[neg_src_arr[te] == s] > 0.5).mean())
            print(f"    FPR on {s:<9}: {fpr:.4f}  (n={m.sum():,})")
            metrics[f"fpr_{s}"] = fpr

    return {"model": clf, "metrics": metrics}


def train_mic(args, rng) -> dict:
    import csv

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import spearmanr

    rows = list(csv.DictReader(open(args.grampa, encoding="utf-8")))
    valid = set(AMINO_ACIDS)

    recs = []
    for r in rows:
        seq = r["sequence"].strip().upper()
        if not (MIN_LENGTH <= len(seq) <= MAX_LENGTH) or not set(seq) <= valid:
            continue
        if r.get("has_unusual_modification") == "True":
            continue
        try:
            val = float(r["value"])
        except (TypeError, ValueError):
            continue
        recs.append((seq, r["bacterium"], val))

    print(f"  usable MIC rows: {len(recs):,}")

    # Collapse replicate measurements to the median per (sequence, species).
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for seq, bug, val in recs:
        agg[(seq, bug)].append(val)
    pairs = [(s, b, float(np.median(v))) for (s, b), v in agg.items()]
    print(f"  unique (sequence, species) pairs: {len(pairs):,}")

    species = sorted({b for _, b, _ in pairs})
    sp_index = {s: i for i, s in enumerate(species)}
    print(f"  species: {len(species)}")

    seqs = [s for s, _, _ in pairs]
    uniq_seqs = sorted(set(seqs))
    seq_pos = {s: i for i, s in enumerate(uniq_seqs)}
    print(f"  featurizing {len(uniq_seqs):,} unique sequences...")
    F = featurize(uniq_seqs)

    X = np.zeros((len(pairs), F.shape[1] + len(species)), dtype=np.float64)
    yv = np.empty(len(pairs))
    for i, (s, b, v) in enumerate(pairs):
        X[i, : F.shape[1]] = F[seq_pos[s]]
        X[i, F.shape[1] + sp_index[b]] = 1.0
        yv[i] = v

    groups = cluster_ids(seqs)
    tr, te = grouped_split(groups, test_frac=0.2, rng=rng)
    print(f"  train {tr.sum():,} / test {te.sum():,}")

    reg = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
        l2_regularization=1.0, random_state=args.seed,
    )
    reg.fit(X[tr], yv[tr])
    pred = reg.predict(X[te])

    metrics = {
        "mae": float(mean_absolute_error(yv[te], pred)),
        "r2": float(r2_score(yv[te], pred)),
        "spearman": float(spearmanr(yv[te], pred).statistic),
    }
    print(f"  MIC regressor (log10 uM): MAE {metrics['mae']:.3f}  R2 {metrics['r2']:.3f}  rho {metrics['spearman']:.3f}")

    for sp in PANEL_SPECIES:
        if sp not in sp_index:
            continue
        m = te & np.array([b == sp for _, b, _ in pairs])
        if m.sum() > 30:
            pr = reg.predict(X[m])
            print(f"    {sp:<18} n={m.sum():<5} MAE {mean_absolute_error(yv[m], pr):.3f}  "
                  f"rho {spearmanr(yv[m], pr).statistic:.3f}")

    return {"model": reg, "species": species, "n_features": F.shape[1], "metrics": metrics}


def train_hemolysis(args, rng) -> dict:
    """HC50 regression, for the Optimal Selectivity category.

    Data: HemoPI2 (Rathore et al.), 1,926 peptides with experimentally measured
    HC50 against mammalian red blood cells. Target is log10(HC50 / micromolar);
    higher is safer. The competition caps HC50 at 128 uM ("non-haemolytic"), so
    values above that are censored to the cap rather than extrapolated.

    Note the ceiling effect: the Safety Window is HC50/MIC50 and HC50 saturates,
    so beyond the cap the only way to improve selectivity is to lower MIC. This
    is why the selectivity score below combines a haemolysis prediction with the
    potency prediction rather than optimising haemolysis alone.
    """
    import csv

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import spearmanr

    valid = set(AMINO_ACIDS)
    seqs, ys = [], []
    for path in (args.hemo_cv, args.hemo_ind):
        if not Path(path).exists():
            print(f"  (missing {path}, skipping)")
            continue
        for r in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
            seq = (r.get("SEQUENCE") or "").strip().upper()
            raw = r.get("μM") or r.get("uM") or ""
            if not (MIN_LENGTH <= len(seq) <= MAX_LENGTH) or not set(seq) <= valid:
                continue
            try:
                hc50 = float(raw)
            except (TypeError, ValueError):
                continue
            if hc50 <= 0:
                continue
            seqs.append(seq)
            ys.append(np.log10(min(hc50, 128.0)))  # censor at the assay ceiling

    if len(seqs) < 200:
        print(f"  only {len(seqs)} usable rows; skipping hemolysis oracle")
        return {}

    print(f"  usable HC50 rows: {len(seqs):,}")
    X = featurize(seqs)
    y = np.asarray(ys)

    groups = cluster_ids(seqs)
    tr, te = grouped_split(groups, test_frac=0.2, rng=rng)
    print(f"  train {tr.sum():,} / test {te.sum():,}")

    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=2.0, random_state=args.seed,
    )
    reg.fit(X[tr], y[tr])
    pred = reg.predict(X[te])
    metrics = {
        "mae": float(mean_absolute_error(y[te], pred)),
        "r2": float(r2_score(y[te], pred)),
        "spearman": float(spearmanr(y[te], pred).statistic),
    }
    print(f"  HC50 regressor (log10 uM): MAE {metrics['mae']:.3f}  "
          f"R2 {metrics['r2']:.3f}  rho {metrics['spearman']:.3f}")
    return {"model": reg, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/antibacterial.fasta")
    parser.add_argument("--uniprot", default="../data/uniprot_neg.fasta")
    parser.add_argument("--grampa", default="../data/grampa.csv")
    parser.add_argument("--hemo-cv", default="../data/hemopi2_cv.csv")
    parser.add_argument("--hemo-ind", default="../data/hemopi2_ind.csv")
    parser.add_argument("--out", default="checkpoint/oracles.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("[1/3] AMP classifier")
    clf_bundle = train_classifier(args, rng)

    print("[2/3] MIC regressor")
    mic_bundle = train_mic(args, rng)

    print("[3/3] HC50 (hemolysis) regressor")
    hemo_bundle = train_hemolysis(args, rng)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(
            {"classifier": clf_bundle, "mic": mic_bundle, "hemolysis": hemo_bundle}, fh
        )
    print(f"\nSaved oracles -> {out} ({out.stat().st_size/1e6:.1f} MB)")

    Path("checkpoint/oracle_metrics.json").write_text(
        json.dumps(
            {
                "classifier": clf_bundle["metrics"],
                "mic": mic_bundle["metrics"],
                "hemolysis": hemo_bundle.get("metrics", {}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
