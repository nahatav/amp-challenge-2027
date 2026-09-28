"""Train the conditional peptide transformer on GPU, export float64 NumPy weights.

Run:  uv run --extra dev python scripts/train_transformer.py --epochs 60

The corpus is the competition's own 39,448-sequence antibacterial reference. That
is small for a transformer, so the configuration is sized accordingly (a few
million parameters, dropout 0.2, weight decay, early stopping on held-out
perplexity). Over-parameterising here does not help: it memorises, and
memorisation is actively penalised — seqme's AuthPct measures the fraction of
generated sequences whose nearest training neighbour is *further* than that
neighbour's own nearest training point, so a model that copies scores badly, and
exact copies are outright disqualified by the compliance check.

Conditioning is on (length, charge, amphiphilicity) buckets with independent
dropout to a null token, giving classifier-free guidance at sampling time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amp_design import descriptors as D  # noqa: E402
from amp_design.fasta import read_sequences  # noqa: E402
from amp_design.neural import (  # noqa: E402
    ACTIVITY_EDGES,
    AMPH_EDGES,
    BOS,
    CHARGE_EDGES,
    EOS,
    LENGTH_EDGES,
    N_BUCKETS,
    N_COND_AXES,
    NULL_BUCKET,
    PAD,
    bucketize,
    build_torch_model,
    condition_tokens,
    encode_sequence,
    export_numpy_weights,
    null_condition,
)


def build_dataset(sequences: list[str], max_len: int, activity: np.ndarray | None = None):
    import torch

    lengths = np.array([len(s) for s in sequences], dtype=np.float64)
    charge = D.net_charge(sequences)
    amph = D.hydrophobic_moment(sequences)

    activity_b = (
        bucketize(activity, ACTIVITY_EDGES)
        if activity is not None
        else np.full(len(sequences), NULL_BUCKET, dtype=np.int64)
    )

    cond = condition_tokens(
        bucketize(lengths, LENGTH_EDGES),
        bucketize(charge, CHARGE_EDGES),
        bucketize(amph, AMPH_EDGES),
        activity_b,
    )

    T = max_len + 2  # BOS + residues + EOS
    tokens = np.full((len(sequences), T), PAD, dtype=np.int64)
    for i, s in enumerate(sequences):
        ids = [BOS] + encode_sequence(s) + [EOS]
        tokens[i, : len(ids)] = ids

    return torch.from_numpy(tokens), torch.from_numpy(cond)


def main() -> None:
    import torch
    import torch.nn.functional as F

    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/antibacterial.fasta")
    ap.add_argument("--out", default="checkpoint/transformer.npz")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cond-dropout", type=float, default=0.15)
    ap.add_argument("--oracles", default="checkpoint/oracles.pkl")
    ap.add_argument("--extra-corpus", nargs="*", default=["../data/dramp_general.fasta"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    sequences = read_sequences(args.reference)
    if args.extra_corpus:
        ref_set = set(sequences)
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        for path in args.extra_corpus:
            extra = [
                s for s in read_sequences(path)
                if 8 <= len(s) <= 50 and set(s) <= valid and s not in ref_set
            ]
            print(f"  + {len(extra):,} new sequences from {path}")
            sequences.extend(extra)
            ref_set.update(extra)
    print(f"corpus: {len(sequences):,} sequences")

    # Score the corpus with the adversarial AMP classifier so the model can be
    # conditioned on how canonically AMP-like each training peptide is.
    activity = None
    if args.oracles and Path(args.oracles).exists():
        from amp_design.oracles import OracleEnsemble
        from amp_design.features import featurize

        oracles = OracleEnsemble.load(args.oracles)
        print("  scoring corpus with the AMP classifier for activity conditioning...")
        activity = oracles.amp_probability(featurize(sequences))
        print(f"  activity: mean {activity.mean():.3f}, "
              f"frac>0.72 {np.mean(activity > 0.72):.3f}")

    max_len = max(len(s) for s in sequences)
    tokens, cond = build_dataset(sequences, max_len, activity)
    seq_len = tokens.shape[1]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(sequences))
    n_val = int(0.05 * len(sequences))
    val_idx = torch.from_numpy(perm[:n_val])
    tr_idx = torch.from_numpy(perm[n_val:])
    print(f"train {len(tr_idx):,} | val {len(val_idx):,}")

    model = build_torch_model(
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
        max_len=seq_len + 3, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    steps_per_epoch = max(len(tr_idx) // args.batch_size, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1
    )

    null = torch.from_numpy(null_condition(1)[0]).to(device)

    def run_batch(idx, train: bool):
        tk = tokens[idx].to(device)
        cd = cond[idx].to(device)
        if train and args.cond_dropout > 0:
            # Drop each conditioning axis independently -> classifier-free guidance.
            drop = torch.rand(cd.shape, device=device) < args.cond_dropout
            cd = torch.where(drop, null[None].expand_as(cd), cd)

        logits = model(tk[:, :-1], cd)          # (B, 3 + T - 1, V)
        logits = logits[:, N_COND_AXES - 1 : -1]   # align: predict token t from t-1
        target = tk[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=PAD
        )
        return loss

    best_val, best_state, patience = float("inf"), None, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = tr_idx[torch.randperm(len(tr_idx))]
        total, nb = 0.0, 0
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            loss = run_batch(order[i : i + args.batch_size], train=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
            nb += 1

        model.eval()
        with torch.no_grad():
            vt, vb = 0.0, 0
            for i in range(0, len(val_idx), args.batch_size):
                vt += run_batch(val_idx[i : i + args.batch_size], train=False).item()
                vb += 1
        train_loss, val_loss = total / max(nb, 1), vt / max(vb, 1)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"ppl {np.exp(val_loss):.3f}  ({time.time()-t0:.0f}s)")

        if patience >= 12:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best val loss {best_val:.4f} (perplexity {np.exp(best_val):.3f})")

    export_numpy_weights(
        model, args.out,
        {"d_model": args.d_model, "n_layers": args.layers,
         "n_heads": args.heads, "max_len": seq_len + 3},
    )
    size = Path(args.out).stat().st_size / 1e6
    print(f"exported NumPy weights -> {args.out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
