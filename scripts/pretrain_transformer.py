"""Pretrain the generator on a broad peptide corpus, then fine-tune on AMPs.

The v2 generator is 4.8M parameters trained only on 41,412 AMPs, and its
train/val gap (1.61 vs 1.72) says it is already at the limit of what that much
data supports. Adding capacity alone would memorise rather than generalise, and
memorisation is directly penalised (seqme's AuthPct, plus the compliance check
rejects verbatim reference reuse).

The scorecard localises the deficit precisely: against the reference AMPs we are
strong on fidelity (FBD 1.02, Precision 0.78) but weak on *coverage* —
FKEA 651 vs the reference's 779, Recall 0.73 vs 0.88. Coverage is what a small
model trained on a narrow corpus fails at: it learns the bulk of the
distribution and misses its tails.

So: pretrain on 2.5M SwissProt-derived peptides to learn general peptide
sequence structure, then fine-tune on the AMP corpus with the conditioning
tokens. This is the documented recipe — Hyformer pretrains on 3.5M general
peptides plus 1M AMPs before fine-tuning on 4,547 with measured MIC; PepBERT
pretrains on 1.97-19.2M peptides.

Run:
    scripts/pretrain_transformer.py --stage pretrain --epochs 6
    scripts/pretrain_transformer.py --stage finetune --epochs 80
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

MAX_LEN = 50


def encode_batch(sequences: list[str], conditioned: bool):
    """Return (tokens, cond) tensors. Unconditioned data gets all-null tokens."""
    import torch

    T = MAX_LEN + 2
    tokens = np.full((len(sequences), T), PAD, dtype=np.int64)
    for i, s in enumerate(sequences):
        ids = [BOS] + encode_sequence(s) + [EOS]
        tokens[i, : len(ids)] = ids[:T]

    if conditioned:
        lengths = np.array([len(s) for s in sequences], dtype=np.float64)
        cond = condition_tokens(
            bucketize(lengths, LENGTH_EDGES),
            bucketize(D.net_charge(sequences), CHARGE_EDGES),
            bucketize(D.hydrophobic_moment(sequences), AMPH_EDGES),
        )
    else:
        cond = null_condition(len(sequences))
    return torch.from_numpy(tokens), torch.from_numpy(cond)


def run(args) -> None:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | stage: {args.stage}")

    if args.stage == "pretrain":
        corpus = Path(args.corpus).read_text(encoding="utf-8").split("\n")
        corpus = [s for s in corpus if s]
        conditioned = False
    else:
        corpus = read_sequences(args.reference)
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        for extra in args.extra_corpus or []:
            if Path(extra).exists():
                known = set(corpus)
                corpus += [s for s in read_sequences(extra)
                           if 8 <= len(s) <= 50 and set(s) <= valid and s not in known]
        conditioned = True
    print(f"corpus: {len(corpus):,} sequences")

    tokens, cond = encode_batch(corpus, conditioned)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(corpus))
    n_val = min(int(0.02 * len(corpus)), 20_000)
    val_idx = torch.from_numpy(perm[:n_val])
    tr_idx = torch.from_numpy(perm[n_val:])
    print(f"train {len(tr_idx):,} | val {len(val_idx):,}")

    model = build_torch_model(
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
        max_len=MAX_LEN + 2 + N_COND_AXES + 2, dropout=args.dropout,
    ).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    if args.init_from and Path(args.init_from).exists():
        data = np.load(args.init_from)
        sd = {k: torch.from_numpy(data[k]) for k in data.files if not k.startswith("cfg_")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"initialised from {args.init_from} "
              f"(missing {len(missing)}, unexpected {len(unexpected)})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    steps = args.epochs * max(len(tr_idx) // args.batch_size, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    null = torch.from_numpy(null_condition(1)[0]).to(device)

    def batch_loss(idx, train: bool):
        tk = tokens[idx].to(device, non_blocking=True)
        cd = cond[idx].to(device, non_blocking=True)
        if train and conditioned and args.cond_dropout > 0:
            drop = torch.rand(cd.shape, device=device) < args.cond_dropout
            cd = torch.where(drop, null[None].expand_as(cd), cd)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(tk[:, :-1], cd)[:, N_COND_AXES - 1 : -1]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                tk[:, 1:].reshape(-1), ignore_index=PAD,
            )
        return loss

    best, best_state, patience = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = tr_idx[torch.randperm(len(tr_idx))]
        tot = nb = 0
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            loss = batch_loss(order[i : i + args.batch_size], True)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); nb += 1

        model.eval()
        with torch.no_grad():
            vt = vb = 0
            for i in range(0, len(val_idx), args.batch_size):
                vt += batch_loss(val_idx[i : i + args.batch_size], False).item(); vb += 1
        vl = vt / max(vb, 1)
        print(f"  epoch {epoch:3d}  train {tot/max(nb,1):.4f}  val {vl:.4f}  "
              f"ppl {np.exp(vl):.3f}  ({time.time()-t0:.0f}s)", flush=True)

        if vl < best - 1e-4:
            best, best_state, patience = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop at epoch {epoch}"); break

    if best_state:
        model.load_state_dict(best_state)
    print(f"best val {best:.4f} (ppl {np.exp(best):.3f})")
    export_numpy_weights(model, args.out, {
        "d_model": args.d_model, "n_layers": args.layers,
        "n_heads": args.heads, "max_len": MAX_LEN + 2 + N_COND_AXES + 2,
    })
    print(f"exported -> {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--corpus", default="../data/pretrain_peptides.txt")
    ap.add_argument("--reference", default="data/antibacterial.fasta")
    ap.add_argument("--extra-corpus", nargs="*", default=["../data/dramp_general.fasta"])
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--cond-dropout", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
