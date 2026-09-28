"""Fast GPU sampling from the trained transformer, for offline evaluation.

This is NOT the submission path — `generate` samples in NumPy float64 on CPU so
the output reproduces across machines. This script exists to answer one
question cheaply: does the transformer produce a better library than the
backoff Markov baseline, judged by the real seqme metrics? Sampling 50k
sequences here takes seconds instead of hours.

Run:
    PYTHONPATH=src <eval-python> scripts/sample_transformer.py \
        --n 50000 --out ../eval/transformer_library.fasta
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amp_design.fasta import read_sequences, write_fasta  # noqa: E402
from amp_design import descriptors as D  # noqa: E402
from amp_design.neural import (  # noqa: E402
    AMPH_EDGES,
    BOS,
    CHARGE_EDGES,
    EOS,
    LENGTH_EDGES,
    N_BUCKETS,
    NULL_BUCKET,
    PAD,
    VOCAB_SIZE,
    bucketize,
    build_torch_model,
    condition_tokens,
    decode_tokens,
    null_condition,
)


def sample_gpu(model, cond, *, device, temperature, guidance, max_length, min_length,
               batch_size, seed):
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    null = torch.from_numpy(null_condition(1)[0]).to(device)
    out: list[str] = []

    for start in range(0, len(cond), batch_size):
        cb = torch.from_numpy(cond[start : start + batch_size]).to(device)
        B = cb.shape[0]
        tokens = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_length):
            logits = model(tokens, cb)[:, -1]
            if guidance != 1.0:
                un = model(tokens, null[None].expand(B, -1))[:, -1]
                logits = un + guidance * (logits - un)
            logits = logits / max(temperature, 1e-6)

            logits[:, PAD] = -1e30
            logits[:, BOS] = -1e30
            if step < min_length:
                logits[:, EOS] = -1e30

            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=g).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            done = done | (nxt == EOS)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
            if bool(done.all()):
                break

        for row in tokens[:, 1:].cpu().numpy():
            out.append(decode_tokens(row.tolist()))
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint/transformer.npz")
    ap.add_argument("--reference", default="data/antibacterial.fasta")
    ap.add_argument("--out", default="../eval/transformer_library.fasta")
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--cond-mode", default="empirical",
                    choices=["empirical", "targeted", "null"])
    ap.add_argument("--activity-bucket", type=int, default=None,
                    help="Condition on this activity bucket (0-7); omit for null.")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.checkpoint)
    cfg = {k[4:]: int(data[k][0]) for k in data.files if k.startswith("cfg_")}
    model = build_torch_model(
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], max_len=cfg["max_len"], dropout=0.0,
    ).to(device)
    model.load_state_dict(
        {k: torch.from_numpy(data[k]) for k in data.files if not k.startswith("cfg_")},
        strict=False,
    )
    model.eval()

    rng = np.random.default_rng(args.seed)
    refs = read_sequences(args.reference)

    if args.cond_mode == "empirical":
        # Draw conditioning from the reference AMPs' own joint bucket
        # distribution: reproduces the target marginals rather than a guess.
        lb = bucketize(np.array([len(s) for s in refs], dtype=np.float64), LENGTH_EDGES)
        cb = bucketize(D.net_charge(refs), CHARGE_EDGES)
        ab = bucketize(D.hydrophobic_moment(refs), AMPH_EDGES)
        pick = rng.choice(len(refs), size=args.n, replace=True)
        act = (np.full(args.n, args.activity_bucket, dtype=np.int64)
               if args.activity_bucket is not None else None)
        cond = condition_tokens(lb[pick], cb[pick], ab[pick], act)
    elif args.cond_mode == "targeted":
        # Push toward the high-density core of the AMP descriptor cloud, which
        # is what seqme's ConformityScore rewards.
        act = (np.full(args.n, args.activity_bucket, dtype=np.int64)
               if args.activity_bucket is not None else None)
        cond = condition_tokens(
            rng.choice([2, 3, 4], size=args.n, p=[0.3, 0.4, 0.3]),
            rng.choice([1, 2, 3], size=args.n, p=[0.3, 0.4, 0.3]),
            rng.choice([3, 4, 5], size=args.n, p=[0.3, 0.45, 0.25]),
            act,
        )
    else:
        cond = null_condition(args.n)

    t0 = time.time()
    with torch.no_grad():
        seqs = sample_gpu(
            model, cond, device=device, temperature=args.temperature,
            guidance=args.guidance, max_length=50, min_length=8,
            batch_size=args.batch_size, seed=args.seed,
        )
    dt = time.time() - t0

    valid = [s for s in seqs if 8 <= len(s) <= 50]
    uniq = sorted(set(valid))
    print(f"sampled {len(seqs):,} in {dt:.0f}s | valid {len(valid):,} | unique {len(uniq):,}")
    print(f"mean length {np.mean([len(s) for s in uniq]):.1f} | "
          f"charge {D.net_charge(uniq).mean():.2f} | "
          f"amphiphilicity {D.hydrophobic_moment(uniq).mean():.3f}")

    ref_set = set(refs)
    overlap = len(set(uniq) & ref_set)
    print(f"exact overlap with reference: {overlap}")

    write_fasta(uniq, args.out)
    print(f"wrote {len(uniq):,} -> {args.out}")


if __name__ == "__main__":
    main()
