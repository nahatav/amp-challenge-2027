"""Verify the NumPy float64 sampler reproduces the torch model's logits.

The entry point samples with `NumpyPeptideLM`, not with torch, so the two
implementations must agree. A silent divergence here would mean we ship a
library drawn from a different distribution than the one we trained and
evaluated — the kind of bug that produces a plausible-looking but much weaker
submission.

Run:  PYTHONPATH=src python scripts/check_numpy_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amp_design.fasta import read_sequences  # noqa: E402
from amp_design import descriptors as D  # noqa: E402
from amp_design.neural import (  # noqa: E402
    AMPH_EDGES,
    BOS,
    CHARGE_EDGES,
    LENGTH_EDGES,
    NumpyPeptideLM,
    bucketize,
    build_torch_model,
    condition_tokens,
    encode_sequence,
)


def main() -> int:
    import torch

    ckpt = Path("checkpoint/transformer.npz")
    data = np.load(ckpt)
    cfg = {
        "d_model": int(data["cfg_d_model"][0]),
        "n_layers": int(data["cfg_n_layers"][0]),
        "n_heads": int(data["cfg_n_heads"][0]),
        "max_len": int(data["cfg_max_len"][0]),
    }
    print(f"config: {cfg}")

    torch_model = build_torch_model(
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], max_len=cfg["max_len"], dropout=0.0,
    )
    state = {k: torch.from_numpy(data[k]) for k in data.files if not k.startswith("cfg_")}
    missing, unexpected = torch_model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  state_dict missing={missing} unexpected={unexpected}")
    torch_model.eval()

    np_model = NumpyPeptideLM.load(ckpt)

    seqs = read_sequences("data/antibacterial.fasta")[:16]
    lengths = np.array([len(s) for s in seqs], dtype=np.float64)
    cond = condition_tokens(
        bucketize(lengths, LENGTH_EDGES),
        bucketize(D.net_charge(seqs), CHARGE_EDGES),
        bucketize(D.hydrophobic_moment(seqs), AMPH_EDGES),
    )

    # Feed a BOS-prefixed prefix of each sequence.
    T = 10
    tokens = np.array([[BOS] + encode_sequence(s)[: T - 1] for s in seqs], dtype=np.int64)

    with torch.no_grad():
        tl = torch_model(torch.from_numpy(tokens), torch.from_numpy(cond))[:, -1].numpy()
    nl = np_model.forward(tokens, cond)

    diff = np.abs(tl.astype(np.float64) - nl)
    print(f"  logits shape torch {tl.shape} numpy {nl.shape}")
    print(f"  max |diff| = {diff.max():.3e}   mean = {diff.mean():.3e}")
    print(f"  argmax agreement = {(tl.argmax(-1) == nl.argmax(-1)).mean():.3f}")

    # float32 weights upcast to float64: a few 1e-3 of disagreement is expected
    # and does not affect which token is drawn.
    ok = diff.max() < 1e-2 and (tl.argmax(-1) == nl.argmax(-1)).all()
    print("  torch/numpy parity:", "OK" if ok else "FAILED")

    # --- KV cache vs full recompute ---------------------------------------
    # Sampling drives `forward_step` with a cache; a cache bug would silently
    # change the sampled distribution while still producing plausible peptides.
    # This compares it against the full recompute on identical input.
    B, T = 8, 12
    rng = np.random.default_rng(0)
    cond2 = condition_tokens(np.full(B, 3), np.full(B, 4), np.full(B, 4))
    body = rng.integers(3, 23, size=(B, T))
    tokens2 = np.concatenate([np.full((B, 1), BOS), body], axis=1)

    full = np_model.forward(tokens2, cond2)

    cache = np_model.new_cache(B, 60)
    prefix = np.concatenate(
        [np_model.w["cond.weight"][cond2],
         np_model.w["tok.weight"][np.full((B, 1), BOS)]], axis=1
    )
    inc = np_model.forward_step(prefix, cache)
    for t in range(T):
        emb = np_model.w["tok.weight"][body[:, t]][:, None, :]
        inc = np_model.forward_step(emb, cache)

    cache_diff = np.abs(full - inc).max()
    cache_ok = cache_diff < 1e-9
    print(f"  KV cache vs full recompute: max |diff| = {cache_diff:.3e} "
          f"({'OK' if cache_ok else 'MISMATCH'})")

    # This is also the empirical determinism margin: float64 roundoff here is
    # ~1e-14, against a 1e-6 logit quantisation grid used when sampling.
    ok = ok and cache_ok
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
