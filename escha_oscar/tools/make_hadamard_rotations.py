#!/usr/bin/env python3
"""Generate the uncalibrated (Hadamard) rotation tensors OSCAR INT2 KV expects.

These are the files pointed at by SGLANG_OSCAR_K_ROTATION_PATH and
SGLANG_OSCAR_V_ROTATION_PATH.  No calibration data and no GPU are needed: the
rotation is a Sylvester-Hadamard matrix scaled to be orthonormal, identical for
every layer.  On this checkpoint the uncalibrated rotations scored *better*
than calibrated ones (40/41 vs 37/41) -- see the README.

Format (what the runtime loads):

    {"layers": {layer_idx: {"rotation": float32 [head_dim, head_dim]}, ...}}

Usage (Qwen3.6-35B-A3B-Escha-W2: 64 layers, head_dim 256):

    python make_hadamard_rotations.py --out-dir /path/to/oscar_rotations

Read head_dim and num_hidden_layers from the model's config.json if your
checkpoint differs.
"""
import argparse
import math
import pathlib

import torch


def hadamard(n: int) -> torch.Tensor:
    """Sylvester construction; n must be a power of two."""
    if n & (n - 1):
        raise ValueError(f"head_dim {n} is not a power of two")
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=64)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rot = (hadamard(args.head_dim) / math.sqrt(args.head_dim)).to(torch.float32)
    err = (rot @ rot.T - torch.eye(args.head_dim)).abs().max().item()
    assert err == 0.0, f"rotation is not orthonormal (max error {err})"

    state = {"layers": {i: {"rotation": rot.clone()} for i in range(args.num_layers)}}

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("k", "v"):
        path = out / f"{name}_rotation_hadamard_hd{args.head_dim}.pt"
        torch.save(state, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
