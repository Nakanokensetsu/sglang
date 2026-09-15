#!/usr/bin/env python3
"""Generate the uncalibrated (Hadamard) rotation tensors OSCAR INT2 KV expects.

These are the files pointed at by SGLANG_OSCAR_K_ROTATION_PATH and
SGLANG_OSCAR_V_ROTATION_PATH.  No calibration data and no GPU are needed: the
rotation is a Sylvester-Hadamard matrix scaled to be orthonormal, identical for
every layer.  On this checkpoint the uncalibrated rotations scored *better*
than calibrated ones (40/41 vs 37/41) -- see the README.

Format (what the runtime loads):

    {"layers": {layer_idx: {"rotation": float32 [head_dim, head_dim]}, ...}}

Usage -- point it at the model's config.json and it reads the shape itself:

    python make_hadamard_rotations.py \
        --config /path/to/Qwen3.6-35B-A3B-Escha-W2/config.json \
        --out-dir /path/to/oscar_rotations

Or state the two numbers by hand (head_dim, then num_hidden_layers):

    python make_hadamard_rotations.py --head-dim 256 --num-layers 40 \
        --out-dir /path/to/oscar_rotations

Only full-attention layers ever consume a rotation, and lookups are by layer
index, so generating an entry for every layer is correct and generating too
many is harmless.
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
    ap.add_argument("--config", help="path to the model's config.json")
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--num-layers", type=int)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.config:
        import json

        cfg = json.loads(pathlib.Path(args.config).read_text())
        # Escha-W2 checkpoints keep the language model under "text_config".
        cfg = cfg.get("text_config", cfg)
        args.head_dim = args.head_dim or cfg["head_dim"]
        args.num_layers = args.num_layers or cfg["num_hidden_layers"]
    if not args.head_dim or not args.num_layers:
        ap.error("give --config, or both --head-dim and --num-layers")
    print(f"head_dim={args.head_dim} num_layers={args.num_layers}")

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
