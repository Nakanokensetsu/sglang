#!/usr/bin/env python3
"""HiCache の mamba バックアップ転送カーネルを本番と同じ形状で単体再現する。

2026-09-20 追加。`--enable-hierarchical-cache` を入れると最初の長文で
`transfer_mamba.cuh:184` が illegal memory access を出す。ホストプール側の
入力(ポインタ・index・ストライド)は全部検査済みで正常だったので、
カーネル単体に切り出してサーバ起動(4分)抜きで叩けるようにする。

  PYTHONPATH=/path/to/sglang/python CUDA_LAUNCH_BLOCKING=1 \
    python3 tools/mamba_transfer_repro.py [host_slots]
"""
import sys

import torch

from sglang.kernels.ops.mamba.transfer_mamba import transfer_kv_mamba_lf_pf
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)

# 本番実測値(hicache_test6.log)
NUM_LAYERS = 48
DEV_SLOTS = 33
STATE = (24, 128, 128)
DTYPE = torch.float16
HOST_SLOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 65


def main():
    dev = "cuda:0"
    item_size = STATE[0] * STATE[1] * STATE[2] * torch.empty(0, dtype=DTYPE).element_size()
    print(f"num_layers={NUM_LAYERS} item_size={item_size}B "
          f"device={NUM_LAYERS}x{DEV_SLOTS}x{STATE} host_slots={HOST_SLOTS}")
    print(f"device arena = {NUM_LAYERS * DEV_SLOTS * item_size / 2**30:.2f} GiB")
    print(f"host  arena = {HOST_SLOTS * NUM_LAYERS * item_size / 2**30:.2f} GiB")

    src = torch.arange(
        NUM_LAYERS * DEV_SLOTS * STATE[0] * STATE[1] * STATE[2],
        dtype=torch.int32, device=dev,
    ).to(DTYPE).view(NUM_LAYERS, DEV_SLOTS, *STATE)
    src_ptrs = torch.tensor(
        [src[i].data_ptr() for i in range(NUM_LAYERS)], dtype=torch.uint64, device=dev
    )

    # 本番と同じ経路で確保する(アロケータ + cudaHostRegister、粒度=1スロット)
    alloc = ALLOC_MEMORY_FUNCS["cuda"]
    dst = alloc(
        (HOST_SLOTS, NUM_LAYERS, 1) + STATE,
        dtype=DTYPE, device="cpu", pin_memory=True,
        allocator=get_allocator_from_storage("default"),
        registration_granularity_bytes=NUM_LAYERS * item_size,
    )
    print(f"host pinned={dst.is_pinned()} bytes={dst.numel()*dst.element_size()} "
          f"registered={hasattr(dst, '_sglang_cuda_host_registered_ranges')}")

    src_idx = torch.tensor([6], dtype=torch.int64, device=dev)
    dst_idx = torch.tensor([0], dtype=torch.int64, device=dev)

    print("launching...", flush=True)
    transfer_kv_mamba_lf_pf(
        src_ptrs=src_ptrs, dst=dst,
        src_indices=src_idx, dst_indices=dst_idx,
        item_size=item_size, dst_layout_dim=item_size * NUM_LAYERS,
        num_layers=NUM_LAYERS,
    )
    torch.cuda.synchronize()
    print("launch ok")

    # 内容一致も見る
    ok = True
    for l in range(NUM_LAYERS):
        if not torch.equal(dst[0, l, 0].to(dev), src[l, 6]):
            print(f"  layer {l}: MISMATCH")
            ok = False
            break
    print("content:", "一致" if ok else "不一致")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
