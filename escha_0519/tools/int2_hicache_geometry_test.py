#!/usr/bin/env python3
"""int2 HiCache ホストプールの幾何(バイト幅)を GPU 無しで検算する。

2026-09-20 追加。`Int2MHATokenToKVPoolHost` は
  ホストの token stride / element_dim / デバイスの1行バイト数
の3つが一致しないと、最初の長文リクエストで CUDA illegal memory access になる。
本番を止めずに算術だけ確かめるため、CPU 上のダミー int2 プールで組み立てる。

  PYTHONPATH=/path/to/sglang/python python3 tools/int2_hicache_geometry_test.py
"""
import sys

import torch

import sglang.srt.mem_cache.pool_host.mha as mha


class FakeInt2Pool:
    """UnifiedInt2HPKVPool ではなく、本番で実際に使われる
    ``MHATokenToKVPool(dtype="int2")`` の持ち物だけを再現したもの。"""

    def __init__(self, layer_num, size, head_num, head_dim, groups, scale_dtype):
        self.dtype = "int2"
        self.store_dtype = torch.uint8
        self.layer_num = layer_num
        self.size = size
        self.page_size = 1
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = head_dim
        self.scale_dtype = scale_dtype
        self.k_num_scale_groups = groups
        self.v_num_scale_groups = groups
        self.device = "cpu"
        self.start_layer = 0
        self.layer_shard_enabled = False
        self.end_layer = layer_num
        slots = size + self.page_size
        self.k_buffer = [
            torch.zeros((slots, head_num, head_dim // 4), dtype=torch.uint8)
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.zeros((slots, head_num, head_dim // 4), dtype=torch.uint8)
            for _ in range(layer_num)
        ]
        self.k_scales_zeros = [
            torch.zeros((slots, head_num, 2 * groups), dtype=scale_dtype)
            for _ in range(layer_num)
        ]
        self.v_scales_zeros = [
            torch.zeros((slots, head_num, 2 * groups), dtype=scale_dtype)
            for _ in range(layer_num)
        ]
        self.k_data_ptrs = torch.tensor(
            [t.data_ptr() for t in self.k_buffer], dtype=torch.uint64
        )
        self.v_data_ptrs = torch.tensor(
            [t.data_ptr() for t in self.v_buffer], dtype=torch.uint64
        )


# 本番(escha_run_0519.sh / Qwen3.8-27B-Escha-W2, TP2)の実値:
#   head_dim=256, num_key_value_heads=4 -> head_num=2/rank
#   --kv-cache-quant-group-size 64 -> groups = 256/64 = 4
#   SGLANG_MIXED_KV_SCALE_DTYPE 未設定 -> float32
# => packed 64B/head, sz 32B/head。K は 1トークン packed 128B + sz 64B。
def build(layer_num=4, size=1024, head_num=2, head_dim=256, groups=4,
          scale_dtype=torch.float32, layout="layer_first"):
    seen = {}

    def fake_jit_probe(*, element_size, **kw):
        # 実際に nvcc を呼ばせない。要求された幅だけ記録する。
        seen["element_size"] = element_size
        return False

    orig_jit = mha.can_use_hicache_jit_kernel
    orig_wb = mha.can_use_write_back_jit_kernel
    mha.can_use_hicache_jit_kernel = fake_jit_probe
    mha.can_use_write_back_jit_kernel = lambda **kw: False
    try:
        dp = FakeInt2Pool(layer_num, size, head_num, head_dim, groups, scale_dtype)
        host = mha.get_mha_host_pool_cls(dp)(
            dp,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=1,
            layout=layout,
            pin_memory=False,
            device="cpu",
        )
        return dp, host, seen
    finally:
        mha.can_use_hicache_jit_kernel = orig_jit
        mha.can_use_write_back_jit_kernel = orig_wb


def main():
    dp, host, seen = build()
    assert type(host) is mha.Int2MHATokenToKVPoolHost, type(host)

    dev_row = dp.k_buffer[0][0].numel() * dp.k_buffer[0].element_size()
    print(f"device k row              : {dev_row} B/token/layer")
    print(f"host token_stride_size    : {host.token_stride_size} B")
    print(f"element_dim * itemsize    : {host.element_dim * host.dtype.itemsize} B")
    probe = seen.get("element_size")
    print(f"JIT probe asked for       : {probe} B"
          + ("  (非CUDA環境なので未呼び出し)" if probe is None else ""))
    print(f"size_per_token (packed+sz): {host.size_per_token} B")
    print(f"host slots                : {host.size}")
    print(f"host sz buffer            : {tuple(host.host_k_sz.shape)} {host.host_k_sz.dtype}")

    assert host.token_stride_size == dev_row
    assert host.element_dim * host.dtype.itemsize == dev_row
    # _is_cuda が False の環境では親が JIT 判定自体を行わない。呼ばれた時だけ検査する。
    if probe is not None:
        assert probe == dev_row, seen
        assert probe % 128 == 0, (
            f"element_size {probe} は JIT カーネルの 128B 境界を満たさない"
        )
    # packed + sz の合計で容量を数えていること(packed だけなら過小申告になる)
    packed = 2 * host.head_num * (head_bytes := dp.head_dim // 4) * dp.layer_num
    sz_itemsize = torch.empty(0, dtype=dp.scale_dtype).element_size()
    sz = 2 * host.head_num * 2 * dp.k_num_scale_groups * sz_itemsize * dp.layer_num
    assert host.size_per_token == packed + sz, (host.size_per_token, packed, sz)
    # 実バッファのバイト数がデバイス1層ぶんの packed と整合すること
    assert host.kv_buffer.numel() == 2 * host.layer_num * host.size * host.head_num * head_bytes

    check_sz_roundtrip(dp, host)
    check_guard_fires()

    print("\nOK: 3つの幅が一致 / sz 往復が一致 / 幅が狂えば起動時に落ちる。")


def check_sz_roundtrip(dp, host):
    """scales/zeros の D2H -> クリア -> H2D 往復がビット一致すること。

    packed 側は CUDA カーネルなので CPU では回せないが、sz 側は素の torch なので
    ここで確かめられる。運び漏れ・層の取り違え・インデックスのずれは
    「前の占有者の scale/zero で解釈されて静かに壊れる」形で出るため、
    起動して気づくのは難しい。
    """
    g = torch.Generator().manual_seed(0)
    for buf in (*dp.k_scales_zeros, *dp.v_scales_zeros):
        buf.copy_(torch.randn(buf.shape, generator=g).to(buf.dtype))
    device_indices = torch.tensor([3, 17, 4, 1000], dtype=torch.int64)
    host_indices = torch.tensor([11, 0, 900, 5], dtype=torch.int64)

    expect_k = [dp.k_scales_zeros[l][device_indices].clone()
                for l in range(dp.layer_num)]
    expect_v = [dp.v_scales_zeros[l][device_indices].clone()
                for l in range(dp.layer_num)]

    # packed 側は CUDA カーネルなので CPU では呼べない。親の実装だけ黙らせて
    # Int2 側の追加分(sz)だけを通す。
    orig_backup = mha.MHATokenToKVPoolHost.backup_from_device_all_layer
    orig_load = mha.MHATokenToKVPoolHost.load_to_device_per_layer
    mha.MHATokenToKVPoolHost.backup_from_device_all_layer = (
        lambda self, *a, **kw: None
    )
    mha.MHATokenToKVPoolHost.load_to_device_per_layer = lambda self, *a, **kw: None
    try:
        host.backup_from_device_all_layer(dp, host_indices, device_indices, "kernel")
        for buf in (*dp.k_scales_zeros, *dp.v_scales_zeros):
            buf.zero_()
        for layer_id in range(dp.layer_num):
            host.load_to_device_per_layer(
                dp, host_indices, device_indices, layer_id, "kernel"
            )
    finally:
        mha.MHATokenToKVPoolHost.backup_from_device_all_layer = orig_backup
        mha.MHATokenToKVPoolHost.load_to_device_per_layer = orig_load

    for l in range(dp.layer_num):
        assert torch.equal(dp.k_scales_zeros[l][device_indices], expect_k[l]), l
        assert torch.equal(dp.v_scales_zeros[l][device_indices], expect_v[l]), l
    # 触っていないスロットが巻き込まれていないこと
    untouched = torch.tensor([2, 50], dtype=torch.int64)
    assert dp.k_scales_zeros[0][untouched].abs().sum() == 0
    print("scales/zeros roundtrip    : 一致 "
          f"({dp.layer_num} 層 x {len(device_indices)} スロット)")


def check_guard_fires():
    """幅が狂ったら起動時に落ちること(直したつもりで直っていない事故の検出)。"""
    orig = mha.Int2MHATokenToKVPoolHost._compute_element_dim
    # 2026-09-20 に踏んだ元のバグ: デバイスの head_dim をそのまま使う
    mha.Int2MHATokenToKVPoolHost._compute_element_dim = (
        lambda self: self.device_pool.head_num * self.device_pool.head_dim
    )
    try:
        build()
    except AssertionError as e:
        assert "element_dim" in str(e), e
        print("guard (element_dim)       : 期待どおり起動時に停止")
    else:
        raise SystemExit("FAIL: element_dim が4倍でも起動してしまった")
    finally:
        mha.Int2MHATokenToKVPoolHost._compute_element_dim = orig

    orig_sz = mha.Int2MHATokenToKVPoolHost.get_size_per_token

    def wrong_head_dim(self):
        # 2026-09-20 に踏んだもう1つのバグ: head_dim に packed+sz を入れる
        out = orig_sz(self)
        self.head_dim = self._packed_bytes_per_head_k + self._sz_bytes_per_head_k
        return out

    mha.Int2MHATokenToKVPoolHost.get_size_per_token = wrong_head_dim
    try:
        build()
    except AssertionError as e:
        assert "token stride" in str(e), e
        print("guard (token stride)      : 期待どおり起動時に停止")
    else:
        raise SystemExit("FAIL: host stride が広くても起動してしまった")
    finally:
        mha.Int2MHATokenToKVPoolHost.get_size_per_token = orig_sz


if __name__ == "__main__":
    sys.exit(main())
