"""Escha 2-/3-bit quantization support for SGLang.

Weights are stored as a packed tail-biting code + rot sign×scale vectors
(rin/rout) + per-channel input/output scales (s_in/s_out). Decode runs on the
escha fused decode-GEMV op for the small-batch fast path, falling back to a
dense-weight reconstruction otherwise.

Per-Linear tensors (``escha_*`` prefix):
    escha_code  int16 (IC//16, OC//16, 16*K)
    escha_rin      fp16 [IC]   (Wscale already folded in — do NOT re-apply)
    escha_rout      fp16 [OC]
    escha_s_in     fp32 [IC]
    escha_s_out    fp32 [OC]
    escha_config   int32 [L, K, V, codebook_id, IC, OC]

Forward (per shard):
    A = (x * s_in).half() ; fused decode-GEMV does had_l(rin) + matmul +
    had_r(rout) ; y = C * s_out (+ bias).

Serving requirements:
  * Set ``TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`` (it must point at the
    ptxas BINARY, not its directory) or Triton's ptxas probe hits a
    PermissionError.
  * Serve dtype ``float16``.
  * INT8-embedding serving is NOT yet handled here (embeddings fall through to
    the default loader, which expects fp16 ``*.weight``). Export with
    ``int8_embedding=False`` for SGLang, or add an embedding quant method.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.parameter import BasevLLMParameter
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = logging.getLogger(__name__)

# the reference codec (MIT) fused code decode+GEMM kernel. Imported lazily-safe so the
# module loads even where the reference codec is absent (apply() then falls back / errors).
# NOTE: the import that would bind `_ref_ext` has never been present in this
# fork, so the NameError below is caught every time and HAS_REF is always False
# — the reference kernel is not used (startup logs `ref_gemm: NO`). Kept as-is
# rather than "fixed": wiring the import would silently switch the GEMM path.
try:
    HAS_REF = hasattr(_ref_ext, "ref_gemm")  # noqa: F821
except Exception:  # pragma: no cover
    _ref_ext = None
    HAS_REF = False

# Clean-room escha-kernels fused decode-GEMV (torch.ops.escha.escham_decode_gemv).
# Beats ref_gemm at batch-1 (~1.08-1.28x, RTX 4090) and is a torch.ops op, so it
# is CUDA-graph / piecewise-capturable (raw-pybind ref_gemm is not). Used as the
# batch-1 decode fast path; ref_gemm handles prefill (batch>1) / fallback.
try:
    import escha  # noqa: F401  (registers torch.ops.escha.*)
    import torch as _torch

    HAS_ESCHAM = hasattr(_torch.ops.escha, "escham_decode_gemv")
except Exception:  # pragma: no cover
    HAS_ESCHAM = False

_ESCHAM_MAX_M = None


def _escham_max_m() -> int:
    """Largest decode batch escham_decode_gemv accepts — PROBED, never hardcoded.

    This gate used to read `batch <= 8`, duplicating the kernel's own limit in
    Python. When the kernel grew to M=16 the hardcode silently kept every bs>8
    decode on ref_gemm, so the kernel change measured as a WASH — the identical
    failure escha_gemv had on 07-16 (cuda_graph_bs listed 16, the gate said 8, and
    bs16 quietly ran a dequant-every-forward fallback at 348 tok/s, slower than
    fp16, until the limit was probed instead). Ask the kernel; never restate it.
    """
    global _ESCHAM_MAX_M
    if _ESCHAM_MAX_M is None:
        try:
            _ESCHAM_MAX_M = int(_torch.ops.escha.escham_decode_gemv_max_m())
        except Exception:  # older kernel build without the probe
            _ESCHAM_MAX_M = 8
    return _ESCHAM_MAX_M


_ESCHAM_MULTI_MAX_M = None


def _escham_multi_max_m() -> int:
    """Largest batch escham_multi_gemv accepts — a SEPARATE probe, deliberately.

    The merged all-shards launch group is a different kernel family
    (escham_multi_gemv_bw / _xh) and was NOT M-tiled when escham_decode_gemv was raised
    to 32 (2026-08-20). Gating the merged path on _escham_max_m() would hand it M in
    17..32 and trip its TORCH_CHECK — inside CUDA-graph capture that is a hard
    failure, not a fallback. Older kernel builds have no probe and no M-tiling, so
    the floor is the historical 16, clamped by the decode cap.
    """
    global _ESCHAM_MULTI_MAX_M
    if _ESCHAM_MULTI_MAX_M is None:
        try:
            _ESCHAM_MULTI_MAX_M = int(_torch.ops.escha.escham_multi_gemv_max_m())
        except Exception:  # older kernel build without the probe
            _ESCHAM_MULTI_MAX_M = min(16, _escham_max_m())
    return _ESCHAM_MULTI_MAX_M


# Benchmark/debug knob: force the ref_gemm path (skip our escham kernel) to A/B the
# two decode kernels in an identical serving setup. Read once at import.
import os as _os

_FORCE_REF = _os.environ.get("ESCHA_FORCE_REF", "0") not in ("0", "", "false", "False")

# Strict escham-only mode: ref_gemm is EVICTED from the hot path. Its DevCtx is a
# process-global singleton (ref_devctx.cu:18) with one raw-cudaMalloc workspace
# (:54) and one spin-lock buffer (:59-69) per device shared by every call; a
# kernel that exits abnormally leaves the locks dirty and the next launch spins
# to Xid 109 CTX SWITCH TIMEOUT (measured 2026-08-16/17, docs/q38_27b_kernel_
# xid_launch_failure_2026-08-17.md). The per-call torch.isfinite sync that
# "accidentally" serialized it is also a per-shard device sync we do not want.
# Strict mode RAISES where the old path would silently fall back — an eval must
# not change arithmetic mid-run. Mutually exclusive with ESCHA_FORCE_REF.
_NO_REF = _os.environ.get("ESCHA_STRICT", "0") not in ("0", "", "false", "False")
if _NO_REF and _FORCE_REF:
    raise RuntimeError("ESCHA_STRICT and ESCHA_FORCE_REF are mutually exclusive")

# Multi-shard decode fast path (r2, 2026-08-12): ONE escham_multi_gemv launch group
# covers every code shard of an apply() call (q+k+v, gate+up, ...) instead of
# a 2-3 kernel sequence per shard — the dense port of the MoE ptr-array batching.
# The A0 profile showed the per-shard launches never reach steady-state bandwidth
# (602 GB/s in-kernel bs1; 244 GB/s on the lovelace M16 path at bs16). Merged
# grids amortize ramp/tail and let the SM-aware split-K shrink its fp32 partials.
# Kill-switch: ESCHA_MULTI=0 reverts to the per-shard loop (escha rule:
# additive + toggle-flagged). Requires a escha build with the op.
HAS_ESCHAM_MULTI = HAS_ESCHAM and hasattr(_torch.ops.escha, "escham_multi_gemv")
_MULTI_ON = _os.environ.get("ESCHA_MULTI", "1") not in ("0", "", "false", "False")
_MULTI_M1 = _os.environ.get("ESCHA_MULTI_M1", "0") not in ("0", "", "false", "False")
# K-grouped merged launches for layers whose shards differ in K (mix247's fused
# gate_up). Default ON; set ESCHA_KGROUP=0 to fall back to the per-shard
# route for an A/B without rebuilding.
_KGROUP = _os.environ.get("ESCHA_KGROUP", "1") not in ("0", "", "false", "False")

# ---------------------------------------------------------------- prefill path
# Large-M (prefill) alternative to ref_gemm: transiently materialise the RAW
# code weight and run the GEMM on cuBLAS. ref_gemm is a decode-shaped kernel —
# it re-decodes the code per output tile and measures ~46-51 TFLOPS effective
# at M=2048, FLAT in M (2026-08-12 head-to-head: our prefill 931-957 tok/s vs
# llama.cpp 1343-1766 on the same card). cuBLAS reaches ~161 TFLOPS fp16/fp32-acc
# on this 4090 (MEASURED; the "82.6 TFLOPS fp32-acc" spec figure is wrong for
# Ada), so paying a one-off decode per chunk and handing the GEMM to cuBLAS wins.
#
# The materialised weight is NEVER cached: all shards at once would be 47.7 GB of
# fp16 on a 24 GB card. It is freed per shard and the caching allocator reuses the
# block, so steady-state churn is one alloc/free of the largest shard.
#
# KEY: we do NOT reconstruct the *deploy* weight. W_deploy = Hr(Hl(w_raw)*rin)*rout
# pays both block-128 transforms on the (IC,OC) WEIGHT — measured 1140 of the
# 1836 ms/step, i.e. the reconstruct would cost more than the GEMM it feeds. The
# transforms are linear and the rot scales diagonal, so the identity
#
#     x @ W_deploy  ==  Hr( Hl(x * rin) @ w_raw ) * rout
#
# moves the whole chain onto the (M,IC)/(M,OC) ACTIVATIONS, which at prefill M are
# 3-9x smaller than the weight. The per-call weight cost collapses to a raw code
# decode (~59 ms/step). This is the SAME decomposition ref_gemm fuses internally,
# so the numerics track the shipped kernel (measured mean rel deviation vs
# ref_gemm output 3.2e-4..5.5e-4, TIGHTER than the dense fallback's 6.3e-4..7.7e-4).
# Whole-step (400 shards) at M=2048: ref 2037 ms -> recon 761 ms = 2.68x.
#
# DEFAULT since 2026-08-12 (opt out with ESCHA_PREFILL=ref). Served 27B W2
# bs1 TTFT, ctx 36864 / MEM 0.80 / CHUNK 2048, paired same-session vs ref:
#   ISL  2048  2141 -> 1097 ms (1.95x)    prefill  962 ->  1878 tok/s
#   ISL  8192  8372 -> 4169 ms (2.01x)    prefill  980 ->  1968 tok/s
#   ISL 32768 34831 ->17651 ms (1.97x)    prefill  941 ->  1857 tok/s
# CHUNK 4096 is a further +2-3% (1944/2008/1883 tok/s) — unlike ref, which was
# SLOWER at 4096. Decode is untouched (70.7-72.4 tok/s both arms) because this
# branch is gated on batch > the escham kernel's probed max M. Quality: MATH-500
# n=500 thinking-off 72.0 vs 72.6 reference (0.3 sigma), none_preds 107 == ref.
# DEFAULT since 2026-08-12 (later that day): "fused" — see the fused-GEMM block
# below. "recon" (the morning's default) and "ref" (the original) remain as
# opt-outs, and "fused" degrades to "recon" automatically if the installed
# escha predates the escham_code_gemm op.
_PREFILL_MODE = _os.environ.get("ESCHA_PREFILL", "fused").strip().lower()
HAS_REF_RECON = (
    HAS_REF and hasattr(_ref_ext, "reconstruct") and hasattr(_ref_ext, "escha_t128")
)

# fp16 ACCUMULATE for the prefill GEMM only (save/restore around the matmul).
# Ada's fp16-acc tensor rate is 2x the fp32-acc rate (measured 260 vs 162 TFLOPS,
# 8192^3). This is a NUMERICS change — unlike the recon path itself, which is
# arithmetically the same operation — so it stays OFF by default and needs its own
# quality gate. Measured deviation vs the fp32-acc recon output: 2.6e-3..3.7e-3
# mean relative, ~8x the fp32-acc path's own deviation from ref_gemm.
_PREFILL_FP16ACC = _os.environ.get("ESCHA_PREFILL_FP16ACC", "0") not in (
    "0",
    "",
    "false",
    "False",
)

# ---------------------------------------------------------------------------
# ESCHA_PREFILL=fused: clean-room FUSED code GEMM (escha-kernels d0afd63).
# Decodes the code straight into mma B-fragments — no fp16 weight round-trip,
# no cuBLAS, and (having no split-K) one owner per output element, so its
# in-kernel WHT+rout+s_out epilogue is structurally outside the c4efc4f race class.
# Whole-step microbench (400 shards, M=2048): ref 1793 ms / recon 734 ms /
# fused fp32-acc 669 ms (1.10x recon, 146 TFLOPS = 90% of this card's measured
# 161.6 fp32-acc roofline) / fused fp16-acc 453 ms (1.62x recon, 215 TFLOPS).
#
# ACCUMULATION is an OP ARGUMENT here, not a process-global cuBLAS mode — which
# is exactly why fp16 accumulate is usable at all on this path: the recon arm had
# to flip torch.backends.cuda.matmul.allow_fp16_accumulation, and that toggle is
# the prime suspect for the sticky launch failure that killed it.
# Measured deviation vs recon-v3 output, mean relative: fp32-acc 6.1e-4..6.8e-4
# (and the fused kernel is ~2.2x TIGHTER than recon/ref vs an fp32 reference —
# it rounds to fp16 twice, they round four times); fp16-acc 3.7e-3..4.1e-3 but
# K-dependent: 6.9e-3 on the IC=17408 down_proj vs 3.7e-3 at IC=5120.
# Hence "mixed": fp16 accumulate only for the shorter-K shards.
# DEFAULT "mixed" — SERVED A/B at ctx 36864 / CHUNK 4096 / MEM 0.72, TTFT vs the
# recon path: ISL 2048 1028 -> 799 ms, 8192 3919 -> 3053 ms, 32768 16752 ->
# 13258 ms (1.26-1.29x); fp32-acc alone is 1.06x. Prefill 2094 -> 2687 tok/s at
# ISL 8192 = 1.52x llama.cpp/Unsloth-Q2_K_XL's best config (1766) and 2.74x the
# ref baseline this campaign started from. Decode is untouched (large-M branch
# only). Peak VRAM 19380 vs 19792 MiB — the fused path materialises no fp16
# weight at all. QUALITY GATES, MATH-500 n=500 think-off batch16 vs the 72.6
# reference (sigma~2.0, none_preds 107): fp32-acc **72.2 / 106 PASS**,
# mixed-acc **72.0 / 107 PASS** (none_preds identical to the reference).
_PREFILL_ACC = _os.environ.get("ESCHA_PREFILL_ACC", "mixed").strip().lower()
_ACC_MIXED_IC_MAX = int(_os.environ.get("ESCHA_ACC_IC_MAX", "6144"))
HAS_ESCHAM_GEMM = HAS_ESCHAM and hasattr(_torch.ops.escha, "escham_code_gemm")


def _acc_mode_for(IC: int) -> int:
    """0 = fp32 mma accumulate, 1 = fp16. Policy is per-shard because the fp16
    error grows with the reduction length K=IC (6.9e-3 at IC=17408 vs 3.7e-3 at
    IC=5120), and the mode is a per-call argument, so mixing costs nothing."""
    if _PREFILL_ACC == "fp16":
        return 1
    if _PREFILL_ACC == "mixed":
        return 1 if IC <= _ACC_MIXED_IC_MAX else 0
    return 0


def _prefill_recon(x_2d, code, rin, rout, s_in, s_out, IC, OC, K, cbA, mul1, x_dtype):
    """Large-M path: raw code decode + activation-side rot + cuBLAS GEMM.

    No NaN/isfinite host sync here, unlike the ref_gemm branch. That guard is
    load-bearing for ref_gemm because that kernel autotunes (coop_autotune /
    ref_devctx) and the sync accidentally serializes its raw-pybind launches —
    removing it hit cudaErrorLaunchFailure under mixed prefill+decode traffic
    (2026-08-12). Neither kernel used here autotunes: ``reconstruct`` and
    ``escha_t128`` are fixed-shape launches that take ``at::cuda::getCurrentCUDAStream()``
    (verified in reconstruct.cu:108 / transform.cu:98), and the GEMM is cuBLAS on
    the same stream. So there is nothing to serialize and no per-call sync — which
    also means this path adds no host sync to the prefill inner loop at all.
    """
    dev = x_2d.device
    w = torch.empty((IC, OC), dtype=torch.half, device=dev)
    _ref_ext.reconstruct(w, code, K, cbA, mul1)

    A = (x_2d * s_in).to(torch.half).contiguous()
    A_had = torch.empty_like(A)
    _ref_ext.escha_t128(A, A_had, rin, None, 1.0)  # pre-scale by rin, then had_128
    del A

    C = A_had @ w  # fp16-acc, if enabled, is toggled once per apply() by the caller
    del w, A_had  # free before the epilogue so the allocator can reuse the block

    out = torch.empty_like(C)
    _ref_ext.escha_t128(C, out, None, rout, 1.0)  # had_128, then post-scale by rout
    return out.to(x_dtype) * s_out


def _a_half_cached(layer, i: int, dtype):
    """Per-shard s_in/s_out cast to `dtype`, computed ONCE and cached on the layer."""
    cache = getattr(layer, "_escha_a_half", None)
    if cache is None:
        cache = layer._escha_a_half = {}
    hit = cache.get(i)
    if hit is None or hit[0].dtype != dtype:
        hit = (
            layer.escha_shard_s_in[i].to(dtype),
            layer.escha_shard_s_out[i].to(dtype),
        )
        cache[i] = hit
    return hit


def _escham_covered(cfgs) -> bool:
    """True iff every real shard satisfies the escham_decode_gemv / escham_code_gemm
    kernel gate (K in (2, 3), IC and OC both multiples of 128) — i.e. no shard
    would ever need to fall through to ref_gemm or the dense fallback. `cfgs` is
    `layer.escha_shard_configs`: a list of `[L, K, V, codebook_id, IC, OC]` (or
    `None` for an absent shard slot). Module-level (not inlined) so ESCHA_STRICT
    coverage can be unit-tested directly against a fixture config list.

    K=3 joined the gate with the mixed-bit work: the dense kernels are templated
    on K (escham_gemv_bw_kernel / escham_code_gemm_kernel / escham_multi_gemv_*), which
    is what lets mix247/mix270 serve on the strict profile. Correctness gate:
    escha-kernels/tests/test_dense_k3_parity.py — K=2 stays BIT-IDENTICAL to the
    pre-change build, K=3 is checked against escham_reconstruct.
    """
    return all(
        c is None or (c[1] in (2, 3) and c[4] % 128 == 0 and c[5] % 128 == 0)
        for c in cfgs
    )


# ============================================================================
# Parameter class (per-shard variable-shape storage; mirrors the upstream
# per-shard variable-shape quant tensor param pattern)
# ============================================================================


class EschaTensorParam(BasevLLMParameter):
    """Per-shard parameter for escha tensors (code, rin, rout, s_in, s_out, config).

    Different shards (e.g. q/k/v of a merged QKV) have different OC, so each
    shard's tensor is stored in a Python list rather than asserting one shape.
    """

    def __init__(
        self,
        num_shards: int,
        suffix: str = "",
        is_row_parallel: bool = False,
        owner_layer=None,
        **kwargs,
    ):
        self.qkv_idxs = {"q": 0, "k": 1, "v": 2}
        self._num_shards = num_shards
        self._shards: list = [None] * num_shards
        self._suffix = suffix
        self._is_row_parallel = is_row_parallel
        self._owner_layer = owner_layer
        super().__init__(**kwargs)
        self._weight_loader = self._escha_weight_loader

    # -- tensor-parallel sharding ---------------------------------------------------
    # `_escha_weight_loader` is the ONLY loader hook the model's weight_loader loop
    # reaches: BasevLLMParameter.weight_loader returns self._weight_loader, which
    # __init__ pins here, so the layer's own weight_loader -- the only place sglang
    # does TP slicing in the default flow -- never runs. Without the code below every
    # rank keeps the full checkpoint tensor and rank 0 dies with "weight must have
    # shape (dim, width)". Reported by @ginerJuanUdesa (escha-tp-fix-qwen3dense).
    def _tp(self):
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank as _r
            from sglang.srt.distributed import (
                get_tensor_model_parallel_world_size as _s,
            )

            return _r(), _s()
        except Exception:  # tests / non-distributed
            return 0, 1

    def _slice(self, t, axis: int, r: int, s: int):
        tot = t.shape[axis]
        assert (
            tot % s == 0
        ), f"escha suffix {self._suffix} dim {axis}={tot} not divisible by tp {s}"
        per = tot // s
        # escha_code is (IC//16, OC//16, 16*K), so 128 IC/OC elements = 8 rows/cols
        # here. The escham kernels gate on IC % 128 == 0 and OC % 128 == 0 (_escham_covered);
        # a per-rank slice that breaks that still RUNS, it just drops silently to
        # ref_gemm / the dense fallback -- correct but far slower, with nothing in the
        # log to say why. Fail loudly instead.
        if self._suffix == "escha_code" and axis in (0, 1):
            assert per % 8 == 0, (
                f"escha_code {'IC' if axis == 0 else 'OC'} per rank = {per * 16} is not "
                f"a multiple of 128 at tp={s}; the escham kernel gate would reject this shard "
                f"and it would fall back to a much slower path"
            )
        return t.narrow(axis, r * per, per).contiguous()

    def _shard_tp(self, t):
        """Slice one already-per-shard escha tensor down to this TP rank."""
        if t is None:
            return t
        r, s = self._tp()
        if s == 1:
            return t
        # config is [L, K, V, cb_id, IC, OC] int32 -- rewrite the sharded dim.
        if self._suffix == "escha_config":
            new = t.clone()
            idx = 4 if self._is_row_parallel else 5
            assert (
                int(new[idx]) % s == 0
            ), f"escha_config dim {idx}={int(new[idx])} not divisible by tp {s}"
            new[idx] = int(new[idx]) // s
            return new
        if self._is_row_parallel:
            # IC-sharded: code axis 0, rin/s_in axis 0; rout/s_out replicated (an
            # output-channel scale is linear, so scaling each rank's partial and
            # all-reducing equals scaling the reduced sum).
            if self._suffix in ("escha_code", "escha_rin", "escha_s_in"):
                return self._slice(t, 0 if self._suffix == "escha_code" else 0, r, s)
            return t
        # column / merged / qkv: OC-sharded.
        if self._suffix == "escha_code":
            return self._slice(t, 1, r, s)
        if self._suffix in ("escha_rout", "escha_s_out"):
            return self._slice(t, 0, r, s)
        return t  # rin / s_in: IC not sharded on column-parallel

    def _split_fused_into_shards(self, t, shard_ids=None):
        """Fused-on-disk merged column linear: ONE checkpoint tensor for a layer that
        declares num_shards>1 (GDN in_proj_qkv over [key, key, value]). Split by the
        pre-TP sub-sizes, TP-slice each sub on OC, and fill _shards[i] separately so
        the runtime sees a normal per-sub-shard MergedColumnParallelLinear."""
        if t is None:
            return
        r, s = self._tp()
        sub_all = getattr(
            self._owner_layer, "_escha_pretp_output_partition_sizes", None
        )
        assert sub_all, "fused merged linear: no pretp output partition sizes"
        # 2026-09-18 自前移植: sglang 0.5.19 は分離チェックポイントを融合レイヤへ読む際、
        # どのシャードに入れるかを loaded_shard_id で指定する
        #   ("in_proj_qkvz.", "in_proj_qkv.", (0,1,2))  ← q/k/v をシャード0,1,2へ
        #   ("in_proj_qkvz.", "in_proj_z.",   3)        ← gate をシャード3へ
        # よって「全シャード」ではなく**指定されたシャードだけ**を対象にする。
        if shard_ids is None:
            shard_ids = tuple(range(self._num_shards))
            assert (
                len(sub_all) == self._num_shards
            ), f"fused merged linear: expected {self._num_shards} sub sizes, got {sub_all}"
        sub = [sub_all[i] for i in shard_ids]
        tgt = list(shard_ids)
        if self._suffix == "escha_config":
            assert int(t[5]) == sum(
                sub
            ), f"fused escha_config OC {int(t[5])} != sum(sub sizes) {sum(sub)}"
            for i, ss in enumerate(sub):
                new = t.clone()
                assert ss % s == 0, f"sub size {ss} not divisible by tp {s}"
                new[5] = ss // s
                self._shards[tgt[i]] = new
            return
        if self._suffix == "escha_code":
            # 2026-09-18 デバッグ: 0.5.19 で sub と実テンソルが食い違う件の調査
            import os as _dbg_os

            if _dbg_os.environ.get("ESCHA_DEBUG_SPLIT") == "1":
                print(
                    f"[ESCHA_SPLIT] layer={type(self._owner_layer).__name__} "
                    f"suffix={self._suffix} sub={sub} num_shards={self._num_shards} "
                    f"t.shape={tuple(t.shape)} expect_sum={sum(ss//16 for ss in sub)}",
                    flush=True,
                )
            for i, p in enumerate(torch.split(t, [ss // 16 for ss in sub], dim=1)):
                self._shards[tgt[i]] = (
                    self._slice(p, 1, r, s) if s > 1 else p.contiguous()
                )
            return
        if self._suffix in ("escha_rout", "escha_s_out"):
            for i, p in enumerate(torch.split(t, list(sub), dim=0)):
                self._shards[tgt[i]] = (
                    self._slice(p, 0, r, s) if s > 1 else p.contiguous()
                )
            return
        for i in range(self._num_shards):  # rin / s_in: IC-dim, replicated
            self._shards[i] = t

    def _escha_weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        # TP>1 ONLY. At world size 1 every line here must reduce to the stock
        # `_shards[idx] = loaded_weight`. The fused-split is NOT a general improvement:
        # it turns in_proj_qkv (48 GDN layers; ONE fused tensor on disk but
        # num_shards=3) from the single fused shard the runtime ships with into a
        # 3-shard layout, changing the launch grouping and the CUDA-graph shard
        # signature. Gate on world size so single-GPU runs are bit-for-bit untouched.
        # 2026-09-18 自前移植: sglang 0.5.19 は融合チェックポイントを読むとき
        # loaded_shard_id にタプル (0,1,2) を渡す(0.5.15 には無い呼び方)。これは
        # 「1つの融合テンソルを全シャードへ分割せよ」の意味なので None と同じ扱いにする。
        if isinstance(loaded_shard_id, tuple):
            # 指定された複数シャードへ分割して入れる(0.5.19 の分離チェックポイント対応)
            param._split_fused_into_shards(loaded_weight, shard_ids=loaded_shard_id)
            return
        if param._tp()[1] > 1:
            if (
                loaded_shard_id is None
                and param._num_shards > 1
                and not param._is_row_parallel
                and _os.environ.get("ESCHA_TP_NAIVE") != "1"
            ):
                param._split_fused_into_shards(loaded_weight)
                return
        idx = 0 if loaded_shard_id is None else param._shard_id_as_int(loaded_shard_id)
        param._shards[idx] = param._shard_tp(loaded_weight)

    def _shard_id_as_int(self, shard_id) -> int:
        # 2026-09-18 自前移植: sglang 0.5.19 は融合チェックポイントを読むとき
        # loaded_shard_id にタプル (0,1,2) を渡してくる(0.5.15 には無い呼び方)。
        # escha 側は str("q"/"k"/"v") か int しか想定しておらず KeyError になる。
        # タプルは「全シャードまとめて」の意味なので先頭を返す。
        if isinstance(shard_id, int):
            return shard_id
        return self.qkv_idxs[shard_id]

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._shards[0] = self._shard_tp(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._shards[0] = self._shard_tp(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        idx = self._shard_id_as_int(shard_id) if shard_id is not None else 0
        self._shards[idx] = self._shard_tp(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_id = kwargs.get("shard_id")
        idx = self._shard_id_as_int(shard_id) if shard_id is not None else 0
        self._shards[idx] = self._shard_tp(loaded_weight)


# ============================================================================
# Config
# ============================================================================


class DIEschaConfig(QuantizationConfig):
    """Config for DI ESCHA (ESCHAM code) quantized models."""

    def __init__(self, codebook: str, bits: float, full_config: Dict[str, Any]) -> None:
        super().__init__()
        self.codebook = codebook
        self.bits = bits
        self.full_config = full_config
        self.lm_head_quantized = False

    def __repr__(self) -> str:
        return f"DIEschaConfig(codebook={self.codebook!r}, bits={self.bits})"

    @classmethod
    def get_name(cls) -> str:
        return "escha"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> DIEschaConfig:
        global_cfg = config.get("global_config", {})
        codebook = global_cfg.get("codebook", config.get("codebook", "cbA"))
        bits = global_cfg.get("bits", config.get("bits", 2.0))
        return cls(codebook=codebook, bits=bits, full_config=config)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            # Honor the export's ignore list (unquantized linears stay fp16).
            ignore = set(self.full_config.get("ignore", []))
            # 2026-09-18 自前移植: チェックポイントに escha_* を持たない層は非量子化。
            # Escha-W2 の GDN では in_proj_a / in_proj_b が素の fp16 のまま焼かれている
            # (quantize_config.json に ignore リストが無いのでコード側で判定する)。
            # 0.5.15 のフォークはこれらを独立レイヤにして回避していたが、0.5.19 は
            # in_proj_ba という融合レイヤを作るため、ここで弾かないとシャードが空になる。
            ignore |= {"in_proj_a", "in_proj_b", "in_proj_ba"}
            leaf = prefix.rsplit(".", 1)[-1] if prefix else ""
            if leaf in ignore or prefix in ignore:
                from sglang.srt.layers.quantization.unquant import (
                    UnquantizedLinearMethod,
                )

                return UnquantizedLinearMethod()
            return DIEschaLinearMethod(self)
        # Embeddings / lm_head: DEFER to the model's own load_weights int8 path.
        # qwen3.py (and siblings) buffer `weight_int8`+`weight_scale` and dequantize
        # into the standard `.weight`. Returning a custom embedding quant method here
        # CONFLICTS with that: the model code consumes the weight_int8/weight_scale
        # keys, so params registered here never get filled (they stay zero -> garbage
        # logits). Returning None gives the embedding/lm_head a standard `.weight`,
        # which the model's load_weights then fills via its int8 dequant. (Validated
        # 2026-06-15: int8-emb 8B-W3 served garbage with the custom method, correct
        # with this deferral.)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


# ============================================================================
# INT8 embedding / lm_head method
# ============================================================================


class DIEschaInt8EmbeddingMethod:
    """Loads the escha standard int8-quantized embed_tokens / lm_head.

    The export writes per-row (per-vocab) int8 weights with a fp16 scale:
      ``{p}.weight_int8``   int8  [vocab, hidden]
      ``{p}.weight_scale``  fp16  [vocab]
    We register those two params, then dequantize to a plain fp16 ``weight`` in
    ``process_weights_after_loading`` and delegate embedding/apply to the
    standard path. Dequant-at-load keeps the int8 ROUNDING (the real deploy
    accuracy) while running the forward in fp16 like the rest of the escha
    serve. Required: the escha GEMV kernel forces fp16, and embeddings must
    match — the fp16 ``weight`` here does.
    """

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        import torch as _torch

        from sglang.srt.utils.common import set_weight_attrs

        num = sum(output_partition_sizes)
        w_int8 = _torch.nn.Parameter(
            _torch.empty(num, input_size_per_partition, dtype=_torch.int8),
            requires_grad=False,
        )
        set_weight_attrs(w_int8, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight_int8", w_int8)
        set_weight_attrs(w_int8, extra_weight_attrs)

        w_scale = _torch.nn.Parameter(
            _torch.empty(num, dtype=_torch.float16), requires_grad=False
        )
        set_weight_attrs(w_scale, {"output_dim": 0})
        layer.register_parameter("weight_scale", w_scale)
        set_weight_attrs(w_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer) -> None:
        import torch as _torch

        w = layer.weight_int8.data.to(_torch.float16) * layer.weight_scale.data.to(
            _torch.float16
        ).unsqueeze(1)
        del layer._parameters["weight_int8"]
        del layer._parameters["weight_scale"]
        layer.register_parameter("weight", _torch.nn.Parameter(w, requires_grad=False))

    def apply(self, layer, x, bias=None):
        import torch.nn.functional as _F

        return _F.linear(x, layer.weight, bias)

    def embedding(self, layer, input_):
        import torch.nn.functional as _F

        return _F.embedding(input_, layer.weight)


# ============================================================================
# Linear method
# ============================================================================

_ESCHA_SUFFIXES = [
    "escha_code",
    "escha_rin",
    "escha_rout",
    "escha_s_in",
    "escha_s_out",
    "escha_config",
]


class DIEschaLinearMethod(LinearMethodBase):
    """Linear method for escha_official code-coded weights."""

    def __init__(self, quant_config: DIEschaConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        num_shards = len(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.escha_num_shards = num_shards
        # Stamp the layer's TP role for the weight loader. Row-parallel (o_proj /
        # down_proj) is the one that shards the INPUT, so input_size_per_partition
        # shrinks; column / merged / qkv shard outputs, and output_partition_sizes
        # arrives already per-rank -- multiply back to recover the pre-TP sub-sizes a
        # fused-on-disk merged linear needs to be split by.
        try:
            from sglang.srt.distributed import (
                get_tensor_model_parallel_world_size as _tps,
            )

            tp_size = _tps()
        except Exception:
            tp_size = 1
        is_row_parallel = input_size_per_partition < input_size
        layer._escha_is_row_parallel = is_row_parallel
        layer._escha_pretp_output_partition_sizes = tuple(
            int(x) * tp_size for x in output_partition_sizes
        )
        for suffix in _ESCHA_SUFFIXES:
            param = EschaTensorParam(
                data=torch.empty(0, dtype=torch.int32),
                num_shards=num_shards,
                suffix=suffix,
                is_row_parallel=is_row_parallel,
                owner_layer=layer,
                weight_loader=weight_loader,
            )
            layer.register_parameter(suffix, param)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The rank's device, NOT a hardcoded cuda:0. sglang calls set_device(gpu_id)
        # in init_torch_distributed() before load_model(), so current_device() is
        # already the device this server/rank owns by the time we get here.
        # With the hardcode, anything that lands on a non-zero device -- tp_size>1,
        # or a single-GPU run started with --base-gpu-id N and no CUDA_VISIBLE_DEVICES
        # -- put every 2-bit buffer on cuda:0 while the input tensor sat on cuda:N,
        # so the kernel launched cross-device: illegal memory access on the FIRST
        # forward, with a traceback that points at the kernel rather than at this line.
        # Reported by @ginerJuanUdesa (escha-tp-fix-qwen3dense), 2026-08-21.
        # No-op wherever cuda:0 was already right (verified: current_device()==0 on
        # every configuration serve.sh ships).
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        cfg_param = layer.escha_config
        n_shards = len(cfg_param._shards)

        def _pad(seq):
            return list(seq) + [None] * (n_shards - len(seq))

        code = _pad(layer.escha_code._shards)
        rin = _pad(layer.escha_rin._shards)
        rout = _pad(layer.escha_rout._shards)
        s_in = _pad(layer.escha_s_in._shards)
        s_out = _pad(layer.escha_s_out._shards)
        cfgs = _pad(cfg_param._shards)

        layer.escha_shard_code = [
            (t.to(device).contiguous() if t is not None else None) for t in code
        ]
        layer.escha_shard_rin = [
            (s.to(device, dtype=torch.float16) if s is not None else None) for s in rin
        ]
        layer.escha_shard_rout = [
            (s.to(device, dtype=torch.float16) if s is not None else None) for s in rout
        ]
        layer.escha_shard_s_in = [
            (s.to(device, dtype=torch.float32).reshape(-1) if s is not None else None)
            for s in s_in
        ]
        layer.escha_shard_s_out = [
            (s.to(device, dtype=torch.float32).reshape(-1) if s is not None else None)
            for s in s_out
        ]
        layer.escha_shard_configs = [
            (c.tolist() if c is not None else None) for c in cfgs
        ]
        # Lazy dense-weight fallback cache (only filled if ref_gemm errors).
        layer.escha_shard_dense_fallback = [None] * n_shards

        for suffix in _ESCHA_SUFFIXES:
            if hasattr(layer, suffix):
                delattr(layer, suffix)

        # Multi-shard decode metadata. Built HERE, after the final .contiguous()
        # copies above, because the int64 ptr array snapshots data_ptr()s — built
        # any earlier it would dangle when the copies re-allocate (the 2026-07-08
        # PackedQwen35MoeExperts stale-data_ptr silent-no-op gotcha). The shard
        # tensors stay referenced by escha_shard_* for the layer's lifetime.
        layer.escha_multi = None
        if HAS_ESCHAM_MULTI and _MULTI_ON:
            cfgs_real = [c for c in layer.escha_shard_configs if c is not None]
            shards_real = [t for t in layer.escha_shard_code if t is not None]

            def _build_multi(idx):
                """Merged-launch metadata for shard indices `idx` (uniform K)."""
                blk_shard, blk_n0, tiles_n = [], [], []
                for j, i in enumerate(idx):
                    oc = layer.escha_shard_configs[i][5]
                    nb = oc // 128
                    blk_shard += [j] * nb
                    blk_n0 += list(range(nb))
                    tiles_n.append(oc // 16)
                cb = layer.escha_shard_configs[idx[0]][3]
                return (
                    torch.tensor(
                        [layer.escha_shard_code[i].data_ptr() for i in idx],
                        dtype=torch.int64,
                        device=device,
                    ),
                    torch.stack([layer.escha_shard_rin[i] for i in idx]).contiguous(),
                    torch.stack([layer.escha_shard_s_in[i] for i in idx]).contiguous(),
                    torch.cat([layer.escha_shard_rout[i] for i in idx]).contiguous(),
                    torch.cat([layer.escha_shard_s_out[i] for i in idx]).contiguous(),
                    torch.tensor(blk_shard, dtype=torch.int32, device=device),
                    torch.tensor(blk_n0, dtype=torch.int32, device=device),
                    torch.tensor(tiles_n, dtype=torch.int32, device=device),
                    sum(layer.escha_shard_configs[i][5] for i in idx),
                    bool(cb == 1),
                    bool(cb == 2),
                )

            # K-GROUPED merged launches, for layers whose shards do NOT share a K.
            # mix247 fuses gate_proj(K=2) with up_proj(K=3) into one
            # MergedColumnParallelLinear on 64 of 65 layers. Such a group used to
            # fall to the per-shard route entirely, which above M=8 lands on the
            # lovelace kernels and collapses: MEASURED on the gate_up pair,
            # per-shard vs one merged launch PER K -- 1.00x at M=4, 1.17x at M=8,
            # 1.73x at M=12, 1.87x at M=16 (394 -> 738 GB/s). gate_up is 60% of the
            # M=16 decode step, so this is ~1.39x on the whole step.
            # One launch per K, results sliced back into original shard order.
            layer.escha_multi_kgroups = None
            _base_ok = (
                cfgs_real
                and len(shards_real) == len(cfgs_real)
                and len({c[3] for c in cfgs_real}) == 1
                and len({c[4] for c in cfgs_real}) == 1
                and cfgs_real[0][4] % 128 == 0
                and all(c[5] % 128 == 0 for c in cfgs_real)
                and all(c[1] in (2, 3) for c in cfgs_real)
            )
            if _base_ok and len({c[1] for c in cfgs_real}) > 1:
                by_k: Dict[int, list] = {}
                for i, c in enumerate(layer.escha_shard_configs):
                    if c is not None:
                        by_k.setdefault(int(c[1]), []).append(i)
                groups = []
                for kk in sorted(by_k):
                    gidx = by_k[kk]
                    groups.append(
                        (
                            _build_multi(gidx),
                            kk,
                            tuple(gidx),
                            tuple(layer.escha_shard_configs[i][5] for i in gidx),
                        )
                    )
                layer.escha_multi_kgroups = groups
                logger.debug(
                    "escha: K-grouped multi for a mixed-K layer: %s",
                    {k: len(v) for k, v in by_k.items()},
                )

            if (
                cfgs_real
                and len(shards_real) == len(cfgs_real)
                # Uniform K per launch group (2 or 3): the merged kernel takes
                # ONE K template arg, so shards of differing K cannot share a
                # launch. Mixed-bit models (mix247/mix270) put K=3 on FFN
                # tensors and K=2 on attention/SSM, and K is per-tensor, so a
                # single layer's shards are uniform in practice; a layer that
                # ever mixed would simply fall to the per-shard path.
                and len({c[1] for c in cfgs_real}) == 1
                and cfgs_real[0][1] in (2, 3)
                and len({c[3] for c in cfgs_real}) == 1  # one codebook
                and len({c[4] for c in cfgs_real}) == 1  # shared IC
                and cfgs_real[0][4] % 128 == 0
                and all(c[5] % 128 == 0 for c in cfgs_real)
            ):
                layer.escha_multi_k = int(cfgs_real[0][1])
                idx = [
                    i for i, c in enumerate(layer.escha_shard_configs) if c is not None
                ]
                blk_shard, blk_n0, tiles_n = [], [], []
                for j, i in enumerate(idx):
                    oc = layer.escha_shard_configs[i][5]
                    nb = oc // 128
                    blk_shard += [j] * nb
                    blk_n0 += list(range(nb))
                    tiles_n.append(oc // 16)
                cb_id = cfgs_real[0][3]
                layer.escha_multi = (
                    torch.tensor(
                        [layer.escha_shard_code[i].data_ptr() for i in idx],
                        dtype=torch.int64,
                        device=device,
                    ),
                    torch.stack([layer.escha_shard_rin[i] for i in idx]).contiguous(),
                    torch.stack([layer.escha_shard_s_in[i] for i in idx]).contiguous(),
                    torch.cat([layer.escha_shard_rout[i] for i in idx]).contiguous(),
                    torch.cat([layer.escha_shard_s_out[i] for i in idx]).contiguous(),
                    torch.tensor(blk_shard, dtype=torch.int32, device=device),
                    torch.tensor(blk_n0, dtype=torch.int32, device=device),
                    torch.tensor(tiles_n, dtype=torch.int32, device=device),
                    sum(c[5] for c in cfgs_real),
                    bool(cb_id == 1),
                    bool(cb_id == 2),
                )

        n_real = sum(t is not None for t in layer.escha_shard_code)
        # Strict-mode coverage verdict: ref_free means every real shard clears the
        # escham kernel gate AND both escham ops are actually loaded, i.e. this layer
        # never needs ref_gemm or the dense fallback at any batch size. Checked at
        # LOAD time (not first forward) so ESCHA_STRICT fails fast, before serving.
        covered = _escham_covered(layer.escha_shard_configs)
        ref_free = covered and HAS_ESCHAM and HAS_ESCHAM_GEMM
        if _NO_REF and not ref_free:
            raise RuntimeError(
                "ESCHA_STRICT set but this layer cannot run escham-only: "
                f"covered={covered} HAS_ESCHAM={HAS_ESCHAM} HAS_ESCHAM_GEMM={HAS_ESCHAM_GEMM}"
            )
        logger.info(
            "Loaded %d escha shards (ref_gemm: %s, multi: %s, prefill: %s%s, ref-free: %s)",
            n_real,
            "YES" if HAS_REF else "NO",
            "YES" if layer.escha_multi is not None else "no",
            (
                _PREFILL_MODE
                if (_PREFILL_MODE != "recon" or HAS_REF_RECON)
                else "recon-UNAVAILABLE->ref"
            ),
            " +fp16acc" if (_PREFILL_MODE == "recon" and _PREFILL_FP16ACC) else "",
            "YES" if ref_free else "no",
        )

    # ------------------------------------------------------------------ apply

    def _dense_fallback(self, layer, i, cfg):
        """Decode shard i's deploy weight (IC, OC) fp16 — cached. Used when the
        fast decode kernel is unavailable or returns non-finite output."""
        if layer.escha_shard_dense_fallback[i] is not None:
            return layer.escha_shard_dense_fallback[i]
        from escha.linear import reconstruct_deploy_weight

        L, K, V, cb_id, IC, OC = cfg
        w = reconstruct_deploy_weight(
            layer.escha_shard_code[i],
            layer.escha_shard_rin[i],
            layer.escha_shard_rout[i],
            IC,
            OC,
            K,
            cb_id == 1,
            cb_id == 2,
        )  # (IC, OC) fp16
        layer.escha_shard_dense_fallback[i] = w.contiguous()
        return layer.escha_shard_dense_fallback[i]

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        batch = x_2d.shape[0]
        x_dtype = x.dtype

        # Multi-shard decode fast path: one launch group for the whole apply().
        # Same M-cap as escham_decode_gemv (probed); fixed-shape, no host sync ->
        # CUDA-graph capturable. Falls through to the per-shard loop on any error.
        # batch >= 2 only: at M=1 the per-shard fused bw kernel measures faster
        # (632 vs 604 GB/s effective, standalone replay) — the merge pays off when
        # the M16 body replaces the latency-bound lovelace path at M 2..16.
        # K-GROUPED merged path: one escham_multi_gemv per K, pieces reassembled in
        # original shard order. Only mixed-K layers have this (mix247's fused
        # gate_up); uniform layers take the single-launch path below unchanged.
        kg = getattr(layer, "escha_multi_kgroups", None)
        _m_lo_kg = 1 if _MULTI_M1 else 2
        if (
            kg is not None
            and _KGROUP
            and not _FORCE_REF
            and _m_lo_kg <= batch <= _escham_multi_max_m()
        ):
            try:
                xin = x_2d.to(torch.half).contiguous()
                pieces = {}
                for meta, kk, gidx, ocs in kg:
                    y = torch.ops.escha.escham_multi_gemv(
                        xin, *meta[:9], kk, meta[9], meta[10], 0
                    )
                    off = 0
                    for i, oc in zip(gidx, ocs):
                        pieces[i] = y[:, off : off + oc]
                        off += oc
                out = torch.cat([pieces[i] for i in sorted(pieces)], dim=-1).to(x_dtype)
                out = out.reshape(orig_shape[:-1] + (out.shape[-1],))
                if bias is not None:
                    out = out + bias
                return out
            except Exception as exc:  # pragma: no cover
                if _NO_REF:
                    raise
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "escha: K-grouped multi failed DURING CUDA-graph capture; "
                        "refusing to record a different route than replay would take"
                    ) from exc

        qm = getattr(layer, "escha_multi", None)
        # ESCHA_MULTI_M1=1: extend the merged multi-shard path down to M=1
        # (default keeps the measured-standalone-faster per-shard bw kernel).
        # bs1 serving trace 2026-08-18: 400 per-shard launches/step = 72% of the
        # 14.0 ms step at 605 GB/s effective; the merge trades per-kernel ramp
        # for one launch group. GATED as a bs1-optimization experiment: accept
        # only if bitwise-identical to the per-shard baseline (split-K grouping
        # must match) AND faster end-to-end.
        _m_lo = 1 if _MULTI_M1 else 2
        if (
            qm is not None
            and not _FORCE_REF
            and _m_lo <= batch <= _escham_multi_max_m()
        ):
            try:
                out = torch.ops.escha.escham_multi_gemv(
                    x_2d.to(torch.half).contiguous(),
                    *qm[:9],
                    getattr(layer, "escha_multi_k", 2),
                    qm[9],
                    qm[10],
                    0,
                ).to(x_dtype)
                out = out.reshape(orig_shape[:-1] + (out.shape[-1],))
                if bias is not None:
                    out = out + bias
                return out
            except Exception as exc:  # pragma: no cover — fall through to per-shard
                if _NO_REF:
                    # Strict mode must never mask a real kernel error as a silent
                    # route change to a slower/different-numerics path.
                    raise
                if torch.cuda.is_current_stream_capturing():
                    # The per-shard loop below is itself capturable, but a
                    # capture that silently records a different route than
                    # replay would take is exactly the 35B stale-ptr class of
                    # bug — fail loudly instead of falling through, chaining the
                    # original exception rather than masking it.
                    raise RuntimeError(
                        "escha: escham_multi_gemv failed DURING CUDA-graph capture "
                        "(batch=%d) — refusing the silent per-shard fallback; a "
                        "capture that records a different route than replay is "
                        "the stale-ptr class of bug. Cap --cuda-graph-bs at "
                        "torch.ops.escha.escham_decode_gemv_max_m()." % batch
                    ) from exc
                pass

        # Large-M path selection, hoisted out of the shard loop.
        _large_m = batch > _escham_max_m()
        # Under ESCHA_STRICT the fused escham_code_gemm path is forced
        # regardless of ESCHA_PREFILL — a stale `ESCHA_PREFILL=ref`
        # (pinned by the old conservative profile) must not reintroduce ref_gemm.
        use_fused = (
            (_PREFILL_MODE == "fused" or _NO_REF) and HAS_ESCHAM_GEMM and _large_m
        )
        # "fused" degrades to "recon" when the installed escha predates
        # the escham_code_gemm op — otherwise a stale kernel build would silently
        # drop the default all the way back to ref (2.7x slower).
        use_recon = (
            _large_m
            and HAS_REF_RECON
            and (
                _PREFILL_MODE == "recon"
                or (_PREFILL_MODE == "fused" and not HAS_ESCHAM_GEMM)
            )
        )
        # fp16-accumulate is toggled ONCE around the whole shard loop rather than
        # per matmul: it flips a PROCESS-GLOBAL cuBLAS math-mode setting, and doing
        # that ~400x per prefill chunk is untested churn. The first fp16acc serving
        # arm (2026-08-12, toggle per shard) died mid-sweep with a sticky
        # "unspecified launch failure" surfaced at the next error check
        # (transform.cu:139) while the fp32-acc arms ran clean — so the toggle rate
        # is the prime suspect and this keeps it at 1 per apply().
        _acc_prev = None
        if use_recon and _PREFILL_FP16ACC:
            _acc_prev = torch.backends.cuda.matmul.allow_fp16_accumulation
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        try:
            out_parts = self._apply_shards(
                layer, x_2d, batch, x_dtype, use_recon, use_fused
            )
        finally:
            if _acc_prev is not None:
                torch.backends.cuda.matmul.allow_fp16_accumulation = _acc_prev

        if not out_parts:
            # 2026-09-18 デバッグ: どの層でシャードが空になるか特定する
            import os as _d

            if _d.environ.get("ESCHA_DEBUG_SPLIT") == "1":
                _pfx = getattr(layer, "prefix", "?")
                _cfgs = getattr(layer, "escha_shard_configs", None)
                _code = getattr(layer, "escha_shard_code", None)
                print(
                    f"[ESCHA_EMPTY] prefix={_pfx} configs={_cfgs} "
                    f"code_is_none={[c is None for c in (_code or [])]}",
                    flush=True,
                )
            raise RuntimeError(
                f"escha: no shards built for layer {getattr(layer,'prefix','?')}"
            )
        out = torch.cat(out_parts, dim=-1) if len(out_parts) > 1 else out_parts[0]
        out = out.reshape(orig_shape[:-1] + (out.shape[-1],))
        if bias is not None:
            out = out + bias
        return out

    def _apply_shards(self, layer, x_2d, batch, x_dtype, use_recon, use_fused=False):
        """Per-shard kernel dispatch; returns the parts for the caller to concat.

        Split out of apply() only so the fp16-accumulate toggle can wrap the whole
        loop in a try/finally without indenting the loop body. Dispatch order per
        shard: escham_decode_gemv (decode) -> recon+cuBLAS (large M, opt-in) ->
        ref_gemm -> cached dense fallback.
        """
        out_parts = []
        for i, cfg in enumerate(layer.escha_shard_configs):
            if cfg is None or layer.escha_shard_code[i] is None:
                continue
            L, K, V, cb_id, IC, OC = cfg
            code = layer.escha_shard_code[i]
            rin = layer.escha_shard_rin[i]
            rout = layer.escha_shard_rout[i]

            part = None
            # Decode fast path (batch <= the kernel's PROBED max M): our fused kernel does
            # had_in(x*s_in, rin) + mma-GEMV + had_out + rout + s_out in one op.
            # Beats ref_gemm (~1.08-1.28x) and, being deterministic (no autotune)
            # + a torch.ops op (no isfinite host-sync), is CUDA-graph capturable —
            # which raw-pybind ref_gemm is not. The kernel computes all rows in one
            # mma (weight decoded once), so a single batched call covers the whole
            # decode bs SGLang captures (else capture hits the ref branch — which on
            # the 27B aborted graph capture at bs16, reconstruct.cu:138).
            # K in (2, 3): the dense escham kernels are templated on K as of the
            # mixed-bit work (escham_gemv_bw_kernel / escham_code_gemm_kernel /
            # escham_multi_gemv_*), gated by tests/test_dense_k3_parity.py which
            # holds K=2 BIT-IDENTICAL and checks K=3 against escham_reconstruct.
            if (
                HAS_ESCHAM
                and not _FORCE_REF
                and K in (2, 3)
                and batch <= _escham_max_m()
                and IC % 128 == 0
                and OC % 128 == 0
            ):
                try:
                    part = torch.ops.escha.escham_decode_gemv(
                        x_2d.to(torch.half).contiguous(),
                        code,
                        rin,
                        rout,
                        layer.escha_shard_s_in[i],
                        layer.escha_shard_s_out[i],
                        OC,
                        K,
                        bool(cb_id == 1),
                        bool(cb_id == 2),
                    ).to(x_dtype)
                except Exception:  # pragma: no cover
                    if _NO_REF:
                        raise
                    part = None
            # Prefill (batch > the escham kernel's max M) alternative: transient raw
            # reconstruct + cuBLAS. Strictly large-M — decode must stay on
            # escham_multi_gemv / escham_decode_gemv above (this path is not CUDA-graph
            # capturable and is slower than the fused GEMV at small M).
            # Prefill alternative #1: the fused code GEMM (one launch, no fp16
            # weight materialised at all — so it also removes the recon path's
            # transient (IC,OC) buffer, which matters under the 22 GB cap).
            if (
                part is None
                and use_fused
                and K in (2, 3)
                and IC % 128 == 0
                and OC % 128 == 0
            ):
                try:
                    part = torch.ops.escha.escham_code_gemm(
                        x_2d.to(torch.half).contiguous(),
                        code,
                        rin,
                        rout,
                        layer.escha_shard_s_in[i],
                        layer.escha_shard_s_out[i],
                        OC,
                        K,
                        bool(cb_id == 1),
                        bool(cb_id == 2),
                        _acc_mode_for(IC),
                    ).to(x_dtype)
                except Exception:  # pragma: no cover — fall through to recon/ref
                    if _NO_REF:
                        raise
                    part = None
            # Capture-time route guard: escham_decode_gemv and escham_code_gemm above
            # are both CUDA-graph capturable (registered torch.ops with fake/meta
            # impls, deterministic allocation, no host sync); recon / ref_gemm /
            # the dense fallback below are not — ref_gemm is a raw pybind call
            # with a host sync (the isfinite NaN guard), and recon and the dense
            # fallback are plain Python/eager ops with no fake/meta impl or
            # graph-safe allocation contract, not raw-pybind-specific. A shard
            # reaching here mid-capture would bake whichever fallback runs at
            # capture time into the graph and silently diverge from replay — the
            # 35B stale-ptr class of bug. Checked only here, once a shard is
            # actually about to leave the capturable path, so it never taxes the
            # hot (escham-covered) route.
            if part is None and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"escha: shard {i} (batch={batch}) left the escham kernel path "
                    "DURING CUDA-graph capture — the ref/recon/dense branches are "
                    "not capturable and would bake wrong arithmetic into the graph. "
                    "Cap --cuda-graph-bs at torch.ops.escha.escham_decode_gemv_max_m()."
                )
            # Prefill alternative #2: transient raw reconstruct + cuBLAS.
            if part is None and (use_recon or use_fused):
                s_in, s_out = _a_half_cached(layer, i, x_dtype)
                try:
                    part = _prefill_recon(
                        x_2d,
                        code,
                        rin,
                        rout,
                        s_in,
                        s_out,
                        IC,
                        OC,
                        K,
                        bool(cb_id == 1),
                        bool(cb_id == 2),
                        x_dtype,
                    )
                except Exception:  # pragma: no cover — fall through to ref_gemm
                    part = None
            if part is None and HAS_REF and not _NO_REF:
                # Prefill (batch>1) / fallback: ref_gemm (input had fused),
                # s_in pre / s_out post. Keep the first-call NaN guard (autotune).
                #
                # s_in/s_out are stored fp32; the fp16 casts live HERE, cached per
                # shard, because only this path and the dense fallback consume them.
                # They used to run unconditionally at the top of the shard loop —
                # 2 casts x ~400 shards = ~817 dead kernel launches and 0.95 ms per
                # decode step (6.3% of bs1, profiled 2026-07-17) computing values
                # the escham fast path never reads. The ZML "aux ops splitting command
                # buffers" lesson, in torch form.
                s_in, s_out = _a_half_cached(layer, i, x_dtype)
                A = (x_2d * s_in).to(torch.half).contiguous()
                A_had = torch.empty_like(A)
                C = torch.empty((batch, OC), dtype=torch.half, device=A.device)
                try:
                    _ref_ext.ref_gemm(
                        A,
                        code,
                        C,
                        rin,
                        A_had,
                        rout,
                        -1,
                        bool(cb_id == 1),
                        bool(cb_id == 2),
                        0,
                    )
                    # REVERTED 2026-08-12: the first-3-calls-only NaN-guard
                    # variant (+4% prefill @32K) FAILED the consolidated MATH-500
                    # gate — mixed prefill+decode ref traffic without this
                    # per-call device sync hit cudaErrorLaunchFailure ~8 min in.
                    # The sync is accidentally load-bearing: it serializes raw-
                    # pybind ref_gemm calls. Do not remove without an ref-side
                    # stream-safety fix.
                    if torch.isfinite(C).all():
                        part = C.to(x_dtype) * s_out
                except Exception:  # pragma: no cover
                    part = None
            if part is None:
                if _NO_REF:
                    raise RuntimeError(
                        f"ESCHA_STRICT: shard {i} (K={K}, IC={IC}, OC={OC}, "
                        f"batch={batch}) fell out of the escham kernel path — the "
                        "strict profile refuses the ref/dense fallback because a "
                        "silent kernel-route change mid-run un-pairs eval arms."
                    )
                # Dense fallback: (x * s_in) @ W_deploy * s_out
                s_in, s_out = _a_half_cached(layer, i, x_dtype)
                w = self._dense_fallback(layer, i, cfg).to(x_dtype)  # (IC, OC)
                part = ((x_2d * s_in) @ w) * s_out
            out_parts.append(part)

        return out_parts
