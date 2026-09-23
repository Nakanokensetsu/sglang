"""DI ESCHA MoE quantization for SGLang (eschamoe).

Serves gpt-oss-style MoE models whose EXPERTS are code-quantized (ESCHAM)
to 2-bit, exported by ``scripts/gptoss/export_gptoss_di_moe.py``. Attention,
router, embeddings, lm_head and norms stay FP16 (the ``ignore`` set).

Per FusedMoE layer the export stores, stacked over the E experts, for each
projection p in {gate_up_proj, down_proj}:
    {p}.escha_code  int16 [E, in_p//16, out_p//16, 16*K]
    {p}.escha_rin      fp16  [E, in_p]     (rot in scale; s_in folded when --fold-scales)
    {p}.escha_rout      fp16  [E, out_p]    (rot out scale; s_out folded)
    {p}.escha_s_in     fp32  [E, in_f]     (ones when folded)
    {p}.escha_s_out    fp32  [E, out_f]
    {p}.escha_bias     fp16  [E, out_f]    (original FP expert bias)
    {p}.escha_config   int32 [9]           ([L,K,V,cb_id,E,in_f,out_f,in_p,out_p])

The runtime forward reuses escha's validated ``PackedGptOssExperts`` (decode the
frozen code on the fly + the exact clamped-interleaved-SwiGLU routing). The
GEMM backend is swappable (pytorch decode-loop first for correctness; a
future user-authored grouped-GEMM kernel can replace ``apply`` later).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Parameter

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = logging.getLogger(__name__)

_PROJS = ("gate_up_proj", "down_proj")
_SUFFIXES = (
    "escha_code",
    "escha_rin",
    "escha_rout",
    "escha_s_in",
    "escha_s_out",
    "escha_bias",
    "escha_config",
)


def _round_up(n: int, m: int = 128) -> int:
    return ((n + m - 1) // m) * m


class DIEschaMoEConfig(QuantizationConfig):
    """Config for eschamoe — reads the quantization_config block our exporter writes."""

    def __init__(
        self,
        codebook: str,
        codebook_id: int,
        bits: float,
        num_experts: int,
        fold_scales: bool,
        int8_embedding: bool,
        ignore: List[str],
        layer_meta: Dict[str, Any],
        full_config: Dict[str, Any],
        experts_kind: str = "gptoss",
    ) -> None:
        super().__init__()
        self.codebook = codebook
        self.codebook_id = codebook_id
        self.bits = bits
        self.num_experts = num_experts
        self.fold_scales = fold_scales
        self.int8_embedding = int8_embedding
        self.ignore = ignore or []
        self.layer_meta = layer_meta or {}
        self.full_config = full_config
        # Which decode-on-the-fly experts module the runtime builds:
        #   "gptoss"   -> PackedGptOssExperts (interleaved/clamped/(up+1) SwiGLU + bias)
        #   "qwen35moe"-> PackedQwen35MoeExperts (contiguous chunk(2) + plain SiLU, no bias)
        # Picking the wrong one is a SILENT numeric corruptor.
        self.experts_kind = experts_kind

    @classmethod
    def get_name(cls) -> str:
        return "eschamoe"

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
    def from_config(cls, config: Dict[str, Any]) -> DIEschaMoEConfig:
        g = config.get("global_config", {})
        return cls(
            codebook=g.get("codebook", "cbA"),
            codebook_id=g.get("codebook_id", 1),
            bits=g.get("bits", 2.0),
            num_experts=g.get("num_experts", config.get("num_experts", 0)),
            fold_scales=config.get("fold_scales", g.get("fold_scales", False)),
            int8_embedding=config.get("int8_embedding", False),
            ignore=config.get("ignore", []),
            layer_meta=config.get("layer_meta", {}),
            full_config=config,
            experts_kind=g.get("experts_kind", "gptoss"),
        )

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        # Experts -> our code MoE method.
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if isinstance(layer, FusedMoE):
            return DIEschaMoEMethod(self)
        # FP16-kept linears (attn/router/lm_head): UnquantizedLinearMethod (NOT None —
        # LinearBase asserts quant_method is not None). Everything in this export is FP
        # except the experts, so any LinearBase is unquantized.
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None


class DIEschaMoEMethod(FusedMoEMethodBase):
    """FusedMoE method that loads stacked per-expert code codes and runs the
    decode-on-the-fly gpt-oss MoE forward (reusing escha's PackedGptOssExperts)."""

    def __init__(self, quant_config: DIEschaMoEConfig) -> None:
        self.quant_config = quant_config
        self.K = max(1, int(round(quant_config.bits)))  # uniform-model default
        # Mixed-bit models (e.g. K=2 gate_up + K=3 down_proj) record per-projection K
        # in layer_meta. create_weights allocates ONE placeholder K per projection, so
        # size it at the SMALLEST per-proj K present and let the load path GROW any
        # larger projection (a K3 code replacing the K2 placeholder). Deriving K
        # from the global `bits` instead over-allocates every projection at the max K
        # — ~2.5 GB of transient VRAM on 35B-A3B, which OOMs construction on 16 GB
        # cards (friction 2026-07-23 #1). Decode stays per-proj-correct regardless: K
        # is read from each loaded code's shape in process_weights_after_loading.
        _lm = getattr(quant_config, "layer_meta", None) or {}
        _ks = [
            m["K"]
            for m in _lm.values()
            if isinstance(m, dict) and isinstance(m.get("K"), int)
        ]
        if _ks:
            self.K = max(1, min(_ks))
        self.codebook = quant_config.codebook
        self.experts_kind = getattr(quant_config, "experts_kind", "gptoss")
        self.packed = None  # built in process_weights_after_loading

    # ---- shapes (uniform across experts/layers for gpt-oss) ----
    def _dims(self, hidden: int, inter: int):
        return {
            "gate_up_proj": dict(
                in_f=hidden,
                out_f=2 * inter,
                in_p=_round_up(hidden),
                out_p=_round_up(2 * inter),
            ),
            "down_proj": dict(
                in_f=inter, out_f=hidden, in_p=_round_up(inter), out_p=_round_up(hidden)
            ),
        }

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.utils import set_weight_attrs

        E = num_experts
        K = self.K
        dims = self._dims(hidden_size, intermediate_size_per_partition)
        layer._diescha_E = E
        layer._diescha_hidden = hidden_size
        layer._diescha_inter = intermediate_size_per_partition
        # weight_loader rides in extra_weight_attrs; keep it so the model's load loop
        # finds param.weight_loader (we copy in-place via a custom branch in gpt_oss.py,
        # but keep the attr present for SGLang's loaded-params bookkeeping).
        attrs = dict(extra_weight_attrs)

        def reg(name, tensor):
            p = Parameter(tensor, requires_grad=False)
            layer.register_parameter(name, p)
            set_weight_attrs(p, attrs)

        for proj in _PROJS:
            d = dims[proj]
            in_p, out_p, in_f, out_f = d["in_p"], d["out_p"], d["in_f"], d["out_f"]
            reg(
                f"{proj}_escha_code",
                torch.zeros(E, in_p // 16, out_p // 16, 16 * K, dtype=torch.int16),
            )
            reg(f"{proj}_escha_rin", torch.zeros(E, in_p, dtype=torch.float16))
            reg(f"{proj}_escha_rout", torch.zeros(E, out_p, dtype=torch.float16))
            reg(f"{proj}_escha_s_in", torch.ones(E, in_f, dtype=torch.float32))
            reg(f"{proj}_escha_s_out", torch.ones(E, out_f, dtype=torch.float32))
            reg(f"{proj}_escha_bias", torch.zeros(E, out_f, dtype=torch.float16))
            reg(f"{proj}_escha_config", torch.zeros(9, dtype=torch.int32))

    def process_weights_after_loading(self, layer) -> None:
        """Build escha's PackedGptOssExperts from the loaded stacked params, then
        load the per-expert s_in/s_out (ones when --fold-scales)."""
        from escha.gptoss_experts import PackedGptOssExperts
        from transformers import AutoConfig  # noqa: F401 (not needed; build cfg shim)

        E = layer._diescha_E
        H, I = layer._diescha_hidden, layer._diescha_inter
        dims = self._dims(H, I)
        codes: Dict[str, dict] = {}
        for proj in _PROJS:
            d = dims[proj]
            tr = getattr(layer, f"{proj}_escha_code")
            rin = getattr(layer, f"{proj}_escha_rin")
            rout = getattr(layer, f"{proj}_escha_rout")
            bias = getattr(layer, f"{proj}_escha_bias")
            # PER-PROJECTION K (mixed-bit safe): the code last dim is 16*K, so a K3
            # down_proj (48) and K2 gate_up (32) in one layer decode with their own K.
            # self.K = round(bits) is only the uniform-model default.
            _Kp = int(tr.shape[-1] // 16)
            for e in range(E):
                codes[f"{proj}_{e}"] = {
                    "code": tr[e],
                    "rin": rin[e],
                    "rout": rout[e],
                    "bias": bias[e],
                    "K": _Kp,
                    "codebook": self.codebook,
                    "in_f": d["in_f"],
                    "out_f": d["out_f"],
                    "in_p": d["in_p"],
                    "out_p": d["out_p"],
                }

        dev = getattr(layer, "gate_up_proj_escha_code").device
        if getattr(self, "experts_kind", "gptoss") == "qwen35moe":
            # Qwen3.5/3.6 MoE: contiguous chunk(2) + plain SiLU, no bias/clamp.
            from escha.qwen35_experts import PackedQwen35MoeExperts

            class _Cfg:  # PackedQwen35MoeExperts reads these attrs
                num_experts = E
                hidden_size = H
                moe_intermediate_size = I

            packed = PackedQwen35MoeExperts(codes, _Cfg(), bias_correction=False).to(
                dev
            )
        else:

            class _Cfg:  # PackedGptOssExperts reads these attrs
                num_local_experts = E
                hidden_size = H
                intermediate_size = I
                alpha = 1.702
                limit = 7.0

            packed = PackedGptOssExperts(codes, _Cfg(), bias_correction=False).to(dev)
        # load s_in/s_out (ones when folded) into the per-expert scale params
        for proj in _PROJS:
            s_in = getattr(layer, f"{proj}_escha_s_in")
            s_out = getattr(layer, f"{proj}_escha_s_out")
            D = (
                packed.gate_up_experts
                if proj == "gate_up_proj"
                else packed.down_experts
            )
            for name, m in D.items():
                e = int(name.rsplit("_", 1)[1])
                with torch.no_grad():
                    m.s_in.copy_(s_in[e].to(m.s_in.dtype))
                    m.s_out.copy_(s_out[e].to(m.s_out.dtype))
        self.packed = packed
        layer._diescha_packed = packed
        # Dedup: the packed experts now own their scales — s_in/s_out are INDEPENDENT copies
        # (nn.Parameter + copy_ above), so dropping the raw stacked s_in/s_out/bias/config frees
        # them. (rin/rout/code are VIEWS into the raw stacked params — deleting those wrappers is
        # a no-op free, so we keep them; the storage is the packed experts' single copy.)
        for _proj in _PROJS:
            for _suf in ("escha_s_in", "escha_s_out", "escha_bias", "escha_config"):
                _nm = f"{_proj}_{_suf}"
                if _nm in layer._parameters:
                    del layer._parameters[_nm]
                elif _nm in getattr(layer, "_buffers", {}):
                    del layer._buffers[_nm]
                elif hasattr(layer, _nm):
                    delattr(layer, _nm)
        # free the raw stacked params (PackedGptOssExperts holds its own views/copies)
        torch.cuda.empty_cache()
        import os as _os

        if _os.environ.get("ESCHA_MEMDIAG"):
            _raw = [
                n
                for n in ("gate_up_proj_escha_code", "down_proj_escha_code")
                if hasattr(layer, n)
            ]
            logger.info(
                "eschamoe MEMDIAG: alloc=%.2fGB reserved=%.2fGB raw_code_still_on_layer=%s",
                torch.cuda.memory_allocated() / 1e9,
                torch.cuda.memory_reserved() / 1e9,
                _raw,
            )
        logger.info(
            "eschamoe: built %s (experts_kind=%s) for a FusedMoE (%d experts)",
            type(packed).__name__,
            getattr(self, "experts_kind", "gptoss"),
            E,
        )

    def create_moe_runner(self, layer, moe_runner_config) -> None:
        self.moe_runner_config = moe_runner_config

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states  # [N, H]
        topk = dispatch_output.topk_output
        topk_ids = topk.topk_ids.to(torch.long)  # [N, top_k]
        topk_w = topk.topk_weights.to(x.dtype)  # [N, top_k]
        packed = self.packed if self.packed is not None else layer._diescha_packed
        out = packed(x, topk_ids, topk_w)  # routed + SwiGLU + bias + weight + sum
        return StandardCombineInput(hidden_states=out.to(x.dtype))
