#!/usr/bin/env bash
# OSCAR INT2 統合チェック: site-packages に必要な要素が揃っているか
SP=$(python -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent.parent)")
ng=0
chk() {  # chk <説明> <ファイル> <探す文字列>
  if grep -q "$3" "$SP/sglang/$2" 2>/dev/null; then
    printf "OK   %s\n" "$1"
  else
    printf "NG   %-46s (%s に %s が無い)\n" "$1" "$2" "$3"; ng=1
  fi
}
for f in QuantKernel/oscar_rotation_clip_int2_kv.py \
         srt/layers/attention/quantized_kv_prefill.py \
         srt/mem_cache/kv_quant_kernels.py; do
  [ -f "$SP/sglang/$f" ] && printf "OK   コピー: %s\n" "$f" || { printf "NG   コピー漏れ: %s\n" "$f"; ng=1; }
done
chk "environ: 回転行列の環境変数"      srt/environ.py                                   "SGLANG_OSCAR_K_ROTATION_PATH"
chk "server_args: int2 の選択肢"        srt/server_args.py                               '"int2"'
chk "server_args: グループサイズ引数"   srt/server_args.py                               "kv_cache_quant_group_size"
chk "model_runner: int2 分岐"           srt/model_executor/model_runner.py               "int2"
chk "kv_cache_mixin: int2 分岐"         srt/model_executor/model_runner_kv_cache_mixin.py "int2"
chk "memory_pool: 回転行列ローダ"       srt/mem_cache/memory_pool.py                     "def load_oscar_rotation_config"
chk "memory_pool: 回転行列取得"         srt/mem_cache/memory_pool.py                     "def get_oscar_rotation"
chk "triton_backend: int2 prefill"      srt/layers/attention/triton_backend.py           "_forward_extend_int2"
chk "decode: int2 カーネル(normal)"     srt/layers/attention/triton_ops/decode_attention.py "def decode_attention_fwd_normal_quant_int2"
chk "decode: int2 カーネル(grouped)"    srt/layers/attention/triton_ops/decode_attention.py "def decode_attention_fwd_grouped_quant_int2"
chk "decode: _safe_block_h パッチ"      srt/layers/attention/triton_ops/decode_attention.py "def _safe_block_h"
[ $ng -eq 0 ] && echo "=> すべて揃っています" || echo "=> 不足あり(上の NG を見てください)"
