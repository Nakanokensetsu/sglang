#!/usr/bin/env bash
# 【2026-09-18 確定】本番 dense: Qwen3.8-27B-Escha-W2 + OSCAR INT2 KV on 素の sglang 0.5.19
#
# 0.5.15(escha wheel 同梱フォーク)からの移行。掃引20構成超の実測で決定した。
#
# 最大の変更点: --disable-prefill-cuda-graph
#   0.5.19 の新機能 prefill CUDA graph は 51形状(4096〜4)を capture して 1.81GB を握る。
#   int2 KV の prefill は prefix を dequant して展開するため、シーケンス長に比例した
#   一時メモリ(112K で約 962MB)が要る。両者が競合して VRAM 崖に落ちる。
#   切ると残VRAM 0.54 → 1.80GB、96K prefill が 202 → 959 tok/s。短文も4〜7%速い。
#   CHUNK 4096 運用では実際に使う形状はほぼ 4096 だけなので恩恵が無く代償だけだった。
#   decode graph は bs=[1,2,4] のみで無駄が無いので残す。
#
# --mamba-radix-cache-strategy extra_buffer_lazy:
#   1リクエストが消費する mamba スロットが 5 → 4。max_running_requests は
#   スロット数÷4 に丸められるので、mamba 32 でちょうど並列8になる。
#   同一VRAM条件で並列4 +47% / 並列8 +32%(bs1 は不変)。
#
# 【2026-09-20 変更】長文の同時保持限界を上げる3点セット(実測で決定):
#   --mem-fraction-static 0.78 -> 0.85    静的予算を +856MiB。KV プールが 148,058 -> 210,000
#   --max-total-tokens 150000 -> 210000   上限の解除。**これ単体では効かない**(予算側が縛る)
#   --chunked-prefill-size 4096 -> 8192   mamba 必要スロット半減(210,000/8,192 = 26 <= 32)
# 3つが連動しており、1つでも欠けると成立しない。CHUNK を上げずにプールだけ広げると
# 210,000/4,096 = 51 > 32 で mamba の prefix cache 容量条件を割り、再利用が崩壊する。
# 効果: 66K 文脈 3ユーザの再訪が 12.4%/57秒 -> 99.9%/0.3秒。
# 速度は全指標で同等以上(冷prefill 970->1010 tok/s、decode 51.9 で同一、
# 長文prefill中の短文TTFT 0.135->0.123s、並列10 エラー0・電力169->175W で崖なし)。
# 品質 Test D(96K) 17/20 -> 18/20。
#
# --context-length 114688:
#   実測で全速を確認した最大値。104K/112K/120K はいずれも 888〜922 tok/s・165W で全速、
#   130K(実122,133tok)で崖(450 tok/s)。設定値を実用上限より大きくすると
#   利用者が知らずに崖を踏むので、ここで止める。
set -uo pipefail
cd ${VENV_ROOT:?set to the venv holding sglang 0.5.19}

# --- 提供モデル名はランタイムの serve.sh から自動取得(ハードコードしない) ---
SERVE_SH=${RUNTIME_DIR:?set to the escha runtime}/sglang/serve.sh
SERVED_NAME=$(grep -oE 'SERVED_NAME=\$\{SERVED_NAME:-[^}]+\}' "$SERVE_SH" 2>/dev/null | sed -E 's/.*:-([^}]+)\}/\1/')
: "${SERVED_NAME:?serve.sh から SERVED_NAME を取得できなかった}"
PORT=${PORT:-8081}

# --- 0.5.15 本番から持ち込む環境変数 ---
export SGLANG_MAMBA_CONV_DTYPE=float16        # --dtype float16 と揃えないと causal_conv1d が落ちる
export SGLANG_DISABLE_CUDNN_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 予約と実使用の差を詰める
export PYTORCH_ALLOC_CONF=expandable_segments:True        # 新しい torch 用の別名
export HF_HUB_OFFLINE=1
export SGLANG_INT8_LM_HEAD=0
# 2026-09-19 公式クックブック(Qwen3.8-27B)記載のノブ。extra_buffer系で
# 1リクエストあたりのGDN状態スロットを1つ解放する(extra_buffer_lazy: S=4 -> 3)。
# --max-mamba-cache-size 32 のままで並列上限が 32/4=8 から 32/3=10 に上がる。
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
export SGLANG_OSCAR_K_ROTATION_PATH=${ROTATION_DIR:?set to your OSCAR rotation checkpoints}/k_rotation_qqt_r_h_pbr.pt
export SGLANG_OSCAR_V_ROTATION_PATH=${ROTATION_DIR:?set to your OSCAR rotation checkpoints}/v_rotation_sst_r_h_pbr.pt
# clip / Lloyd-Max は既定OFFのまま。2026-09-14 に 0.96/0.92+LloydMax を実測し
# 判別テスト D が 38/41 → 33/41 と悪化・prefill -1.7% のため不採用。

# --- ビルド/実行環境 ---
# 2026-09-20: OSCAR mixed-KV 移植ツリーを PYTHONPATH で被せる。
# venv の sglang 0.5.19 は無傷のまま残す(pip 再インストールで消えない利点がある一方、
# 逆に venv 側を更新しても効かなくなる。移植ツリーが正)。
# 中身: 上流 #38191/#37943/#39526 の取り込み + mixed-KV 移植(既定OFFで不活性)。
# 戻す時はこの export を消すだけ。
export PYTHONPATH=${PORT_TREE:?set to this repo's python/ overlay}${PYTHONPATH:+:$PYTHONPATH}

NVLIBS=${VENV_ROOT:?set to the venv holding sglang 0.5.19}/.venv/lib/python3.12/site-packages/nvidia
export PATH="/usr/local/cuda/bin:$PATH"
export LIBRARY_PATH="/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH="$(find "$NVLIBS" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -):/usr/lib/wsl/lib"

echo "[escha-0519] name=$SERVED_NAME port=$PORT ctx=114688 mamba=32 並列8 prefill-graph=OFF"

exec ./.venv/bin/python -m sglang.launch_server \
  --model-path ${MODEL_DIR:?set to your Qwen3.8-27B-Escha-W2 checkout} \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --dtype float16 --tp-size 2 \
  --attention-backend triton \
  --mamba-ssm-dtype float16 \
  --kv-cache-dtype int2 --kv-cache-quant-group-size 64 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --context-length 114688 \
  --chunked-prefill-size 8192 \
  --max-mamba-cache-size 32 \
  --disable-prefill-cuda-graph \
  --max-running-requests 10 \
  --max-total-tokens 210000 \
  --mem-fraction-static 0.85 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-custom-logit-processor \
  --enable-metrics --enable-cache-report --log-requests \
  --soft-watchdog-timeout 120 --watchdog-timeout 600 \
  --enable-mixed-chunk \
  --triton-attention-num-kv-splits 64 \
  --allow-auto-truncate \
  --trust-remote-code --log-level info "$@"
