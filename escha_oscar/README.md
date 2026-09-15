# Escha-W2 (2-bit MoE) + OSCAR INT2 KV — production setup and patches

**A 35B model with a 262,144-token context, serving 8 concurrent users, on two
12 GB consumer GPUs.**

This is the full configuration for running **Qwen3.6-35B-A3B-Escha-W2** — a
2-bit quantized MoE — together with **OSCAR INT2 KV cache**, plus the patches
needed to make the combination actually fit and stay responsive.

The two do not ship together. The `escha` wheel provides the 2-bit MoE weights
and CUDA kernels; OSCAR INT2 KV lives in an open sglang PR (#32129). Everything
below was measured on the setup described, not estimated.

---

## At a glance

| | |
|---|---|
| **Model** | `Qwen3.6-35B-A3B-Escha-W2` — Qwen3.6-35B-A3B, 2-bit MoE, 256 experts, 3B active |
| **Model size on disk** | **12 GB** (2-bit experts, int8 dense layers) |
| **GPUs** | 2× 12 GB consumer Blackwell (sm_120), tensor-parallel 2 + expert-parallel 2 |
| **VRAM used** | **11.3 GB / 11.7 GB** of 12.0 GB per card |
| **Context length** | **262,144 tokens** |
| **KV cache** | OSCAR **INT2**, group size 32 — K 0.21 GB + V 0.21 GB |
| **KV pool** | **350,257 tokens** total across all concurrent requests |
| **Max concurrency** | **12** (see the note below — asking for 16 gets you 12) |
| **Runtime** | `escha-1.0.2+qwen3moe` (an sglang fork) + OSCAR INT2 patches |
| **OS** | WSL2 / Ubuntu |

### What you actually get

| | |
|---|---|
| Decode, one stream | **87.5 → 56.8 tok/s** (falls as context grows) |
| Decode, 8 concurrent, 400-token answers | **295 tok/s** aggregate |
| First token, 180 K cold prompt | **46.2 s** |
| First token, same 180 K prompt resent (prefix cache hit) | **1.87 s** |
| Short request arriving during a long prefill | delayed **1.10×**, p50 **0.29 s** |

Model quality, from the model card (not measured here): HumanEval+ 92.07,
MMLU-Pro 80.9, MATH-500 93.8, GPQA-Diamond 77.8.

### How the KV pool translates to real sessions

The pool is 350,257 tokens **shared by everyone at once**. That is the number
that decides how many people can use the box, not the 262 K context limit.

| One session of | Pool used | Roughly how many fit |
|---|---|---|
| 8 K | 2 % | dozens (concurrency caps at 12 first) |
| 32 K | 9 % | 11 |
| 80 K | 23 % | 4 |
| 180 K | 51 % | 1, plus short ones |
| 262 K (full) | 75 % | 1 |

> **Concurrency is clamped.** `--max-running-requests 16` comes back as
> `max_running_requests=12` — the engine reduces it given the mamba cache size
> and the pool. Size your capacity planning against **12**.

> **The VRAM cliff is at 11,830 MiB.** Past it the GPU reports 100 %
> utilisation while drawing 40–50 W, and the watchdog eventually fires. Under
> WSL this never surfaces as an allocation failure — it just looks like the
> model got slow. The settings here deliberately stop below it.

---

## Getting it running

Links you will need:

- Model — [EschaLabs/Qwen3.6-35B-A3B-Escha-W2](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2)
- Runtime — [EschaLabs/escha-runtime-qwen3moe](https://huggingface.co/EschaLabs/escha-runtime-qwen3moe) (the wheel, `serve.sh` and `thinking_budget.py` live here)
- OSCAR INT2 KV — [sglang PR #32129](https://github.com/sgl-project/sglang/pull/32129)

### 1. Environment

The wheel is built for **CPython 3.12** (`cp312`, `manylinux_2_28`). You need
glibc ≥ 2.28 and an NVIDIA driver — no CUDA toolkit (`ptxas` ships inside
`triton`).

```bash
micromamba create -n eschamoe python=3.12 -y     # or conda/venv
micromamba activate eschamoe
```

### 2. Runtime and model

```bash
git clone https://huggingface.co/EschaLabs/escha-runtime-qwen3moe
pip install escha-runtime-qwen3moe/sglang/escha-1.0.2+qwen3moe-cp312-cp312-manylinux_2_28_x86_64.whl

git clone https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2   # 12 GB
```

The wheel installs a `sglang` package — that is the fork it serves with.

### 3. Add OSCAR INT2 KV

The wheel does **not** include it. Take the INT2 files from PR #32129 and drop
them into the installed tree, then apply the patches in `patches/`:

```bash
SP=$(python -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent.parent)")
cd "$SP"
for p in /path/to/escha_oscar/patches/*.patch; do
    patch -p0 --backup --suffix=.orig < "$p"
done
```

You also need the rotation tensors OSCAR uses. Plain Hadamard is what is used
here — see *Two results that went against expectation* for why the calibrated
ones were dropped.

```bash
export SGLANG_OSCAR_K_ROTATION_PATH=/path/to/k_rotation_hadamard_hd256.pt
export SGLANG_OSCAR_V_ROTATION_PATH=/path/to/v_rotation_hadamard_hd256.pt
```

> Patching `site-packages` is undone by any wheel reinstall. Keep the `.orig`
> backups, or put the modified files on `PYTHONPATH` as an overlay instead of
> editing in place.

### 4. Launch

`serve.sh` (from the runtime repo) takes its configuration from the
environment. On WSL you also have to point `LD_LIBRARY_PATH` at the NVIDIA
libraries pip installed:

```bash
NVLIBS=$(python -c "import nvidia, pathlib; print(pathlib.Path(nvidia.__file__).parent)")
export LD_LIBRARY_PATH="$(find "$NVLIBS" -maxdepth 2 -type d -name lib | paste -sd: -):/usr/lib/wsl/lib"

export MODEL=/path/to/Qwen3.6-35B-A3B-Escha-W2
export PORT=8081 HOST=127.0.0.1
export MEM=0.88 CTXLEN=262144 MAXMAMBA=64 MAXREQ=16 RADIX=1
export MAMBA_SCHED=extra_buffer ATTN_BACKEND=triton
export GRAPHS=1 CUDA_GRAPH_BS="1 2 4 8 16"
export ENABLE_CLP=1
export SGLANG_CHUNK_DECODE_CAP=1536 SGLANG_CHUNK_FAIR_FLOOR=256

bash escha-runtime-qwen3moe/sglang/serve.sh \
  --tp-size 2 --ep-size 2 \
  --kv-cache-dtype int2 --kv-cache-quant-group-size 32 \
  --enable-metrics --enable-cache-report \
  --allow-auto-truncate
```

Startup takes about 33 s. You should see:

```
KV Cache is allocated. #tokens: 350257, K size: 0.21 GB, V size: 0.21 GB
max_total_num_tokens=350257, ... max_running_requests=12, context_len=262144
```

### 5. Check it

```bash
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"escha-qwen36-35b-a3b-w2",
       "messages":[{"role":"user","content":"What is 1+1?"}],
       "max_tokens":64}' | python -m json.tool
```

---

## Settings you must not change

Every one of these was learned by breaking it.

| Setting | Why |
|---|---|
| `--ep-size 2` | With `--tp-size 2` alone the expert weights are **replicated on both GPUs**: 11.06 GB against 12 GB, and it will not load |
| `ATTN_BACKEND=triton` | On sm_120 (RTX 50-series) flashinfer **asserts** on the hybrid-GDN path |
| `--kv-cache-dtype int2` | fp16 KV needs 2.52 GB per GPU for 262 K. INT2 needs **0.39 GB**. Without it the context does not fit, full stop |
| `MAMBA_SCHED=extra_buffer` | The only way to keep the overlap scheduler while `RADIX=1`. With `no_buffer` the log says *"Disabling overlap schedule"* and aggregate throughput drops **14 %** |
| `MEM=0.88`, `MAXMAMBA=64` | **Do not raise these.** At `MEM=0.92` / `MAXMAMBA=80` the GPU reached 11,791 MiB as the pool filled, crossed the **VRAM cliff (~11,830 MiB)**, and prefill collapsed from 3,056 to **16 tok/s** — a 190× loss. At 0.88/64 the pool is *larger* (350,257 vs 303,751) with ~1.1 GB of headroom. The only cost is concurrency 16 → 12 |

**Do not enable `--enable-mixed-chunk`.** Combined with the INT2 kernels it
crashes with `CUDA error: invalid configuration argument`. `CHUNK=8192`
alongside it also fails; `CHUNK=4096` on its own works but trades aggregate
throughput for single-stream speed.

## The patches

| Patch | What it fixes | Measured effect |
|---|---|---|
| `..._triton_backend` + `..._kv_quant_kernels` + `..._quantized_kv_prefill` | INT2 prefill holds the expanded prefix **three times** (`3P+2E`). Preallocate the combined buffer and dequantize straight into its slices (`out=`) → `P+E` | **−746 MB (−37 %)** transient at 192 K. This is what lets a long context and 8 streams coexist at all |
| `..._scheduler` + `..._schedule_policy` | A long prefill starves streams that are already decoding: the batch duration becomes a gap between their tokens | Worst inter-token gap **5.1 s × 21 occurrences → 0.14 s**. Enabled by `SGLANG_CHUNK_DECODE_CAP`; the default `0` is byte-identical to stock behaviour |
| `..._decode_attention` | `_safe_block_h` — a grouped-decode head tile straddling a KV head makes query heads read **another KV head's cache**. Silent: nothing asserts, no NaN, only the benchmark score moves | See below |

### The silent one

`_fwd_grouped_kernel_stage1` addresses heads as

```
VALID_BLOCK_H = min(BLOCK_H, kv_group_num)
cur_head      = cur_head_id * VALID_BLOCK_H + arange(BLOCK_H)
cur_kv_head   = cur_head_id // cdiv(kv_group_num, BLOCK_H)
```

`cur_head` is a flat query-head index while `cur_kv_head` comes from the block
index. They agree only when a head block sits wholly inside one KV group, i.e.
`BLOCK_H >= kv_group_num` **or** `kv_group_num % BLOCK_H == 0`.

Stock sglang hardcodes `BLOCK_H = 16`, which holds for every power-of-two
`kv_group_num` — which is why it has never bitten upstream. It stays reachable
there with `kv_group_num > 16` that is not a multiple of 16 (96 query heads over
4 KV heads, say), and it bites at once if `BLOCK_H` is tuned or made batch-size
dependent: `BLOCK_H=4` against `kv_group_num=6` makes head block 1 cover query
heads 4..7 while reporting `cur_kv_head=0`, so query heads 6 and 7 quietly
attend to KV head 0.

`_safe_block_h` rounds up to a power of two so that
`VALID_BLOCK_H == kv_group_num` — exactly one block per KV head.

---

## Two results that went against expectation

**Calibrated rotations made quality worse.** The OSCAR paper reports a large
loss *without* calibration (−66.29 pt on Qwen3-32B). On this checkpoint the
opposite held: **40/41 uncalibrated vs 37/41 calibrated** on a long-context
discrimination test, with third-task recall dropping 0.969 → 0.819. Plain
Hadamard rotations are used here. Calibrate *and then measure*; do not assume.

**Needle-in-a-haystack could not tell the two apart.** It scored 15/15 on
everything, including configurations that were visibly worse in use. The
discriminating test needed many similar distractors, aggregation across
positions, and ordered enumeration.

---

## Licence

sglang is Apache-2.0 and so are these patches. The `escha` runtime and the
Escha-W2 weights are distributed separately by their authors and are **not**
redistributed here.
