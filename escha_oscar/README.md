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

### What you are signing up for

Steps 1, 2, 4 and 5 are ordinary installation and take about half an hour.
**Step 3 is a manual source merge and is most of an afternoon** — the two
projects do not fit together on their own, and no script can do it for you.
Nothing below needs CUDA-kernel experience, but you do need to be comfortable
reading a diff and moving functions between files.

Hardware and software this was done on:

| | |
|---|---|
| GPUs | 2 × 12 GB, compute capability 12.0 (Blackwell, RTX 50-series) |
| Disk | ~15 GB for the model, plus a checkout of sglang |
| Python | CPython **3.12** — the wheel is `cp312` only |
| glibc | ≥ 2.28 (`manylinux_2_28`) |
| CUDA toolkit | not needed; `ptxas` ships inside `triton` |
| OS | WSL2 / Ubuntu (plain Linux is fine; the `LD_LIBRARY_PATH` line in step 4 is WSL-specific) |

Less VRAM than 24 GB total will not fit this configuration. More is fine, and
makes the tuning in *Settings you must not change* less critical.

Work through the steps in order. Each one ends with something to check, and a
short list of what it means when that check fails.

### 1. Environment

The wheel is built for **CPython 3.12** (`cp312`, `manylinux_2_28`). You need
glibc ≥ 2.28 and an NVIDIA driver — no CUDA toolkit (`ptxas` ships inside
`triton`).

```bash
micromamba create -n eschamoe python=3.12 -y     # or conda/venv
micromamba activate eschamoe
python -c "import sys; print(sys.version)"       # must start with 3.12
```

Keep this environment for the runtime alone. The wheel installs a package
called `sglang` that is *not* upstream sglang, so anything else in the same
environment that expects upstream sglang will break.

### 2. Runtime and model

```bash
git clone https://huggingface.co/EschaLabs/escha-runtime-qwen3moe
pip install escha-runtime-qwen3moe/sglang/escha-1.0.2+qwen3moe-cp312-cp312-manylinux_2_28_x86_64.whl

git clone https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2   # 12 GB
```

The wheel installs a `sglang` package — that is the fork it serves with. Do
**not** also `pip install sglang`; the two overwrite each other.

Check it imported, and remember where it landed — every path in step 3 is
relative to this directory:

```bash
python -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent.parent)"
# -> /path/to/env/lib/python3.12/site-packages
```

| If this fails | Cause |
|---|---|
| `ERROR: ... not a supported wheel on this platform` | Not CPython 3.12, or glibc < 2.28 |
| `ImportError` mentioning `libcuda`/`libnvidia` | Driver not visible yet. It is worth continuing — step 4 sets `LD_LIBRARY_PATH` |

### 3. Add OSCAR INT2 KV (the long step)

The wheel does **not** include it, and PR #32129 is **not a drop-in**. Two
things get in the way:

- The PR is written against upstream sglang's current layout, the wheel ships
  an older one. The INT2 decode kernels live in
  `python/sglang/kernels/ops/attention/decode_attention.py` in the PR and have
  to land in `sglang/srt/layers/attention/triton_ops/decode_attention.py` in
  the wheel. `git apply` will not do this for you.
- Of the ten files involved, **six are files the escha fork has already
  modified**. Overwriting them with the PR's copies breaks the runtime. They
  have to be merged by hand.

Budget an afternoon for this step. Everything below is the mapping that was
actually used here, recovered by diffing the running installation against the
untouched wheel.

#### 3a. Pin the PR

```bash
git clone https://github.com/sgl-project/sglang.git sglang-pr32129
cd sglang-pr32129
git fetch origin pull/32129/head:oscar-int2-kv
git checkout oscar-int2-kv
# the mapping below was taken against 4b28b26bb4cca06984a97bacba25eb43cfd44145
```

#### 3b. Copy the four files that are genuinely new

These do not exist in the wheel, so they can be copied verbatim. Source paths
are relative to `sglang-pr32129/python/`, destinations to `$SP` (site-packages):

| Copy from (PR) | To (wheel) |
|---|---|
| `sglang/QuantKernel/oscar_rotation_clip_int2_kv.py` | same path |
| `sglang/srt/layers/attention/quantized_kv_prefill.py` | same path |
| `sglang/srt/mem_cache/kv_quant_kernels.py` | same path |
| `sglang/kernels/ops/attention/decode_attention.py` | **not copied** — see 3c |

```bash
SP=$(python -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent.parent)")
for f in sglang/QuantKernel/oscar_rotation_clip_int2_kv.py \
         sglang/srt/layers/attention/quantized_kv_prefill.py \
         sglang/srt/mem_cache/kv_quant_kernels.py; do
    cp "python/$f" "$SP/$f"
done
```

#### 3c. Merge the six files the escha fork already owns

Sizes are the total delta of the running install against the stock wheel, so
they include the patches applied in step 3e.

| File | Δ lines | What has to come over from the PR |
|---|---|---|
| `srt/environ.py` | 12 | The env keys: `SGLANG_OSCAR_{K,V}_ROTATION_PATH`, `SGLANG_OSCAR_{K,V}_CLIP_RATIO`, `SGLANG_LLOYD_MAX` |
| `srt/server_args.py` | 11 | `int2` added to the `--kv-cache-dtype` choices, plus the new `--kv-cache-quant-group-size` argument and its `kv_cache_quant_group_size` field |
| `srt/model_executor/model_runner.py` | 5 | The `kv_cache_dtype == "int2"` branch that selects the INT2 pool |
| `srt/model_executor/model_runner_kv_cache_mixin.py` | 52 | INT2 branches in KV-pool construction; threading `kv_cache_quant_group_size` through to the pool |
| `srt/mem_cache/memory_pool.py` | 411 | `OscarRotationConfig`, `load_oscar_rotation_config()`, `load_oscar_rotations()`, `_resolve_quant_grouping()`, the scale/zero buffers (`_allocate_scales_zeros_buffers`, `get_{key,value}_scales_zeros`, `get_raw_{key,value}_buffer`, `get_oscar_rotation`) and the `dtype == "int2"` allocation path |
| `srt/layers/attention/triton_backend.py` | 292 | `_forward_extend_int2()` and its dispatch (`is_int2 = getattr(kv_pool, "dtype", None) == "int2"`), `_apply_oscar_rotation`, `apply_inverse_v_rotation` |
| `srt/layers/attention/triton_ops/decode_attention.py` | 1494 | The INT2 decode kernels, taken from the PR's `kernels/ops/attention/decode_attention.py`: `_fwd_kernel_stage1_quant_int2`, `_fwd_grouped_kernel_stage1_quant_int2`, `_decode_att_m_fwd_quant_int2`, `_decode_grouped_att_m_fwd_quant_int2`, `decode_attention_fwd_normal_quant_int2`, `decode_attention_fwd_grouped_quant_int2`, plus `decode_attention_fwd_quantized` and the `_get_scale_group_size` / `_get_shared_kv_scale_group_size` helpers. The PR's `decode_attention_fwd_int2_unified` dispatcher is *not* needed — the wheel's `triton_backend.py` dispatches to the two entry points directly |

The last one is the real work: the PR file sits in a different package, so its
imports and the surrounding helper names have to be rewritten for the
`srt/layers/attention/triton_ops/` tree rather than moved wholesale.

#### 3d. What is deliberately *not* ported

PR #32129 also touches `srt/mem_cache/kv_cache_dtype.py`,
`srt/mem_cache/unified_kv_pool.py`, `srt/models/{qwen3,glm4_moe,utils}.py` and
`QuantKernel/gpu_flush_int2.py`. The first two do not exist in the wheel's
tree, the model files are not needed for Escha-W2, and the data-free Hadamard
*fallback* inside the pool was not ported either — which is why both rotation
paths below are mandatory rather than optional.

#### 3e. Apply the patches in `patches/`

These sit on top of the integration above — they patch files that only exist
once 3b/3c are done.

```bash
SP=$(python -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent.parent)")
cd "$SP"
for p in /path/to/escha_oscar/patches/*.patch; do
    patch -p1 --batch --backup --suffix=.orig < "$p"
done
```

> `-p1`, not `-p0`: the diff headers carry a `<site-packages>/` prefix that has
> to be stripped. With `-p0` every patch reports *"can't find file to patch"*
> and is silently skipped.

Expect six lines of `patching file ...` and nothing else. What the other
outcomes mean:

| Output | Meaning |
|---|---|
| `can't find file to patch` | Wrong directory, or `-p0`. `cd "$SP"` first |
| `Hunk #1 FAILED` on `quantized_kv_prefill.py` / `kv_quant_kernels.py` | Step 3b did not copy the PR's version of that file |
| `Hunk #1 FAILED` on `triton_backend.py` / `decode_attention.py` | Step 3c's merge differs from the one here — apply that hunk by hand, it is small |
| `Reversed (or previously applied) patch detected` | Already applied. Answer `n` |

Add `--dry-run` first if you want to see the verdict without touching
anything.

#### 3f. Rotation tensors

OSCAR needs a rotation checkpoint for K and for V. Uncalibrated (Hadamard)
rotations are what is used here — see *Two results that went against
expectation* for why the calibrated ones were dropped — and they need no
calibration data, no GPU and no model weights:

```bash
python /path/to/escha_oscar/tools/make_hadamard_rotations.py \
    --config /path/to/Qwen3.6-35B-A3B-Escha-W2/config.json \
    --out-dir /path/to/oscar_rotations
# head_dim=256 num_layers=40
# wrote /path/to/oscar_rotations/k_rotation_hadamard_hd256.pt
# wrote /path/to/oscar_rotations/v_rotation_hadamard_hd256.pt

export SGLANG_OSCAR_K_ROTATION_PATH=/path/to/oscar_rotations/k_rotation_hadamard_hd256.pt
export SGLANG_OSCAR_V_ROTATION_PATH=/path/to/oscar_rotations/v_rotation_hadamard_hd256.pt
```

The script reads `head_dim` and `num_hidden_layers` out of `config.json` (they
live under `text_config` in these checkpoints — 256 and 40 for this MoE). Its
output is bit-identical to the rotation files in production here.

Two things that look wrong but are not:

- **One entry per layer, but only some layers use it.** This is a hybrid
  model: `config.json`'s `layer_types` marks only every fourth layer as
  `full_attention` (10 of 40 here), and only those have a KV cache to rotate.
  Lookups are by layer index, so generating all 40 is correct and generating
  more than needed is harmless — the file in production here carries 64.
- **Every layer gets the same matrix.** Uncalibrated rotations are
  layer-independent by construction. Calibrated ones differ per layer, which
  is the only visible difference between the two kinds of file.

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

If it does not start:

| Symptom | Cause |
|---|---|
| `ValueError: Oscar int2 KV cache requires both SGLANG_OSCAR_..._ROTATION_PATH` | Step 3f's exports are missing from *this* shell (or from the systemd unit, if you run it as a service) |
| CUDA OOM while loading weights | `--ep-size 2` missing — see *Settings you must not change* |
| An assertion inside flashinfer | `ATTN_BACKEND=triton` missing on an RTX 50-series card |
| `CUDA error: invalid configuration argument` | `--enable-mixed-chunk` is set. It cannot be used with the INT2 kernels |
| `K size` in the log is gigabytes | `--kv-cache-dtype int2` did not take effect — the integration is incomplete |
| Starts, then everything crawls at 100 % GPU and 40–50 W | The VRAM cliff. Lower `MEM` |

### 5. Check it

```bash
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"escha-qwen36-35b-a3b-w2",
       "messages":[{"role":"user","content":"What is 1+1?"}],
       "max_tokens":64}' | python -m json.tool
```

A reply only proves the server is up — it does not prove the INT2 path is the
one being used. Two things to confirm:

- the startup log line above says `K size: 0.21 GB, V size: 0.21 GB`. An
  unquantized pool at this context is an order of magnitude larger, so a
  gigabyte-scale figure here means `--kv-cache-dtype int2` silently fell back.
- the integration is loaded at all (run it with the step 3f exports set):

```bash
python - <<'EOF'
from sglang.srt.mem_cache.memory_pool import load_oscar_rotation_config
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_grouped_quant_int2,
)
print("OSCAR INT2 present:", load_oscar_rotation_config())
EOF
```

An `ImportError` here means step 3b/3c is incomplete. A `ValueError` naming
`SGLANG_OSCAR_K_ROTATION_PATH` means the integration is fine but step 3f's
exports are missing from the environment — the data-free Hadamard fallback was
not ported, so the server will refuse to start the same way.

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

## Doing the same on Qwen3.8-27B (the dense Escha-W2)

The dense sibling — `Qwen3.8-27B-Escha-W2`, served by the
`escha-runtime-qwen3dense` wheel — is also run here with OSCAR INT2 KV, at
196,608 tokens on the same two 12 GB cards. **The step 3 integration is the
same job on the same ten files**, so the mapping above transfers. What does
not transfer:

| | MoE (this document) | Dense Qwen3.8-27B |
|---|---|---|
| Wheel | `escha-1.0.2+qwen3moe` | `escha-1.2.1+qwen3dense` |
| Layers / head_dim | 40 / 256 | 64 / 256 |
| Full-attention layers | 10 (every 4th) | 16 (every 4th) |
| Heads per GPU at TP 2 | 8 q / 1 kv → `kv_group_num` 8 | 12 q / 2 kv → `kv_group_num` **6** |
| Rotations in production | uncalibrated (Hadamard) | **calibrated**, per layer |
| KV quant group size | `--kv-cache-quant-group-size 32` | `64` |
| Expert parallelism | `--ep-size 2` required | not applicable |
| `--enable-mixed-chunk` | crashes with the INT2 kernels | works, and is used |
| Context | 262,144 | 196,608 |

Three consequences worth knowing before you start:

- **The patches in `patches/` are cut against the MoE wheel.** Dry-run against
  the dense tree, three of the six do not apply (`triton_backend`,
  `schedule_policy`, and the second hunk of `scheduler`); `decode_attention`
  only applies with fuzz. The *changes* are the right ones for both — the
  context lines are not. Port them by hand and re-measure rather than forcing
  them.
- **`_safe_block_h` matters much more here.** At TP 2 the dense model has
  `kv_group_num = 6`, which is neither ≥ `BLOCK_H` nor a multiple of it — the
  exact shape that makes a head tile straddle a KV head. The MoE's group of 8
  is a power of two, so it stays safe whatever `BLOCK_H` is.
- **Do not copy the rotation choice across.** Uncalibrated rotations won the
  measurement on the MoE checkpoint; the dense deployment runs calibrated
  ones. Which wins is a property of the checkpoint, so generate the Hadamard
  pair with the tool above, and only invest in calibration if a test that can
  actually discriminate says it helps.

Otherwise the launch differs only in the flags: `--tp-size 2` without
`--ep-size`, `--kv-cache-quant-group-size 64`,
`--triton-attention-num-kv-splits 64` (the default of 8 caps the dynamic split
count and costs ~8 % on long-context decode), `--mamba-ssm-dtype float16`, and
`CTXLEN=196608 MEM=0.87 CHUNK=6144 MAXREQ=12 MAXMAMBA=40`.

## Licence

sglang is Apache-2.0 and so are these patches. The `escha` runtime and the
Escha-W2 weights are distributed separately by their authors and are **not**
redistributed here.
