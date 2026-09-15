# Escha-W2 (2-bit MoE) + OSCAR INT2 KV — production patches

Running **Qwen3.6-35B-A3B at 2-bit weights** (the `escha` runtime) **together with
OSCAR INT2 KV cache** (`--kv-cache-dtype int2`, sglang PR #32129) on **2× 12 GB
consumer GPUs**.

The two do not ship together: the `escha` wheel provides the 2-bit MoE weight
loading and CUDA kernels, and OSCAR INT2 KV lives in an open sglang PR. This
directory holds the patches needed to make the combination work well, plus the
measurements that justify each one.

All numbers below were measured on the configuration in *Environment*, not
estimated.

## What is here

| Patch | What it fixes | Measured effect |
|---|---|---|
| `..._triton_backend.py` + `..._kv_quant_kernels.py` + `..._quantized_kv_prefill.py` | INT2 prefill holds the expanded prefix **three times** (`3P+2E`). Preallocate the combined buffer and dequantize directly into its slices (`out=`) → `P+E` | **−746 MB (−37 %)** transient at 192 K. This is what makes 192 K context and 8 concurrent streams fit at the same time |
| `..._scheduler.py` + `..._schedule_policy.py` | A long prefill starves streams that are already decoding: the batch duration becomes a gap between their tokens | Max inter-token gap **5.1 s × 21 occurrences → 0.14 s**. Enabled with `SGLANG_CHUNK_DECODE_CAP=1536`; default `0` is byte-identical to stock behaviour |
| `..._decode_attention.py` | `_safe_block_h`: a grouped-decode head tile that straddles a KV head makes query heads read **another KV head's cache**. Silent — nothing asserts, no NaN, only the benchmark score moves | See *The silent one* below |

## The silent one

`_fwd_grouped_kernel_stage1` addresses heads as

```
VALID_BLOCK_H = min(BLOCK_H, kv_group_num)
cur_head      = cur_head_id * VALID_BLOCK_H + arange(BLOCK_H)
cur_kv_head   = cur_head_id // cdiv(kv_group_num, BLOCK_H)
```

`cur_head` is a flat query-head index while `cur_kv_head` comes from the block
index. They only agree when a head block lies wholly inside one KV group, i.e.
`BLOCK_H >= kv_group_num` **or** `kv_group_num % BLOCK_H == 0`.

Stock sglang hardcodes `BLOCK_H = 16`, which satisfies this for every
power-of-two `kv_group_num` — that is why it has never bitten upstream. It is
still reachable there with `kv_group_num > 16` that is not a multiple of 16
(e.g. 96 query heads over 4 KV heads), and it bites immediately once `BLOCK_H`
is tuned or made batch-size dependent: `BLOCK_H=4` against `kv_group_num=6`
makes head block 1 cover query heads 4..7 while reporting `cur_kv_head=0`, so
query heads 6 and 7 silently attend to KV head 0.

`_safe_block_h` rounds up to a power of two so `VALID_BLOCK_H == kv_group_num`,
i.e. exactly one block per KV head.

## Applying

The patches target the `sglang` tree **bundled inside the `escha` wheel**
(`escha-1.0.2+qwen3moe`), which is based on an older sglang layout than this
branch — file paths and surrounding code differ, so they will not `git apply`
here unchanged. Treat them as documented deltas, not drop-in commits.

```bash
SP=$(python -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent.parent)")
cd "$SP"
patch -p0 --backup --suffix=.orig < patches/sglang_srt_mem_cache_kv_quant_kernels.py.patch
# ... and so on
```

> Patching `site-packages` is lost on any wheel reinstall. Keep the `.orig`
> backups, or put the modified files on `PYTHONPATH` as an overlay instead of
> editing in place.

## Serving

```bash
export MEM=0.88 CTXLEN=262144 MAXMAMBA=64 MAXREQ=16 RADIX=1
export MAMBA_SCHED=extra_buffer ATTN_BACKEND=triton
export GRAPHS=1 CUDA_GRAPH_BS="1 2 4 8 16"
export SGLANG_CHUNK_DECODE_CAP=1536     # needs the scheduler patch
export SGLANG_CHUNK_FAIR_FLOOR=256
export ENABLE_CLP=1                     # required for the thinking budget
```

Key server flags:

```
--tp-size 2 --ep-size 2
--kv-cache-dtype int2 --kv-cache-quant-group-size 32
--context-length 262144 --chunked-prefill-size 2048
--max-running-requests 16 --max-mamba-cache-size 64
--attention-backend triton --mamba-scheduler-strategy extra_buffer
--enable-custom-logit-processor
--disable-piecewise-cuda-graph --cuda-graph-bs "1 2 4 8 16"
```

Resulting KV pool: **350,257 tokens**, K and V 0.21 GB each, ~1.0 GB VRAM left.

> `--max-running-requests 16` is silently clamped to **12** by the engine given
> the mamba cache size and pool. Size concurrency against 12, not 16.

## Measured

| | |
|---|---|
| 180 K cold TTFT | 46.2 s |
| 180 K resend (prefix cache hit) | 1.87 s |
| Decode, single stream | 87.5 → 56.8 tok/s as context grows |
| Concurrency 8, 400-token generations | 295 tok/s aggregate |
| Short request during a long prefill | delayed 1.10×, p50 0.29 s |
| Quality vs the dense 27B it replaced | long-context discrimination 40/41 vs 38/41 |

### Calibrated rotations made it worse

OSCAR's paper reports a large gain from calibrated rotations (−66.29 pt without
calibration on Qwen3-32B). On this checkpoint the opposite held: **40/41
uncalibrated vs 37/41 calibrated**, with third-task recall dropping 0.969 →
0.819. Hadamard rotations are used instead. Calibrate-then-measure; do not
assume.

### The VRAM cliff

Above **11,830 MiB** the GPU reports 100 % utilisation while drawing only
40–50 W, and the watchdog eventually fires. Under WSL this does not surface as
an allocation failure — it looks like the model simply became slow. Budget
against the cliff, not against total VRAM.

## Environment

| | |
|---|---|
| GPU | 2× 12 GB consumer Blackwell (sm_120) |
| OS | WSL2 / Ubuntu |
| Model | Qwen3.6-35B-A3B, 2-bit MoE (256 experts) |
| Runtime | `escha-1.0.2+qwen3moe` |
| KV cache | OSCAR INT2, group size 32 |
| Context | 262,144 |

## Licence

sglang is Apache-2.0 and so are these patches. The `escha` runtime is
distributed separately by its authors and is **not** redistributed here; install
it from its own source.
