# Escha-W2 (2-bit) + OSCAR INT2 KV on **stock sglang 0.5.19**

**A 27B dense hybrid-SSM model at 131K context, 8 concurrent users, on two 12 GB
consumer GPUs — running on unmodified upstream sglang 0.5.19.**

Companion to [`escha_oscar/`](../escha_oscar), which covers the same stack on the
sglang fork bundled inside the `escha` wheel (0.5.15). This directory is the
**stock-0.5.19 route**: it gets you the newer mamba/radix features that the
bundled fork does not have.

Measured 2026-09-18 on **Qwen3.8-27B-Escha-W2** (2-bit weights), 2x RTX 5070 12 GB
(sm_120), WSL2, TP2. Every number below is measured, not estimated.

---

## The headline: `--disable-prefill-cuda-graph`

sglang 0.5.19 captures a **prefill CUDA graph** by default — **51 token shapes**
(4096 down to 4) costing **1.81 GB** of VRAM.

INT2 KV prefill dequantizes the cached prefix into a dense tensor, so it needs
transient memory **proportional to sequence length** (~873 MB at 96K). When the
graph has already taken 1.81 GB, the two collide and you fall off the VRAM cliff.

| Prompt | graph on | **graph off** | speedup |
|---|---|---|---|
| 10K | 907 tok/s | **968** | 1.07x |
| 32K | 955 | **994** | 1.04x |
| 64K | 918 | **985** | 1.07x |
| **96K** | **202** | **893** | **4.4x** |
| free VRAM | 0.54 GB | **2.35 GB** | +1.81 GB |

With `--chunked-prefill-size 4096`, essentially the only shape ever used is 4096,
so the graph buys **almost nothing and costs 1.81 GB**. Disabling it makes short
prompts 4-7% faster too. Keep the *decode* graph — it only captures `bs=[1,2,4]`
and wastes nothing.

> sglang does have auto-disable rules for the prefill graph, but all of them are
> model-correctness driven (`--enable-unified-memory`, Inkling, Muse-Glimmer).
> **There is no branch anywhere that looks at free VRAM** (verified in the 0.5.19
> source). Low-VRAM + long-context + quantized KV appears to be outside the
> tested envelope.

## Second: `--mamba-radix-cache-strategy extra_buffer_lazy`

On hybrid SSM (mamba) models, **the number of mamba state slots one running
request consumes is decided by the radix strategy**:

| strategy | slots / request |
|---|---|
| `extra_buffer` | 5 |
| **`extra_buffer_lazy`** | **4** |
| `no_buffer` | 3 (incompatible with overlap schedule) |
| radix disabled | 1 |

`max_running_requests` is effectively clamped to
`--max-mamba-cache-size / slots-per-request`. Measured with everything else held
identical (same VRAM headroom, same chunk size, same pool):

| | default | `extra_buffer_lazy` |
|---|---|---|
| effective max concurrency | 3 | **4** |
| aggregate @ 4 clients | 83 tok/s | **122** (+47%) |
| aggregate @ 8 clients | 106 tok/s | **140** (+32%) |
| bs1 | 52.6 | 52.1 (unchanged) |

---

## Measured results (final configuration)

| | TTFT | aggregate |
|---|---|---|
| bs1 | 0.12 s | 52.4 tok/s |
| cold (~10K doc) | 8.25 s | — |
| warm (prefix reuse) | **0.21 s** | — |
| 2 clients | 0.25 s | 86 tok/s |
| 4 clients | 0.30 s | 127 tok/s |
| 8 clients | **0.48 s** | **210 tok/s** |

Prefill throughput: 10K **978** / 32K **1020** / 64K **1005** / 96K **959** tok/s.

**Quality**: on a discriminative long-context test (two-field extraction among
many near-identical distractors, multi-position aggregation, ordered
enumeration) this build scores **19/20** — **identical** to the same test on the
0.5.15 production stack. A needle-in-a-haystack test is useless here; it
saturates at 15/15 on every configuration.

## How to recognise the VRAM cliff

GPU utilisation alone cannot tell you. **Look at power draw and temperature.**

| | utilisation | power | temp |
|---|---|---|---|
| full load | 97-99% | 150-204 W | 58-73 C |
| **cliff** | **100%** | **40-57 W** | **39-45 C** |

On the cliff the GPU reports "100% busy" while it is actually stalling in memory
allocation. Under WSL this is especially slow.

## Pool, concurrency and long context share one budget

- `--max-total-tokens` only acts as an **upper bound**. Once mamba slots are
  allocated, the computed pool ceiling arrives first — asking for more changes
  nothing (250,000 requested still gave 140,378).
- To actually grow the pool you must either raise `--mem-fraction-static`
  (which eats the transient headroom long-context prefill needs → cliff) or cut
  mamba slots (which costs concurrency).

---

## Launch command

```bash
python -m sglang.launch_server \
  --model-path <path to Qwen3.8-27B-Escha-W2> \
  --dtype float16 --tp-size 2 \
  --attention-backend triton \
  --mamba-ssm-dtype float16 \
  --kv-cache-dtype int2 --kv-cache-quant-group-size 64 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --context-length 131072 \
  --chunked-prefill-size 4096 \
  --max-mamba-cache-size 32 \
  --disable-prefill-cuda-graph \
  --max-running-requests 8 \
  --max-total-tokens 150000 \
  --mem-fraction-static 0.78 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code
```

Required environment:

```bash
# conv_state defaults to bfloat16; escha runs fp16 and causal_conv1d will fail
# with a dtype mismatch unless you pin this.
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_OSCAR_K_ROTATION_PATH=<k_rotation_*.pt>
export SGLANG_OSCAR_V_ROTATION_PATH=<v_rotation_*.pt>
```

Rotations come from OSCAR's RotationZoo (`Zhongzhu/OSCAR-RotationZoo`) or your own
calibration. **This port requires a rotation**; the data-free Hadamard fallback
was deliberately not carried over.

## Install order (the order is the whole trick)

```bash
pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu129
pip install <sglang_kernel cu129 wheel>   # the PyPI build is CUDA-13/sm100 only and will not run on sm_120
pip install "sglang==0.5.19"
pip install <escha runtime wheel>          # MUST be last
```

The `escha` wheel ships its own sglang fork inside. **Install it first and then
add stock sglang on top, and the escha quantization methods vanish from the
registry** — the files are present but never registered, and the model fails to
load with a confusing quantization error.

`escha` 1.2.2 builds against the PyTorch **stable ABI**, which is what makes this
route possible at all: it runs on torch 2.13 even though the model card still
pins torch 2.9.

## Patches

`patches/` holds 8 unified diffs against **stock sglang 0.5.19**, plus one
hand-merge file.

| file | what it does |
|---|---|
| `01-memory_pool.patch` | INT2 pool, OSCAR rotations, scales/zeros buffers, write path |
| `02-triton_backend.patch` | INT2 prefill/decode dispatch; explicitly refuses DCP |
| `03-decode_attention.patch` | INT2 decode kernels (byte-identical to OSCAR upstream) |
| `04-pool_configurator.patch` | INT2 cell size (2 bits = 0.25 B) |
| `05-kv_cache_dtype.patch` | `int2` dtype resolution |
| `06-environ.patch` | `SGLANG_OSCAR_*` environment variables |
| `07-model_config.patch` | quantization method registration |
| `08-escha_quant.patch` | escha quantization shard handling |
| `09-manual-merge.txt` | the three files with no clean baseline, with 3 lines of context |

All eight apply with **`patch -p0`** against an unmodified 0.5.19 tree; this was
verified by applying them to a pristine copy and syntax-checking the result.
Still, run each one with `--dry-run` first — a previous release of this
repository shipped patches that did not apply, and that is an easy mistake to
repeat.

### Two bugs worth knowing about

- **INT2 + decode context parallelism silently returns wrong results.** The
  ported INT2 decode kernels lack the `output_lse` path needed to merge partial
  attention across DCP ranks, and the INT2 branch returns before the DCP branch.
  `02-triton_backend.patch` turns this into an explicit `NotImplementedError`.
- **Watch for duplicate class definitions when porting.** Pasting a block from
  0.5.15 into 0.5.19's `memory_pool.py` brought a second `class KVCache` along
  with it. Python takes the later definition, so `MHATokenToKVPool` silently
  inherited the *old* base class and lost attributes that only exist in 0.5.19.
  The symptom was a bare `AttributeError`; `grep -c "^class KVCache"` returning
  2 was what actually found it.

## License / redistribution

sglang is Apache-2.0. **The escha runtime itself is not redistributed here.**
This directory contains only diffs against stock sglang 0.5.19, measured numbers,
and instructions.
