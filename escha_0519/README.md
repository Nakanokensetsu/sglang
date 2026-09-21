# Escha-W2 dense (Qwen3.8-27B) + OSCAR INT2 KV + HiCache, on stock sglang 0.5.19

Two 12 GB consumer cards, a 32 GB host, WSL2. 131,072-token context, a
210,000-token KV pool, 8–10 concurrent users. This is the configuration that
actually runs, with the reasoning behind each number and the ways it fails.

The base branch [`escha-0519-base`](../../tree/escha-0519-base) is the
unmodified `v0.5.19` tag, so
[PR #1](https://github.com/Nakanokensetsu/sglang/pull/1) is exactly what this
adds — seven commits, separable.

The 0.5.15 MoE setup this started from is on
[`escha-oscar-int2`](../../tree/escha-oscar-int2) and is *not* superseded: it
serves a different model.

## At a glance

| | |
|---|---|
| Model | [Qwen3.8-27B-Escha-W2](https://huggingface.co/centraly/Qwen3.8-27B-Escha-W2) — 2-bit dense, hybrid GDN + full attention, 64 layers (16 full-attention, 48 mamba) |
| Runtime | [escha-runtime-qwen3dense](https://huggingface.co/EschaLabs/escha-runtime-qwen3dense) |
| GPUs | 2 × 12 GB, TP2. 10.1 GB idle, 11.1 GB at full prefill |
| Host | 32 GB. **This is the tight resource, not VRAM** — see *Host memory* |
| KV cache | OSCAR INT2 (packed 2-bit + per-group scales/zeros) |
| L2 cache | HiCache in host RAM, `--hicache-ratio 1` (5.2 GB pinned) |

## Measured

Same build, HiCache on vs off:

| | off | on |
|---|---|---|
| cold prefill (75K tokens) | 1010 tok/s | 964 tok/s |
| decode, bs1 | 51.5 tok/s | 50.7 tok/s |
| short-prompt TTFT | 0.132–0.153 s | 0.123–0.153 s |
| TTFT during a long prefill | 0.140 s | 0.141 s |
| 8 concurrent, distinct 18K prefixes, **second pass** | 5.2 s | **1.6 s** |
| Test D (96K aggregation + ordered enumeration, 20 questions) | 18/20 | 18/20, identical per question |

Separately, widening the KV pool (see *The pool, not the mamba cache*) moved
three users holding 66K contexts each from **12.4 % cache hit / 57 s** to
**99.9 % / 0.3 s** on revisit.

## Requirements

* 2 × 12 GB CUDA GPUs. One card is not enough for this context length.
* 32 GB host RAM is workable but has no slack. See *Host memory*.
* CUDA toolkit on `PATH` — the HiCache and mamba transfer kernels are JIT
  compiled with `nvcc` at first use.
* The INT2 KV cache needs two rotation matrices. Generate Hadamard ones with
  [`escha_oscar/tools/make_hadamard_rotations.py`](../escha_oscar/tools/make_hadamard_rotations.py);
  `head_dim` here is 256.
  Calibrated rotations scored *worse* than Hadamard in our tests (37/41 vs
  40/41), so do not assume calibration is an upgrade.

## Install

Either check out this branch and run from source, or keep a stock `v0.5.19`
install and put this tree in front of it:

```bash
git clone -b escha-0519 https://github.com/Nakanokensetsu/sglang.git
export PYTHONPATH=/path/to/sglang/python   # takes precedence over the wheel
```

The overlay form is what we run: the stock wheel stays untouched and reverting
is one `export` away. It also means **deleting the tree stops the server from
starting**, so treat it as part of the deployment, not as scratch space.

## Launch

```bash
export SGLANG_MAMBA_CONV_DTYPE=float16   # must match --dtype, or causal_conv1d dies
export SGLANG_DISABLE_CUDNN_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_INT8_LM_HEAD=0
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
export SGLANG_OSCAR_K_ROTATION_PATH=/path/to/k_rotation_hadamard_hd256.pt
export SGLANG_OSCAR_V_ROTATION_PATH=/path/to/v_rotation_hadamard_hd256.pt

python -m sglang.launch_server \
  --model-path /path/to/Qwen3.8-27B-Escha-W2 \
  --host 0.0.0.0 --port 8081 \
  --dtype float16 --tp-size 2 \
  --attention-backend triton \
  --mamba-ssm-dtype float16 \
  --kv-cache-dtype int2 --kv-cache-quant-group-size 64 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --context-length 131072 \
  --chunked-prefill-size 8192 \
  --max-mamba-cache-size 32 \
  --disable-prefill-cuda-graph \
  --max-running-requests 10 \
  --max-total-tokens 210000 \
  --mem-fraction-static 0.85 \
  --enable-mixed-chunk \
  --enable-hierarchical-cache --hicache-ratio 1 \
  --hicache-io-backend kernel --hicache-mem-layout page_first \
  --triton-attention-num-kv-splits 64 \
  --enable-metrics --enable-cache-report \
  --allow-auto-truncate --trust-remote-code
```

## Why these numbers

### The pool, not the mamba cache

The limit on how many long sessions stay cached is the **KV pool**, not the
mamba state cache. Three flags move together and one without the others does
nothing or makes it worse:

```
--mem-fraction-static 0.85    the static budget; this is what actually frees the space
--max-total-tokens 210000     a cap. On its own it changes nothing, the budget still binds
--chunked-prefill-size 8192   halves the mamba slots a prefill needs
```

A hybrid SSM model needs roughly `KV pool ÷ chunked-prefill-size` mamba slots
for prefix reuse to hold (upstream issue #36935). At 210,000 / 8,192 = 26 that
fits in `--max-mamba-cache-size 32`. Widen the pool *without* raising the chunk
size and you get 210,000 / 4,096 = 51 > 32, the condition breaks and reuse
collapses — the pool gets bigger and the cache gets worse.

### `--disable-prefill-cuda-graph`

0.5.19's prefill CUDA graph captures 51 shapes and holds 1.81 GB. INT2 prefill
dequantizes the prefix, so it needs transient memory proportional to sequence
length (~962 MB at 112K). The two together fall off the VRAM cliff. Turning the
graph off took 96K prefill from 202 to 959 tok/s. The decode graph is kept —
it only captures bs=[1,2,4,8].

### `--mamba-radix-cache-strategy extra_buffer_lazy`

Drops the mamba slots one request holds from 5 to 4 (3 with
`SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`). `max_running_requests` is floored to
slots ÷ per-request, so 32 slots gives 8–10 concurrent instead of 6.

### `--context-length 131072`

The largest value that still runs at full speed here. 104K/112K/120K all hold
888–922 tok/s; 130K falls to 450 tok/s. Setting a limit above the usable one
just lets users walk off the cliff without knowing.

## Host memory

HiCache's L2 tier lives in **host RAM**, not VRAM, and it is **pinned** — the
OS cannot page it out. On this machine:

| `--hicache-ratio` | pinned (2 ranks) | MemAvailable after start | effective page cache |
|---|---|---|---|
| 1 | 5.2 GB (KV 1.29 + mamba 1.25, per rank) | 10.2 GB | 2.0 GB |
| 2 | 10.3 GB (KV 2.58 + mamba 2.55, per rank) | 3.7 GB | **589 MB** |

At ratio 2 the machine swaps continuously while idle and everything on it gets
slower — the pinned pages cannot move, so everything else is evicted instead.
**Ratio 1 is the practical limit on a 32 GB host.**

Ratio 2 also does not start at all with the stock 10 GiB reserve, because the
host pools are sized *while the weights are still loading*, which is the low
point of `MemAvailable` (we measured 565 MB). `SGLANG_HICACHE_HOST_RESERVE_GB`
exists for that, but lowering it to make ratio 2 fit is a bad trade here.

Note the mamba host pool is expensive per slot: 37.7 MB, so 33 slots cost about
as much as 210,001 tokens of INT2 KV.

## Things that will bite you

**`--hicache-mem-layout layer_first` does not work.** `MambaPoolHost` accepts
only `page_first` / `page_first_direct` and asserts on anything else. Use
`page_first`.

**HiCache on WSL2 needs `cudaHostAlloc`.** The mamba backup kernel stores to
the host pool through its host pointer. Memory registered with
`cudaHostRegister` is not device-mappable under WSL2 — with any flag — and the
kernel faults with `illegal memory access`. This branch detects WSL and uses
`cudaHostAlloc` instead; `SGLANG_HICACHE_HOST_ALLOC=pin|register` overrides it.
Reproduce in 30 seconds with
[`tools/mamba_transfer_repro.py`](tools/mamba_transfer_repro.py):

```
dst on device         -> ok
dst torch pin_memory  -> ok                    (cudaHostAlloc)
dst cudaHostRegister  -> illegal memory access (flags 0/1/2/3)
dst pageable          -> illegal memory access
```

This is why HiCache looked like it "silently breaks hybrid GDN models" here. It
is unrelated to upstream #39830.

**L3 storage backends do not work with INT2.** `--hicache-storage-backend`
writes a flat data page that carries the packed codes without their
scales/zeros, so a restored slot is read with the previous occupant's
quantization parameters and corrupts without any error. The host pool raises
`NotImplementedError` rather than let that happen.

**MTP draft pools are likewise refused** — the scales/zeros transfer has no
packed-draft path.

**A VRAM cliff presents as slowness, not as an allocation failure.** If
throughput drops by a third and power draw falls while utilisation stays at
100 %, you are over the edge, not under load.

## Checking it

| tool | answers |
|---|---|
| [`tools/int2_hicache_geometry_test.py`](tools/int2_hicache_geometry_test.py) | do the INT2 host pool's three widths agree? **No GPU needed.** Includes negative tests that re-introduce the original bugs |
| [`tools/mamba_transfer_repro.py`](tools/mamba_transfer_repro.py) | is `cudaHostRegister`ed memory device-mappable on this machine? |
| [`tools/hicache_correctness.py`](tools/hicache_correctness.py) | does a prefix restored from the host tier still give the right answer? Checks against values known from the seed, not against itself |
| [`tools/ab/speed.py`](tools/ab/speed.py), [`tools/ab/conc.py`](tools/ab/conc.py) | the four speed numbers; concurrency N with VRAM/power/temperature sampling |
| [`tools/ab/load_multi.py`](tools/ab/load_multi.py) | how many distinct long chains stay cached — the single most useful measurement here |
| [`escha_oscar/tools/check_integration.sh`](../escha_oscar/tools/check_integration.sh) | is the INT2 integration complete? |

The startup log should contain, per rank:

```
Int2 HiCache host pool geometry verified: 256 B/token/layer packed (K+V), ...
HiCache host pools use cudaHostAlloc (pin_memory) instead of cudaHostRegister
Allocating kv hierarchical KV host pool: 210001 tokens, 1.29 GB host memory.
max_total_num_tokens=210000 ... context_len=131072
```

## Note on the source comments

Comments in the code written for this fork are in Japanese.

## Licence

Apache 2.0, same as upstream SGLang.
