# Long-context prefix cache on a hybrid GDN model: the limit is the KV pool, not the mamba cache

**A 27B hybrid Gated-DeltaNet model on 2× 12 GB consumer cards. Three users with
66 K-token contexts went from a 57-second cold prefill on every turn to 0.3 s,
by moving three flags. No VRAM was added that was not already sitting idle.**

| | before | after |
|---|---|---|
| KV pool (`max_total_num_tokens`) | 148,058 | **210,000** |
| 66 K × 3 chains, revisit | 12.4 % / 57 s | **99.9 % / 0.3 s** |
| cold prefill (75,150 tok) | 970 tok/s | **1,010 tok/s** |
| decode, bs=1 | 51.9 tok/s | 51.9 tok/s |
| short-prompt TTFT during a long prefill | 0.135 s | 0.123 s |
| concurrency 10 | 0 errors, 169 W, 70 °C | 0 errors, 175 W, 78 °C |
| discriminative long-context test (96 K) | 17/20 | 18/20 |

Not one metric got worse.

## The three flags, and why all three are needed

```
--mem-fraction-static 0.78 -> 0.85     static budget +856 MiB -> KV pool 148,058 -> 210,000
--max-total-tokens  150000 -> 210000   lifts the cap. ON ITS OWN THIS DOES NOTHING
--chunked-prefill-size 4096 -> 8192    halves mamba checkpoint demand (210,000/8,192 = 26 <= 32)
```

**`--max-total-tokens` is a cap, not a size.** Setting it to 210000 while
`--mem-fraction-static` stayed at 0.78 left `max_total_num_tokens` at 148,058,
unchanged. The pool is derived from the static budget; the flag only stops it
growing past a number you choose. The lever that actually moves it is
`--mem-fraction-static`.

**`--chunked-prefill-size` is load-bearing here.** A hybrid SSM model donates one
mamba checkpoint per chunked-prefill boundary, so a chain needs roughly
`pool_tokens / chunked_prefill_size` checkpoint slots. Widening the pool to
210,000 while leaving the chunk at 4096 needs `210,000/4,096 = 51` slots against
a pool of 32 — over the line, where prefix reuse for hybrid models collapses.
Doubling the chunk brings it to 26 and the two changes fit together. Raising the
pool alone would have made things worse.

## How the diagnosis went wrong first, and what fixed it

The symptom was: three separate 66 K chains, asked in turn, three laps. Lap 2
should be a cache hit. It was 0 %.

The first hypothesis was upstream
[#36935](https://github.com/sgl-project/sglang/issues/36935) — the mamba LRU
evicting a path's states shallow-first, which is the inverse of what a branch
match needs. `mamba_evictable` sat at 24 of 32, which looked like pressure.

Three mamba-side levers were tried. **None of them moved the number at all:**

| lever | lap 2/3 hit |
|---|---|
| baseline (chunk 4096) | 0.0 % |
| chunk 4096 → 8192 | 12.4 % |
| + PR #38000 (thin cached states by coverage instead of the LRU tail) | 12.4 % — *no change* |
| + int8 mamba checkpoint pool, 32 slots (≈2× capacity) | 12.4 % — *no change* |
| **3 chains → 2 chains** | **99.9 %** |

That last row is the whole answer. 3 × 66,000 = 198,000 tokens against a KV pool
of 148,058. **The pool could not hold three chains.** `mamba_evictable = 24` was
simply three chains' worth of states sitting there, not evidence of pressure.

A depth-mapping probe confirmed the mamba side was healthy: checkpoints existed
at every depth, and the match walk picks the *deepest* valid node rather than
stopping at the first gap (`_update_best_if_valid` in `unified_tree_core.py`
never breaks the walk). Re-touching the same chain progressively returned it to
100 % at 2.0 s.

**Lesson: measure the cheap structural bound before reaching for the subtle
mechanism.** `sum(chain_lengths) vs max_total_num_tokens` is one subtraction.

## "It started" is not "it works"

Two configurations started cleanly and then died under a long prefill:

```
Triton kernel 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64' device-loaded
  after serving started (free device mem: 0.00 GiB).
  Pre-load it during engine init to avoid CUDA OOM.
RuntimeError: CUDA driver error: device not ready
```

Triton kernels JIT onto the device on first use, after the startup VRAM numbers
are printed. `--int8-mamba-ckpt-size 64` fit at boot (11,329 / 12,227 MiB) and
then had nothing left for that. **Startup validation has to include one long
prefill.**

Measured slot costs on this model (48 linear-attention layers, 16 full-attention
layers, TP 2): **fp16 mamba slot ≈ 38 MB, int8 checkpoint slot ≈ 23 MB** — 60 %
of fp16, not 50 %.

Upstream [#36266](https://github.com/sgl-project/sglang/pull/36266) pre-warms the
mamba COW kernel before serving; on a VRAM-thin box that is worth having.

## Checking for the VRAM cliff

The cliff on this hardware does not present as an allocation error. It presents
as the GPU sitting at 100 % utilisation while power and temperature *fall* —
work is not actually flowing. Healthy full load here is ~97-99 % at 169-175 W and
70-78 °C; the cliff looks like 100 % at 45 W and 48 °C.

At concurrency 10 the new configuration drew **175 W at 78 °C with zero errors**,
against 169 W / 70 °C before. Power went *up*: more real work, not less.

## Launch flags

The full script, with local paths replaced by `${...}` placeholders:

[`escha_run_0519.sh`](escha_run_0519.sh)

The flags that matter for this result:

```
--kv-cache-dtype int2 --kv-cache-quant-group-size 64
--mamba-radix-cache-strategy extra_buffer_lazy
--attention-backend triton
--context-length 114688
--chunked-prefill-size 8192          # was 4096
--max-mamba-cache-size 32
--max-total-tokens 210000            # was 150000
--mem-fraction-static 0.85           # was 0.78
--max-running-requests 10
--disable-prefill-cuda-graph
--enable-mixed-chunk
--enable-metrics --enable-cache-report
```

`--disable-prefill-cuda-graph` and `extra_buffer_lazy` are carried over from the
earlier 0.5.19 work in [`../escha_oscar_0519/README.md`](../escha_oscar_0519/README.md).

## Measurement scripts

In [`tools/`](tools/):

| script | what it answers |
|---|---|
| `load_multi.py` | **K distinct long chains, retention across laps. Varying K is the single most informative knob** |
| `depth_probe.py` | which depths still hold a checkpoint, by slicing the prefix |
| `speed.py` | cold prefill tok/s, short TTFT, decode tok/s, short TTFT *during* a long prefill |
| `conc.py` | concurrency N with per-3-second VRAM / utilisation / power / temperature sampling |
| `probe.py` + `compare.py` | byte-identical A/B of two builds on fixed prompts |

Two measurement traps cost real time here, both now guarded in the scripts:

1. **`--allow-auto-truncate` silently discards the tail.** A 114 K-limit server
   given a 180 K prompt answers from a truncated document. An early run showed
   "99.9 % cache hit" that was really every request having been truncated to the
   same prefix. `load_long.py` now asserts the prompt stayed under the limit.
2. **Comparing a warm server against a cold one.** The first A/B showed a
   difference that vanished once both sides were measured from a cold start.

## Rebasing this onto a moving upstream

This is a diff held against a target that moves weekly. v0.5.20 landed the day
before these measurements. Four things decide how much a rebase costs.

**1. Check that the class you are patching is actually constructed.**
Upstream PR #32129 patches `RadixCache`. On 0.5.19 `registry.py` routes every
model — hybrid SWA, hybrid SSM and plain — to `UnifiedRadixCache`;
`RadixCache(params)` is only reached from `create_simulated()`, and
`MambaRadixCache` from tests. Patching either runs nothing. Upstream
[#40313](https://github.com/sgl-project/sglang/pull/40313) deletes both. One
`grep -rn "ClassName(" --include=*.py` and a read of the factory would have
caught this before the work, not after.

**2. A three-way merge reporting "no conflict" is not a passing test.**
Hunks that apply cleanly can still reference APIs the new version removed —
`req.req_pool_idx`, `req.kv_committed_len`, `req.fill_ids` all moved onto
`req.kv` and merged silently. Two such landed here. So did an import of a symbol
the fork never had, and an import that was missing entirely.

**3. Import success is not name resolution.** A name used inside a function but
never imported raises `NameError` only when that function runs. One of those
sat through a clean `python -c "import ..."` of all 19 touched modules and only
surfaced at engine start, inside `_build_token_to_kv_pool_allocator`. The
`tools/undef.py` in this repo is a small pyflakes-shaped AST pass; run it over
every changed file.

**4. Measure the churn in the functions you retargeted, not the whole file.**
Between v0.5.19 and v0.5.20, in the files this port touches:

| file | v0.5.19 → v0.5.20 |
|---|---|
| `layers/attention/triton_backend.py` | +403 / −101 |
| `model_executor/pool_configurator.py` | +374 / −75 |
| `arg_groups/overrides.py` | +270 / −212 |
| `mem_cache/base_prefix_cache.py` | +127 / −33 |
| `mem_cache/radix_cache.py` | +73 / −28 |
| `mem_cache/common.py` | +63 / −37 |
| `mem_cache/chunk_cache.py` | +3 / −3 |
| `mem_cache/memory_pool.py` | +1017 / −324 |

Large, but every API this port retargeted against — `free_kv_row`,
`free_kv_row_segments`, `free_segment`, `check_decode_capacity`,
`_page_size_default` — still exists in v0.5.20. **The retargeting decisions
carry; only the merge has to be redone.** Re-deriving them would be the
expensive part, and that part is reusable.

One thing to watch: [#39627](https://github.com/sgl-project/sglang/pull/39627)
makes the Rust tree core the default. `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND`
is `"python"` on 0.5.19, so Python-side patches to
`unified_cache/components/mamba_component.py` take effect. If that PR lands,
pin the env var explicitly or the patches are silently bypassed.

## Upstream fixes carried here

| PR | effect here |
|---|---|
| [#38191](https://github.com/sgl-project/sglang/pull/38191) | skip the empty-key insert in `cache_finished_req`. **Observed firing on real traffic** |
| [#37943](https://github.com/sgl-project/sglang/pull/37943) | `extra_buffer_lazy` only: skip a checkpoint instead of asserting when no slot can be donated |
| [#39526](https://github.com/sgl-project/sglang/pull/39526) | keep extend-row mamba tracking across `mix_with_running`, fixing [#39342](https://github.com/sgl-project/sglang/issues/39342) |
| [#38000](https://github.com/sgl-project/sglang/pull/38000) | included; **measured no effect on this workload** — its premise is shallow-first eviction, which is not what limits this box |

On this build the live prefix cache is `UnifiedRadixCache` with
`ComponentType.MAMBA`; `RadixCache` and `MambaRadixCache` are never constructed
(see [#40313](https://github.com/sgl-project/sglang/pull/40313)). A port that
patches the legacy classes runs nothing.
