# ARCHITECTURE

llama.cpp internals that the expert-offload work depends on. Read with
`src/llama-moe-cache.cpp` and `src/llama-moe-cache.h` open; line references are
against the current branch HEAD and will drift.

## The configuration this project is about

```
-ngl 99 -ncmoe 40 --moe-cache-mib N
```

All 40 MoE layers' experts stay in host memory (`-ncmoe 40`); dense weights and
attention run on the GPU (`-ngl 99`). On 8 GiB this is the only configuration a
20.81 GiB model fits in, because 18.32 GiB of it is expert weights. The cache
then buys back a share of the expert traffic with `N` MiB of VRAM.

## The upstream expert cache

Already in master. `llama_moe_cache` is constructed per context from the model,
the context's backends and one total `size` in bytes.

### Objects

| | |
|---|---|
| `device` | one GPU: its backend, buffer type, the host bytes of the layers it caches, and its `split` share of the total budget |
| `group` | layers **on the same device with the same expert tensor layout**. Owns the device-side banks, the slot count, and one `moe_cache_slru` |
| `layer` | its group index, a host-memory `slot_map` (`I32 [1, n_expert]`), and the host expert tensors in bank order |
| `binding` | a view of a bank used in place of the host expert tensors in the graph |

### How a hit happens

1. The graph asks for `get_slot_map(il, n_tokens, n_expert_used)`.
2. That returns the layer's slot map only if the layer is cached, `n_tokens` is
   nonzero and **`<= max_batch` (32)**, and the tokens' experts fit the group's
   slot count. Otherwise `nullptr`, and the graph takes the host path.
3. The scheduler copies the slot map host-to-device. That copy is intercepted by
   `llama_moe_cache::copy`, which is registered as the `ggml_backend_sched` copy
   callback.
4. Inside the callback the cache reads the selected expert ids, resolves each to a
   slot, and **uploads only the experts that are missing**, then writes the
   resolved slot map to the device.

So the upload is a side effect of the slot-map copy, and the cache's unit of work
is one (layer, ubatch) slot-map copy. The telemetry counters and the trace are
keyed on exactly that unit: 40 layers x 255 tokens = 10202 copies for a 256-token
greedy decode.

### Budget split, two levels

1. **Across devices**: by `tensor_split` if the user set one, otherwise by each
   device's reported **free** memory. If no device reports free memory, evenly
   among devices that hold host experts.
2. **Within a device, across groups**: proportional to each group's `host_bytes`,
   so every group caches the same *fraction* of its own experts. The slot count
   is then grown one slot at a time while `alloc_size(g, n+1) <= budget`.

A group that cannot afford even `n_expert_used` slots is **not cached at all**
(`n_slots = 0`) and logs a warning.

### The three cache groups of this model, and why they matter

The Unsloth `UD-Q4_K_XL` quantization gives the layers different expert tensor
types, so the 40 layers split into three groups:

| layers | MiB per expert |
|---:|---:|
| 36 | 1.812 |
| 3 | 1.945 |
| 1 | 2.195 |

Each group has its **own** slot budget and its **own** LRU. Two consequences,
both measured:

- A small total budget can starve a group to zero slots, which then takes the
  bypass path for those layers. This is why the 384 MiB configuration gains less
  throughput than its traffic saving would predict.
- "Slots per token" must be computed per group, not globally. The branch's
  crossover test is `g.n_slots < n_expert_token`, where `n_expert_token` is
  summed over that group's layers.

### The replacement policy

`moe_cache_slru`, one per group. A segmented LRU over slots: two lists
(probationary and protected), `is_protected` per slot, `n_protected` capped by
`max_protected`. `max_protected = 0` degenerates to plain LRU.

The invariant that constrains every policy change: **an expert the current
request needs cannot be evicted.** It is enforced through `seen` / `seen_gen`,
a generation-stamped array over experts, checked in `victim()`.

Upstream's `victim()` walks the probationary list then the protected list from
the LRU end and takes the first evictable slot.

## What the branch changes

Seven commits on top of `d854d98ba`; see EXPERIMENTS.md for the index and
`moe-bench/RESULTS.md` section 11 for upstream status.

### Sampled least-used victim (`3db731656`)

The research contribution. Inside `moe_cache_slru`:

- `n_sample` (default 16 when enabled, 0 = upstream behaviour), set at `init`
  **only when `g.n_slots < n_expert_token`** - the same condition upstream
  already warns about, so no new tuning constant is introduced.
- `use`, one `uint32_t` per (layer, expert), 40 KiB for this model, incremented
  when an expert is selected.
- `victim_sampled(il)` draws up to `n_sample` candidate slots with a small LCG
  (`rng*1664525 + 1013904223`), skipping slots holding an expert this request
  needs, returning a free slot immediately if it draws one, and otherwise the
  **least used** of the sample. It gives up after `4*n_sample` draws.
- `victim()` tries `victim_sampled()` first and **falls back to the upstream LRU
  walk** if it returns -1, so the eviction path can never fail.
- Aging: every `age_every = 8*n_slots` fills, all counters are halved
  (`u >>= 1`). Measured as not optional - see FAILED_APPROACHES.md.

Above the crossover `n_sample` is 0 and the code is byte-for-byte upstream, which
B1 confirms on hardware to the hundredth of a MiB.

### Host expert page pinning (`c71dee331`)

`cudaMemcpyAsync` from pageable memory is neither asynchronous nor full speed: it
goes through a driver staging buffer. The commit page-aligns each host expert
tensor's range (`llama_moe_cache_page_size()`, `GetSystemInfo` on Windows and
`sysconf(_SC_PAGESIZE)` elsewhere) and registers it with the backend's host
register hook, keeping the registered spans in `pinned_spans` to release in the
destructor.

Opt-in via `GGML_CUDA_REGISTER_HOST=1`, capped at a quarter of system RAM because
pinned pages compete with the OS, `LLAMA_MOE_PIN_MIB` overrides the cap, and any
range the driver refuses simply stays pageable.

### Telemetry (`6c9e79086`)

- `stats_small` (ubatch <= 8), `stats_large`, `stats_copy` (experts read out of
  the cache for large batches) existed; `stats_upload` is new and accounts for
  the large-batch host uploads that were previously unmeasured.
- `LLAMA_MOE_CACHE_LOG=N` logs hit rate and upload volume every N slot-map
  copies, 0 = at teardown only.
- `LLAMA_MOE_CACHE_TRACE=path` writes `call,layer,n_ids,n_hit,n_miss,ids` per
  slot-map copy, which is what `moe_sim.py` and `trace_stats.py` consume.

All default off.

### SLRU protected-set ramp (`0955c91b2`)

**Someone else's open upstream PR (#30848)**, applied locally only so it could be
measured, with a local `LLAMA_MOE_CACHE_SLRU_RAMP` toggle on top. It replaces the
fixed `n_slots/2` protected cap with one that ramps from 0 to 50% as the cache
grows from 1.0 to 1.8 slots per token. **Must not be resubmitted.**

### Benchmark plumbing (`111d8372f`)

`-mcmoe`/`--moe-cache-mib` as a llama-bench sweep axis, plus the matching
`scripts/compare-llama-bench.py` column. Was upstream PR #31121, closed on
process grounds.

## Memory pathways, end to end

What one missed expert costs, and where each stage can be attacked:

```
GGUF on disk (mmap)
  -> host page cache          <- misses here if free RAM < 18.32 GiB (confounds timing)
  -> pageable host pages      <- cudaHostRegister removes the next stage's staging copy
  -> driver staging buffer    <- 3.4 to 5.5 GB/s measured, vs ~14 GB/s link
  -> PCIe 4.0 x8
  -> cache bank in VRAM       <- the slot the replacement policy chose
  -> MoE matmul
```

The replacement policy attacks the *number* of traversals; pinning attacks the
*rate* of one traversal. They are independent, which is why the pinning sweep is
run at 1024 MiB where the policy is inert.

## Prefill bypass

`get_slot_map` returns `nullptr` for `n_tokens > max_batch = 32`, so prompt
processing takes the host path and **nothing populates the cache from the
prefill**. `copy_experts()` lets a large batch still *read* experts that happen to
be resident, but it never admits new ones.

This is deliberate upstream: a large batch touches most experts of a layer and
would evict the generation hot set. Measured cost: 5.7 GiB over PCIe for a
54-token prompt. Upstream #30594 is the attack on it.
