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

1. `build_moe_ffn` calls `build_lookup` for every host-resident layer (upstream
   #32157, which replaced `backend()` and the graph-side `get_slot_map` call). It
   asks for `get_slot_map(il, n_tokens, n_expert_used, cpu_part)`.
2. That returns the layer's slot map only if the layer is cached, `n_tokens` is
   nonzero and **`<= max_batch` (32)**, and the tokens' experts fit the group's
   slot count. Otherwise `nullptr`, and the graph takes the host path.
3. The lookup is a `ggml_get_rows` of the slot map by the selected ids, placed on
   the cache's device. The slot map is a host weight, so the scheduler copies it
   host-to-device. That copy is intercepted by `llama_moe_cache::copy`, which is
   registered as the `ggml_backend_sched` copy callback.
4. Inside the callback the cache reads the selected expert ids, resolves each to a
   slot, and **uploads only the experts that are missing**, then writes the
   resolved slot map to the device. `build_experts` then builds the matmuls on
   the banks with the resolved slots. With a CPU share (`cpu_share`, 0 by
   default in the CLI) some misses are computed from the host experts instead
   (`cpu_map`) and added in.

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

The LRU `victim()` walks the probationary list then the protected list from the
LRU end and takes the first evictable slot. This was upstream's policy through
base `a91eab67d`. Upstream has since replaced it with ClassLRU (#31233). The
merge `10c477876` (upstream `cddaa915a`) kept the branch's policy instead and
ported upstream's other cache changes onto it: CPU misses, chosen by lowest CRF;
"not cached" as a slot `>= n_slots` instead of -1; and the binary trace
(`LLAMA_MOE_CACHE_TRACE_BIN`, read by `scripts/moe-cache-tune.py`).

## What the branch changes

Seven commits on top of `d854d98ba`; see EXPERIMENTS.md for the index and
`moe-bench/RESULTS.md` section 11 for upstream status.

### Cyclic-distance victim score (EXP-0002)

Builds on the sampled victim below. Inside the same `victim_sampled`:

- `n_group_layers`, `n_calls`, and `last_call` (one `int64_t` per layer, 320
  bytes for this model);
- `plan()` records `last_call[il] = ++n_calls`;
- `dist(il)` returns `max(1, n_group_layers - (n_calls - last_call[il]))`, the
  calls until the sweep returns to that layer, and returns 1 when the group has a
  single layer or the term is off;
- the victim becomes the sampled slot with the largest `dist/use`, compared by
  cross-multiplication to stay in integers, instead of the smallest `use`.

The group's layer count comes from `g.layers.size()`, which the cache already
has, so no tuning constant is added. `init()` takes it; passing 0 disables the
term, which is how `LLAMA_MOE_CACHE_DIST=0` restores the previous policy.

Why it is sound: the sweep visits a group's layers in order, so the next visit to
a layer is known rather than predicted. Belady evicts the entry whose next use is
furthest; approximating the time to the next use of expert `e` of layer `v` as
`dist(v) / P(e picked at v)` and using the counter as a proxy for `P` gives the
score above. Correctness does not depend on the sweep being cyclic - in any other
order `dist` is just an arbitrary bounded weight, and the "cannot evict what this
request needs" rule and the LRU fall back are untouched.

### Sampled least-used victim (`3db731656`)

The first research contribution. Inside `moe_cache_slru`:

- `n_sample` (default 16 when enabled, 0 = plain LRU, upstream's behaviour
  before ClassLRU), set at `init`
  **only when `g.n_slots < n_expert_token`** - the same condition upstream
  already warns about, so no new tuning constant is introduced.
- `use`, one `uint32_t` per (layer, expert), 40 KiB for this model, incremented
  when an expert is selected.
- `victim_sampled(il)` draws up to `n_sample` candidate slots with a small LCG
  (`rng*1664525 + 1013904223`), skipping slots holding an expert this request
  needs, returning a free slot immediately if it draws one, and otherwise the
  **least used** of the sample. It gives up after `4*n_sample` draws.
- `victim()` tries `victim_sampled()` first and **falls back to the LRU walk** if
  it returns -1, so the eviction path can never fail.
- Aging: every `age_every = 8*n_slots` fills, all counters are halved
  (`u >>= 1`). Measured as not optional - see FAILED_APPROACHES.md.

Above the crossover `n_sample` is 0 and the code is the plain LRU, which was
byte-for-byte upstream's until ClassLRU; B1 confirmed that on hardware to the
hundredth of a MiB on every base through `a91eab67d`.

### Host expert page pinning (`c71dee331`)

`cudaMemcpyAsync` from pageable memory is not asynchronous: it goes through a
driver staging buffer and blocks the host. Its DMA runs at full speed (12.9 to
13.1 GB/s) unless the host is staging the next pageable copy at the same time,
when it drops to 8.2 GB/s (EXP-0004; KNOWLEDGE_BASE.md Finding 4). The commit
page-aligns each host expert tensor's range (`llama_moe_cache_page_size()`,
`GetSystemInfo` on Windows and `sysconf(_SC_PAGESIZE)` elsewhere) and registers
it with the backend's host register hook, keeping the registered ranges in
`pinned_ranges` to release in the destructor.

Opt-in via `GGML_CUDA_REGISTER_HOST=1`, capped at a quarter of system RAM because
pinned pages compete with the OS, `LLAMA_MOE_PIN_MIB` overrides the cap, and any
range the driver refuses simply stays pageable. The log line gives the MiB
pinned, the MiB left pageable and the number of ranges.

### Profile-ordered pinning and split uploads (EXP-0005)

**`LLAMA_MOE_PIN_PROFILE=<trace>`** replaces address order with selection order.
`ranges_by_use` reads a `LLAMA_MOE_CACHE_TRACE` CSV
(`call,layer,n_ids,n_hit,n_miss,ids`), counts selections per (layer, expert)
over the small-batch rows only, and sorts the (layer, expert) units by count,
ties by address. Under the cap it takes, first fit, the page-rounded slice of
every expert tensor of each unit (gate, up and down together), then merges
adjacent ranges. At the default cap that is 7299 ranges here, at 15000 MiB 4881,
registered in 0.26 to 0.64 s. A profile it cannot read, or with no selections,
logs a warning and falls back to whole tensors in address order.

**Why every upload is split.** A host-to-device copy that starts in a registered
range and runs past its end fails (KNOWLEDGE_BASE.md, the copy-edge rule).
Expert tensors start off page boundaries, so the page-rounded range of a pinned
expert covers the first partial page of its neighbour, and the cache fills and
large-batch uploads copy runs of consecutive experts.
`llama_moe_cache_set_async` therefore splits every host upload of expert data
where it leaves a pinned range (required) and where it enters one (so the pinned
part goes by DMA, not through staging). It uses a **process-wide, sorted table**
of the ranges every MoE cache registered, under a mutex. A second context on the
same model registers none of the first one's ranges (with a profile, nothing at
all; in address order it goes on to pin the next whole tensors up to its own
cap), because the pages are already registered, but still splits at the first
context's ranges. A context without a cache uses it too. Registration holds the
lock, so no upload can cross a range between its registration and its entry in
the table, and the destructor unregisters its own ranges and removes them.

**The scheduler's fallback.** When `sched_copy_experts` declines a host weight
(the split's first node is not the `MUL_MAT_ID` that reads it), the scheduler
would copy the whole tensor in one piece. While any range is pinned
(`llama_moe_cache_has_pinned`), the callback uploads it through the split
instead. Not reached with this model, where every expert tensor starts its own
split (122 splits in the prefill graph).

### Online selection profile (EXP-0006)

**`LLAMA_MOE_PIN_ONLINE=<N>`** builds the profile from the run itself, so no
trace file is needed. At load `pin_host_experts` sets up the cap and the
registration hook as usual but registers nothing. In `copy()`, after the router
read-back has synchronized the stream and before the fills, `pin_online` adds
the ids of every small batch (up to 8 tokens) to a per-(layer, expert) count. A
pass starts at the first cached layer; at the start of pass N + 1 it ranks the
counts with `ranges_by_count`, the same ranking `LLAMA_MOE_PIN_PROFILE` applies
to a trace (`ranges_by_use` now only parses the file), and registers the result
with `register_ranges`, which both modes share. Registration holds the range
table's lock, and the stream is idle at that point, so no copy is in flight
from a page whose state changes. Unselected experts fill the rest of the cap in
address order, which makes long runs: 2832 ranges at 15000 MiB after 64 passes,
registered in 164 ms (smoke run). `ranges_by_count` returns the merged ranges
most selected first (a merged range takes the best rank of its experts), so if
the driver refuses part of the budget, the least selected experts are the ones
left pageable; the log line ends with the MiB refused. If
`LLAMA_MOE_PIN_PROFILE` is also set, it is ignored with a warning. If every
expert is already in a pinned host buffer, there is nothing to pin and online
mode stays off. Nothing is ever unregistered before the destructor, so the
first N passes decide for the whole life of the context.

Measured costs (EXP-0006, model evicted before launch; level 1 for the timings):
the register calls take 0.075-0.078 ms per range for N = 64 and 128 on prompt a
(544-565 ms for 7104 to 7392 ranges), and 587-593 ms for 6501 ranges on the
story prompt (one call 142-147 ms). `cudaHostRegister` on evicted pages can
return fast: address order registers 7.66 GiB at load in 2-3 ms, and 0.9 to 2.9
s of GPU idle follows at the next stream sync before the prompt. Mid-run at
15000 MiB one token stalls 1.4 to 1.8 s: inside one register call (1.5-1.6 s)
for N = 64, at the next stream sync (1.2-1.3 s) for N = 128. Level 4: the
leading candidate is paging in the non-resident registered pages, the balloon's
pressure the second; the mechanism is unknown. An untraced run without eviction
logged 14.57 GiB registered in 164 ms.

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

## One generated token, in order (EXP-0004)

Level 2, from the code: three readers' claims as corrected by six adversarial
verifiers (`research/results/experiments/EXP-0004/code_reading.json`). Figures
given as measured are level 1, from the EXP-0004 trace and `device_query.json`.

**42 splits per generated token**, matching the `splits = 122 / 42` log line:
one CPU split (the token-embedding `get_rows`, which CUDA never offloads), one
CUDA split from layer 0's attention to its router and top-k, then 40 CUDA splits.
Each of those 40 starts at a layer's slot-map lookup, because the lookup reads a
host weight and the scheduler cuts a split there. Split `S(il)` holds the routed
experts of layer `il`, its shared expert, the residual, then layer `il+1`'s
attention or Gated DeltaNet and its router. `S(39)` ends with the output norm and
the 515 MiB LM head instead. 30 of the 40 layers are Gated DeltaNet; 10 are full
attention (`src/models/qwen35moe.cpp`).

At every MoE layer:

1. The host enters `S(il)` and copies its one host-weight input, the slot map.
   There are no scheduler events (`n_copies = 1`, because there is one device and
   `-ncmoe` adds tensor overrides), so it first calls `ggml_backend_synchronize`:
   **the host waits until all of `S(il-1)` has run** (`ggml-backend.cpp`, about
   line 1844).
2. The copy callback reads the 8 selected ids back (32 bytes) and **synchronizes
   again** (`llama-moe-cache.cpp`, about line 846). The GPU is idle from here.
3. The host plans hits and fills in the SLRU.
4. It issues one `cudaMemcpyAsync` per run of consecutive misses for each expert
   tensor, in the order gate, up, down, then the slot map, **all on the one
   compute stream**. From pageable memory each call is staged by the driver and
   blocks the host.
5. Only then does the scheduler launch `S(il)`, as a CUDA graph.

So **uploads and compute are serialized twice over**: by stream order on the
device, and by the two host waits per layer. Nothing can overlap today without
code changes. `set_tensor_async` has no stream argument, the scheduler creates
no events here, and the only other host-to-device path (`cudaStreamPerThread` in
buffer `set_tensor`) synchronizes the host.

`GGML_CUDA_REGISTER_HOST=1` changes no stream or graph eligibility, but while
anything is pinned every expert upload, and the scheduler's whole-tensor
fallback, goes through `llama_moe_cache_set_async`, split at pinned-range edges
(no copy crosses one in address order; a profile adds 12 to 70 small copies per
token). Otherwise it only makes the registered expert ranges pinned, capped by
default at RAM/4. The slot map is never registered, so the last copy of every
layer stays pageable.

**What could legally overlap layer `il`'s upload without prediction:** the shared
expert and its gate (they read only the FFN input; the routed+shared fused
kernel cannot fire here, because the shared expert is Q8_0 and the routed experts
are K-quants), the routed matmul over the hits if `MUL_MAT_ID` were split by
readiness, and each missed expert's gate/up while its down tensor uploads. All of
it is small, probably tens of microseconds per layer (level 4: EXP-0004 traced
with `graph:host-only`, which gives no kernel times inside a graph). Everything
from layer
`il+1` depends on layer `il`'s routed output through the residual, so
cross-layer overlap needs prediction. A copy stream would also have to order each
fill after the last kernel that read the evicted slot: the slot pool is shared
by every layer of a cache group.

**Hardware limits on any future overlap** (`device_query.json`): the RTX 4060 Ti
reports **one copy engine** (`asyncEngineCount = 1`), so a host-to-device upload
could run beside a kernel but not beside the id read-back. Read-only host
registration is supported. The card runs under WDDM.

CUDA graphs **are** used: the trace shows 41 graph launches and zero individual
kernel launches per generated token. Every 256th token re-captures and runs 37
of the 41 graphs eagerly, with compute 0.2 to 0.3 ms above a graph token (1.7 ms
in one of 12 cases). Graphs cut launch cost and do nothing about the per-layer
host round trip, which EXP-0004 measured at 65 to 86 us per layer (2.6 to 3.4 ms
per token): two syncs, the read-back, the plan, the pageable slot-map copy and
the launch.

**At 256 and 586 MiB one layer gets no cache slots** ("budget is too small for 1
layers") and runs its MoE on the CPU, adding five syncs per token (114 against
109) and about 2 ms of GPU idle.

Measured costs of each stage, for the pathway below, are in KNOWLEDGE_BASE.md
Findings 4 and 9.

## Memory pathways, end to end

What one missed expert costs, and where each stage can be attacked:

```
GGUF on disk (mmap)
  -> host page cache          <- a miss is a hard fault: ~175 us per 64 KiB, serial
  -> pageable host pages      <- cudaHostRegister removes the next stage
  -> driver staging buffer    <- the host stages copy n+1 while copy n is in flight
  -> PCIe 4.0 x8              <- 12.8 GB/s pinned; pageable 13 GB/s alone, 8.2 contended
  -> cache bank in VRAM       <- the slot the replacement policy chose
  -> MoE matmul               <- only after every copy of the layer (one stream)
```

The replacement policy attacks the *number* of traversals; pinning attacks the
*rate* of one traversal, and the page faults. They are independent, which is why
the pinning sweep is run at 1024 MiB where the policy is inert. Overlap would
attack neither and hide part of the traversal behind compute instead.

The figures are from EXP-0004. The old "3.4 to 5.5 GB/s" on the staging line was
uploaded bytes over whole-token time, not a rate of this stage.

**`--load-mode none`** takes a different path: the loader reads the experts into
a `CUDA_Host` buffer instead of mapping the file. The CUDA host buffer type caps
a buffer at 4 GiB (`ggml_backend_cuda_host_buffer_type_get_max_size`), so 19275
MiB becomes five chunks, and a chunk that `cudaMallocHost` refuses becomes a
pageable CPU buffer, logged only at debug level. On this host the fourth chunk
(layers 25 to 32, plus one tensor each of layers 24 and 33) fell back
(EXP-0005). Its pageable experts sit in private memory, so they stage but never
fault.

**Where the experts are computed.** For a decode token, the experts of a cached
layer are computed on the GPU from the cache bank. Those of an uncached layer
(the cache off, or a group with no slots) are host weights, and the CUDA backend
offloads an op on host weights only at a batch of `GGML_OP_OFFLOAD_MIN_BATCH`
(default 32) or more, so the CPU computes them. That changes rounding, which is
why B3 now sets the threshold to 1 (EXP-0005).

## Prefill bypass

`get_slot_map` returns `nullptr` for `n_tokens > max_batch = 32`, so prompt
processing takes the host path and **nothing populates the cache from the
prefill**. `copy_experts()` lets a large batch still *read* experts that happen to
be resident, but it never admits new ones.

This is deliberate upstream: a large batch touches most experts of a layer and
would evict the generation hot set. Measured cost: 5.7 GiB over PCIe for a
54-token prompt. Upstream #30594 is the attack on it.
