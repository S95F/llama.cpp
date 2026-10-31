# ROADMAP

Prioritized optimization opportunities. Reordered after every experiment, not
followed blindly. Each item carries an estimate of gain, complexity, risk and
cost, and names the evidence that motivates it.

Last reprioritized: 2026-10-31, after EXP-0004.

**What EXP-0004 changed.** The policy work (R1 to R6) cut uploaded bytes on the
premise that compute hides behind the copy. It does not: nothing overlaps, and a
warm 2048 MiB token is 12.8 ms compute + 27.1 DMA + 0.1 read-back + 16.1 host
waits. Bytes still matter, but the rate of each byte and the host's waits are
now the larger levers, and their bounds are computed from measured per-layer
times (KNOWLEDGE_BASE.md Finding 10). The host's
memory comes first, overlap second, policy third.

## Done

### R1 - Re-collect the four traces on `d854d98b` and re-validate the simulator

**EXP-0001, Adopt.** Simulator validated to exact hit and fill counts on all four
traces; every simulated figure reproduces within +0.45 points and no ranking
changed. Raw results committed. This unblocked R2 and R3, and it added the paired
per-trace screening rule that both must now use (BENCHMARKS.md).

### R2 - Close more of the Belady gap below the crossover

**EXP-0002, Adopt.** Candidate 2 (frequency plus position in the sweep), chosen
by the Q2 per-layer analysis rather than guessed. Weighting the sampled victim by
the known cyclic distance to its layer's next visit gains **+1.74 to +2.94 points**
of hit rate on hardware and a further **2.3 to 3.4%** of PCIe traffic, is
sign-consistent on all four traces at all eight simulated sizes, stays
byte-identical above the crossover, and passed the correctness gate 8 of 8.
KNOWLEDGE_BASE.md Findings 7 and 8.

Candidate 1 (per-layer slot budgets) was **ruled out by measurement**: the
shared pool with a frequency-scored victim already allocates by cacheability
(top8 share vs per-layer hit rate, r = +0.979). Candidate 3 (admission control)
remains unexplored but EXP-0001 lowered its expected value.

### R6 - Raise the sample count below the crossover

**EXP-0003, Adopt.** 16 -> 32 sampled slots: **+0.53 to +0.86 points** of hit
rate on hardware and 0.76 to 1.12% less PCIe traffic, matching the simulated
prediction within 0.18 points, byte-identical above the crossover, correctness
8 of 8. One constant.

The screen also mapped the whole curve: marginal gain per doubling is +1.40
(8->16), +0.62 (16->32), +0.21 (32->64), -0.04 (64->96), so it saturates near 48
to 64. **32 deliberately leaves +0.21 on the table** for half the sampling work,
per `AGENTS.md`'s preference for the simpler change. Revisiting it is a
one-constant change. KNOWLEDGE_BASE.md Finding 7.

### Q6 - Measure whether the upload overlaps compute

**EXP-0004, Adopt (measurement).** It does not: 0 ns of overlap in every token of
15 traced runs, by construction of the code. The decomposition, the page-fault
mechanism and the bounds below all come from it. KNOWLEDGE_BASE.md Findings 4, 9
and 10.

## Priority 1

### R10 - A controlled baseline: page-cache state, persistent process, B2

| | |
|---|---|
| Motivated by | EXP-0004: pageable runs after a pinned run started partly warm (prefill 3.7 to 7.3 s instead of 13.4 to 13.8 s); a cold 128-token run is 2.1 to 2.5x slower per token than warm; the traced CLI is 10 to 36% slower per token than B2 at 2048 MiB |
| Hypothesis | With the cache state fixed at the start of each run - cold (standby list flushed) or warm (a discarded generation first, in the same process) - repeated timelines agree within 3%, and an untraced run reconciles with `llama-bench` at the same configuration. |
| Falsifiable by | Reps of one arm differing by more than 3% under a fixed state, or the B2 gap surviving the same process, flags and token window. |
| Expected gain | None by itself. It is what makes R11 to R14 measurable at the 10% level. |
| Complexity | Low: a warm-up generation in one process (`llama-server` with two requests, or the bench harness), and the prefill time plus prefill slow copies (about 6480 when cold and pageable, about 3600 cold with 42% pinned) recorded as a coldness check. |
| Risk | Low. Flushing the standby list needs an admin tool, so a cold arm needs the user. |
| Cost | Low. Fold it into EXP-0005 as its first step rather than spend an iteration on it. |

### R11 - Pin up to the WDDM cap, most-missed ranges first

| | |
|---|---|
| Motivated by | Finding 10: pinning every expert bounds a warm 2048 MiB token at about 35 ms, and the WDDM cap at about 38 to 39 (from 56.2; projections); Finding 4's sweep measured +88% at 81% pinned; cold, 78 to 86% of pinning's gain is avoided page faults |
| Hypothesis | Pinning about 15000 MiB cuts the warm 2048 MiB token from 56.2 to 42 ms or less, and cold short runs by more than 40%; choosing ranges by miss frequency beats address order at the same pinned MiB. |
| Falsifiable by | A warm token above 46.6 ms (the 42% figure) at about 15000 MiB, or frequency order not beating address order in page faults plus DMA per token. |
| Expected gain | About -17 to -18 ms per token warm at 2048 MiB (-30 to -32%; a projection), more cold. |
| Complexity | **None for the cap** (`LLAMA_MOE_PIN_MIB`). Small for the ordering, in `pin_host_experts`. |
| Risk | Medium at the machine level: 15 GiB locked of 31 leaves about 1.5 to 3 GiB free, so watch the desktop and the free RAM. Partial pinning slows the unpinned layers a little (194.7 against 181.7 us/MiB). |
| Cost | Low. |

**Check before writing the ordering code:** `pin_host_experts` pins tensor
ranges in address order, first fit per tensor (the expert tensors are separate
ranges and never merge). EXP-0004 found that the default 7842 MiB pins layers 0
to 15 plus layer 16's gate and down tensors and covers about 49% of the warm
upload bytes for 42% of the expert bytes, because the lowest layers are the
least cacheable (Finding 8; layer 0 hits 1.84%). So address order tracks miss
frequency at the bottom but not at the tail: at 15266 MiB it leaves layer 32's
up tensor and layers 33 to 39 pageable, about 18% of the warm upload bytes,
while leaving the least-missed layers pageable instead would leave about 13 to
14%. Ordering experts within layers could add to that. Both need
per-(layer, expert) miss counts from EXP-0001's traces through the simulator;
measure that coverage before writing code.

R5 (`--load-mode none`) belongs in the same experiment as a comparison arm: it
pins the full 18.3 GiB, which the WDDM budget suggests may not even allocate
here.

## Priority 2

### R12 - Fault in each layer's missed experts with one batched prefetch

| | |
|---|---|
| Motivated by | Finding 9: a decode fault reads 64 KiB synchronously in about 175 us (0.37 GB/s), while the startup prefetch reads the same disk at 3 to 4 GB/s; `llama_prefetch_rows` exists and the MoE cache never calls it |
| Hypothesis | One `PrefetchVirtualMemory` call per layer over all missed expert ranges, issued right after the router read-back and before the copies, turns serial faults into a parallel read and halves slow-copy idle in cold runs. |
| Falsifiable by | A CPU-only microbenchmark (scattered 576 to 840 KiB ranges of the GGUF, cold cache) showing batched prefetch no faster than faulting by touch; or in-situ slow-copy idle not falling. |
| Expected gain | Cold 2048 MiB: slow-copy idle is 91.7 ms per token, about 10 at prefetch speed, so up to about -80 ms. Warm: at most 6.7 ms pageable, 3.9 at 42% pinned, less after R11. |
| Complexity | Small; the helper exists. |
| Risk | Low: prefetch is advisory and changes no data path. |
| Cost | Low, but the cold microbenchmark needs the user to flush the standby list. |

Ranked below R11 because R11 also removes most of the faults it targets on the
pinned share, and R12's large gain is only for cold or short generations.

### R13 - A copy stream with next-layer prediction

| | |
|---|---|
| Motivated by | Finding 10: overlap can hide at most 9.4 ms per token now (17%) and about 9 of about 35 after full pinning (26%); the two closest studies measure 8 to 15% of token time from prefetch alone, the wider literature +5 to +34% tokens per second, about 5 to 25% of token time (SOURCES.md) |
| Hypothesis | Uploading layer `il+1`'s predicted experts on a second stream during split `S(il)` hides at least 4 ms per token warm. |
| Falsifiable by | Gate 1: a microbenchmark where a pinned host-to-device copy on a second stream does not run concurrently with a kernel under WDDM with one copy engine. Gate 2: next-layer recall below about 70% (Q7 offline from the traces for selection-based predictors; the router-on-current-hidden-state predictor needs a hidden-state dump). |
| Expected gain | 4 to 7 ms per token, bounded at 9.4. |
| Complexity | **Large**: a second stream and events, ordering each fill after the last kernel that read the evicted slot (the pool is shared by a whole cache group), and the CUDA graphs. |
| Risk | Medium to high: correctness of slot reuse, and upstream acceptance of a scheduling change. |
| Cost | High. Run both gates first; they are cheap. |

### R14 - Shorten the per-layer host round trip

| | |
|---|---|
| Motivated by | EXP-0004: sync, 32-byte read-back, plan, the pageable slot-map copy and the graph launch cost 65 to 86 us per layer, 2.6 to 3.4 ms per token; the slot map is never registered |
| Hypothesis | Registering the slot-map buffer and dropping one of the two syncs per layer saves at least 1.5 ms per token. |
| Falsifiable by | Less than 1 ms per token of change on the timeline. |
| Expected gain | At most 2.6 to 3.4 ms (5% now, about 10% of a pinned token). |
| Complexity | Small for the registration; dropping the scheduler's sync touches `ggml-backend.cpp`, which is shared by every backend. |
| Risk | Medium: the sync orders the slot-map write against earlier reads. Needs B3. |
| Cost | Low. |

## Priority 3

### R2b - Close the rest of the Belady gap

| | |
|---|---|
| Motivated by | Finding 6 after EXP-0002: the policy now closes 58 to 62% of the gap at 256 to 528 MiB, but Belady still gets 32.01% where it gets 18.68% at 256 MiB |
| Hypothesis | None formed yet. The two structural facts available (routing skew, cyclic sweep) are both now exploited, so the next candidate needs a new source of information. |
| Falsifiable by | n/a until a hypothesis exists. |
| Expected gain | 13 points of hit rate remain at 256 MiB, but the easy structure is spent. Each point is worth less than it looked before EXP-0004: warm, DMA is about half a token, so 1% fewer bytes is about 0.5% of token time. |
| Complexity | Unknown. |
| Risk | - |
| Cost | Low to screen, once there is something to screen. |

**Screening rule, from EXP-0001:** compare paired per trace and require the sign
of the delta to agree on all four traces. One policy varies by 4.2 to 7.3 points
between traces, so a positive mean is not evidence.

Candidate status after EXP-0002:

1. **Per-layer slot budgets - ruled out.** The per-layer breakdown showed the
   shared pool already allocates by cacheability (top8 share vs per-layer hit
   rate, r = +0.979) while the gap to Belady is near-uniform across layers
   (stdev 3.23). There is little for an explicit budget to win. KNOWLEDGE_BASE.md
   Finding 8.
2. **Frequency plus position in the sweep - done,** EXP-0002, adopted.
3. **Admission control - still open, expected value lowered.** EXP-0001 showed
   S3-FIFO is a wash with the branch from 528 to 1024 MiB, so the admission idea
   is worth isolating from the FIFO structure but is unlikely to be large.

Two genuinely new sources of information, neither yet tried:

4. **Layer-to-layer prediction** (OPEN_QUESTIONS.md Q7). Pure offline analysis on
   committed traces. Now doubly useful: it is also R13's predictor.
5. **Profile-driven resident hot set**, which is what upstream #30594 is building.
   Prefer the upstream solution.

Not a candidate: **full LFU**. EXP-0001 showed it wins only at 256 MiB, and not
sign-consistently. See FAILED_APPROACHES.md.

### R3 - Measure whether the crossover switch should be a ramp

| | |
|---|---|
| Motivated by | Finding 3: frequency wins below 1.0 slots per token, loses 2 to 3 points above it; the switch is currently a hard step at 1.0 |
| Hypothesis | Between 1.0 and about 1.2 slots per token a blend beats both pure policies, because thrash decays gradually rather than stopping at exactly one token. |
| Falsifiable by | No blend beating `max(frequency, recency)` at any size in 1.0 to 1.3 slots per token, with the sign agreeing on all four traces. |
| Expected gain | Small, 1 to 3 points over a narrow band of cache sizes. |
| Complexity | Low. |
| Risk | Low, but it adds a tuning constant, which upstream dislikes and which the current hard switch deliberately avoids. |
| Cost | Low - the simulator is validated. |

The band is narrow and the gain is small, and the present design's main virtue
with reviewers is that it introduces **no** new constant.

EXP-0001 made this slightly less attractive. At 586 MiB (0.99 slots per token)
the sampled victim already wins by 7.88 points, sign-consistent on all four
traces, and at 645 MiB (1.09) it loses by 1.75. The transition is sharp rather
than gradual, which is what a hard switch handles well. The remaining question is
only whether anything useful sits between 0.99 and 1.09, a band this model's
group arithmetic barely resolves.

### R4 - Separate the pinning gain from page residency

| | |
|---|---|
| Motivated by | Finding 4: on this host the pinning gain is partly avoided page faults. EXP-0004 split it at 42% pinned (cold 78 to 86% faults; warm mostly DMA contention), so what is left is whether pinning pays when nothing faults |
| Hypothesis | With the page cache fully warm and enough free RAM that the mmap working set never misses, pinning still gives a double-digit gain from removing staging contention alone. |
| Falsifiable by | Pinning giving no gain once page residency is controlled for. |
| Expected gain | None new - this is **attribution**, which decides whether the pinning commit is worth proposing upstream at all. EXP-0004's warm window suggests about 6.9 ms of the 9.6 saved is DMA and staging, which favours it (level 4 until a host without faults confirms it). |
| Complexity | Low in code, high in method: this machine has 31.11 GiB against an 18.32 GiB working set. |
| Risk | Low. |
| Cost | Medium. Needs a smaller MoE model whose expert working set fits comfortably in RAM, which is **not currently on disk**. |

Blocked on acquiring a smaller MoE model. Recorded in OPEN_QUESTIONS.md as Q3.

### R5 - Benchmark `--load-mode none` against capped in-place pinning

| | |
|---|---|
| Motivated by | RESULTS.md section 9: this comparison is explicitly still open |
| Hypothesis | Capped in-place pinning beats `--load-mode none` on this machine, because the latter pins the full 18.3 GiB and reads the whole file at startup. |
| Falsifiable by | `--load-mode none` matching or beating it. |
| Expected gain | None new; it tells us whether the pinning commit duplicates an existing upstream option, which decides its upstream value. |
| Complexity | None - it is an existing flag. |
| Risk | Medium at the machine level: pinning 18.3 GiB of 31.11 GiB may drive the host into swap, and EXP-0004's reading of the WDDM budget (about half of RAM) suggests the allocation may simply fail. Check free RAM first and be ready to abort. |
| Cost | Low. Run it as an arm of R11. |

## Priority 4 - not yet justified

### R15 - Staging-contention microbenchmark

EXP-0004 found pageable DMA at 8.2 GB/s while the host stages the next copy and
13 GB/s when it does not. A microbenchmark (pageable and pinned DMA with and
without a concurrent host `memcpy`) would say whether host memory bandwidth or
the driver's staging path is the cause. Bounded at about 8 ms per token at
2048 MiB pageable today, but R11 removes most of it, leaving about 1.5 ms on the
bytes still pageable at the cap. Mainly explanatory.

### R9 - Populate the cache from the prefill

Finding 5 shows prompt processing pushes 5.7 GiB with zero reuse. Upstream
#30594 is already attacking this, and the knowledge base records the bypass as
deliberate. **Prefer the upstream solution**; revisit only if #30594 stalls.
Watching it is cheaper than duplicating it.

### R7 - Second MoE architecture

Every policy result rests on one model with 256 experts and 8 used per token.
Finding 2's skew is what makes the frequency policy work, and a model with fewer
experts or flatter routing should show less. This is the single largest
**external validity** gap in the project, but it needs a model that is not on
disk. See OPEN_QUESTIONS.md Q4.

### R8 - Multi-GPU and heterogeneous placement

Listed in the project directive, but this machine has one GPU. Cannot be
investigated here; recorded so it is not mistaken for untried-but-possible.

## Explicitly not on the roadmap

Rejected with evidence; see FAILED_APPROACHES.md before reviving any of these.

- Previous-token expert prefetch (token-to-token overlap is only 31 to 37%).
- Frequency sampling from the LRU end.
- Pure random eviction.
- Full LFU everywhere.
- Timing policy changes with `llama-completion -v`.
- Reading uploaded bytes over token time as a link rate.

S3-FIFO is a partial exception: rejected as a whole-cache policy, but its
admission idea survives as R2 candidate 3.
