# ROADMAP

Prioritized optimization opportunities. Reordered after every experiment, not
followed blindly. Each item carries an estimate of gain, complexity, risk and
cost, and names the evidence that motivates it.

Last reprioritized: 2026-10-31, after EXP-0005.

**What EXP-0005 changed.** Pinning near the Windows budget, the experts a
selection profile uses most first, took the warm 2048 MiB token from 52.95 to
36.43 ms (-31%): 12.9 compute + 18.2 DMA + 5.2 idle. Address order at the same
MiB was 7.3% slower (39.35 against 36.65 ms, matched window 387-841; profile
order is 6.8% faster). On this host the memory lever is now mostly spent, so the
remaining time is the DMA itself and the host's per-layer waits. Overlap (R13)
and the round trip (R14) are what is left. R17 makes the profile gain usable
without a trace file. R10's fixed launch state did not make runs agree, and the
r1 - r2 gaps track both the rise in available RAM (r = 0.79) and the time
between reps (r = 0.91), which this design cannot separate; holding RAM fixed
(R16) has to come with the next timing experiment.

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

### R11 - Pin up to the budget, most-selected experts first

**EXP-0005, Adopt** (opt-in `LLAMA_MOE_PIN_PROFILE`, plus the upload split at
pinned-range edges that expert-level pinning requires). Warm, tokens 387-1022:
off 52.95 ms, address order 15000 MiB 39.35 (matched window 387-841), profile
order 14924 MiB **36.43**. Profile against address order is -2.69 ms (-6.8%) at
15000 MiB and -3.55 ms (-8.1%) at the default cap, the same sign in both rounds.
The offline coverage model replays the pageable share to 0.1 points over its
traces' window. The budget is about 15 GiB, and at that size the desktop ran
nearly out of memory in address order (median 243 and 814 MiB available, warm);
the watchdog stopped one rep at 205 MiB. KNOWLEDGE_BASE.md Finding 11.

### R5 - `--load-mode none` against capped in-place pinning

**Settled as an arm of EXP-0005.** It does not pin the full 18.3 GiB. The CUDA
host buffer type caps buffers at 4 GiB, and a chunk that `cudaMallocHost`
refuses silently falls back to pageable memory, so it ran about 80% pinned. Warm
38.21 ms on the matched window: 1.14 ms faster than address order at 15000 MiB,
1.56 ms slower than profile order, and a 15 s load instead of 6.3 to 7.5 s.

### R10 - A controlled baseline (done in part)

**EXP-0005, falsified in part.** Evicting the model by memory pressure before
every run made the launch state reproducibly cold (0 to 7 of 256 probed pages
cached), but reps still differed by up to 3.9% warm and 28% in tokens 3-130, and
r1 was slower than r2 in every arm. The page cache refills from the load's
whole-file prefetch, and how far it gets depends, by hypothesis (confounded
with run order), on available RAM, which rose 1.5 GiB through the session. The
persistent-process run and the B2 reconciliation were not done. They move to
OPEN_QUESTIONS.md Q12, the headroom control to R16.

## Priority 1

### R17 - An online selection profile

| | |
|---|---|
| Motivated by | EXP-0005: profile pinning is 6.8 to 8.1% faster than address order but needs a trace file from a previous run; offline, counting a run's own first 64 or 128 tokens and pinning then leaves 5.5 to 7.6% or 2.8 to 4.2% of miss bytes pageable at 15000 MiB (file profile, held out: 2.1 to 5.7%) |
| Hypothesis | Registering by selection count after the first N generated tokens (N about 64 to 128) reaches within 1 ms per token of the file profile, warm, at the same cap. |
| Falsifiable by | The online arm more than 1 ms per token slower than the file profile on tokens 387 to 1022, or the mid-run registration (0.3 to 0.6 s for 5000 to 7300 ranges) costing more than it saves over a 1024-token generation. |
| Expected gain | The file profile's -2.7 to -3.6 ms per token over address order, without a file. |
| Complexity | Small. The cache already sees every selection (the trace writer); the ranking and registration code exists (`ranges_by_use`). Registration must not race uploads: do it between tokens, under the range table's lock. |
| Risk | Low to medium: 0.3 to 0.6 s of registration in the middle of a generation; and a first request that is unrepresentative of later ones (a long-lived server should re-rank). |
| Cost | Low. Re-uses EXP-0005's harness; needs R16's fixed headroom. |

### R16 - Hold RAM headroom fixed (R10's missing control)

| | |
|---|---|
| Motivated by | EXP-0005: launch-state eviction alone left reps 1.1 to 3.9% apart warm and r1 slower in every arm, correlated with the session's RAM drift (r = 0.79) and run position (r = 0.91); prefill slow copies sat in the file tail the load left unread (level 1; that available RAM sets how much stays unread is a level 4 hypothesis) |
| Hypothesis | With a fixed balloon allocation held through each run (for example 0, 2, 4, 6 GiB, randomized order), reps of one arm agree within 2% warm, and prefill time and slow copies rise with the balloon. |
| Falsifiable by | Reps still more than 2% apart at a fixed balloon; or prefill slow copies not depending on it. |
| Expected gain | None by itself. It makes 2 to 3 ms differences (R13, R14, R17) measurable, and it tests whether EXP-0004's 13.4 s prefill regime is a RAM-shortfall effect. |
| Complexity | Low: one more script next to `evict.py`; per-process counters (IO read bytes, working set) and standby and modified list sizes in typeperf. |
| Risk | Low; the balloon must leave the desktop above the watchdog's floor. |
| Cost | Low. Fold it into the next timing experiment. |

## Priority 2

### R13 - A copy stream with next-layer prediction

| | |
|---|---|
| Motivated by | Finding 10: overlap can hide at most 9.4 ms per token pageable (17%) and about 9 of about 35 after full pinning (26%); EXP-0005's profile-pinned token is 36.4 ms, of which 18.2 is DMA serialized with 12.9 of compute. The two closest studies measure 8 to 15% of token time from prefetch alone, the wider literature +5 to +34% tokens per second, about 5 to 25% of token time (SOURCES.md) |
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
| Expected gain | At most 2.6 to 3.4 ms (5% pageable, 7 to 9% of EXP-0005's 36.4 ms pinned token; its sync idle alone is 2.3 ms). |
| Complexity | Small for the registration; dropping the scheduler's sync touches `ggml-backend.cpp`, which is shared by every backend. |
| Risk | Medium: the sync orders the slot-map write against earlier reads. Needs B3. |
| Cost | Low. |

### R12 - Fault in each layer's missed experts with one batched prefetch

| | |
|---|---|
| Motivated by | Finding 9: a decode fault reads 64 KiB synchronously in about 175 us (0.37 GB/s), while the startup prefetch reads the same disk at 3 to 4 GB/s; `llama_prefetch_rows` exists and the MoE cache never calls it |
| Hypothesis | One `PrefetchVirtualMemory` call per layer over all missed expert ranges, issued right after the router read-back and before the copies, turns serial faults into a parallel read and halves slow-copy idle in cold runs. |
| Falsifiable by | A CPU-only microbenchmark (scattered 576 to 840 KiB ranges of the GGUF, cold cache) showing batched prefetch no faster than faulting by touch; or in-situ slow-copy idle not falling. |
| Expected gain | Cold 2048 MiB: slow-copy idle is 91.7 ms per token, about 10 at prefetch speed, so up to about -80 ms. Warm: at most 6.7 ms pageable, 3.9 at 42% pinned, 0.9 with profile pinning at 15000 MiB (EXP-0005). |
| Complexity | Small; the helper exists. |
| Risk | Low: prefetch is advisory and changes no data path. |
| Cost | Low. The cold microbenchmark can use EXP-0005's `evict.py` (no admin
rights; 0 to 7 of 256 probed pages cached after it). |

Ranked below R13 and R14 because profile pinning (R11, done) already removes
most of the faults it targets, and R12's large gain is only for cold or short
generations on hosts that cannot pin.

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

## Priority 4 - not yet justified

### R15 - Staging-contention microbenchmark

EXP-0004 found pageable DMA at 8.2 GB/s while the host stages the next copy and
13 GB/s when it does not. A microbenchmark (pageable and pinned DMA with and
without a concurrent host `memcpy`) would say whether host memory bandwidth or
the driver's staging path is the cause. Bounded at about 8 ms per token at 2048
MiB pageable today, but R11 removes most of it, leaving about 1.5 ms on the
bytes still pageable at the cap. Mainly explanatory. EXP-0005 adds a second
thing to explain: pageable copies ran at 8.5 to 8.9 GB/s in off, the
address-order arms and none, and 10.5 to 11.3 in the profile arms. Their count
does not explain it: prof7842 (22.3% of warm upload bytes pageable) has more
than addr15000 (20.2%) or none (18.5%), which ran at 8.5 to 8.9.

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
- Expert-level pinning without splitting copies at the pinned ranges' edges
  (the driver rejects a copy that starts in a registered range and runs past it).

S3-FIFO is a partial exception: rejected as a whole-cache policy, but its
admission idea survives as R2 candidate 3.
