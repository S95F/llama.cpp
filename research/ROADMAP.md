# ROADMAP

Prioritized optimization opportunities. Reordered after every experiment, not
followed blindly. Each item carries an estimate of gain, complexity, risk and
cost, and names the evidence that motivates it.

Last reprioritized: 2026-11-01, after EXP-0007.

**What EXP-0007 changed.** R12 works, in its probed form. One
`PrefetchVirtualMemory` call per layer over the missed experts' pages that are
not in the working set (`LLAMA_MOE_PREFETCH=2`) takes the profile-pinned warm
token from 41.77 to 39.18 ms (-6.2%; -9.7% over tokens 3-1022) and address
order from 45.16 to 43.42 (-3.9%). Prefetching every missed range (mode 1)
costs more than it saves once pages are resident. The fault attribution
(Q18) changed the picture of what is left: 99.9% of the remaining hard faults are a
pageable expert's first read since the load, and an expert once read
essentially never faults again (Finding 13). Whether a first read faults depends on the expert's
place in the file: the file's last 2.1 GiB never stayed resident after the
load, and the head is lost in file order as the run grows. So the fault cost is
a first-touch cost on pages the RAM budget did not keep, now paid in one
blocking parallel read per layer (0.87 ms per token warm, 3.1 ms over tokens
3-130). Two new items follow: read those experts off the decode thread (R20),
and give the experts more of the budget by releasing the pages of the tensors
uploaded to the GPU at load, which stay mapped after it (R21). A profile-pinned token is now 12.9 compute + 20.1 DMA + 6.1 idle, of which 2.1 ms is the per-layer sync (R14) and 1.6 no-API
idle, 0.87 of it the prefetch calls. 1 GiB less
launch headroom cost 0.5-0.6 s of prefill and 0.93 ms per token warm (R16b, one
level).

**What EXP-0006 changed.** The online profile (R17) works as designed but is not
a free substitute for a file profile. Warm it is within 1 ms of a file profile
from the same kind of text, and it costs about 3 s per 1024-token run up front
at the default cap (6-7 s at 15000 MiB): the passes before the registration are
pageable, and registering takes 0.54-0.57 s. Where the profile does not match
the text it is 6.2% faster warm (one out-of-domain prompt). The larger finding
is about what is left after pinning: every remaining hard fault is a pageable
copy, and under profile pinning 2.0 to 12% of the pageable expert copies fault,
against 0.9% under address order (level 4: the pageable remainder is the cold
tail, whose pages leave RAM between uses; Q18). Slow-copy idle stays at 1.6 to
3.8 ms per token warm under profile pinning (2.6 under address order), which
raises R12. Load-time registration is followed by 0.9 to 2.9 s of GPU idle
before the prompt (R19). With the launch headroom held fixed and a warm-up run,
warm reps of one arm agree within 0.32 ms and no run-position drift is
detectable (R16, its reproducibility half); a drift half EXP-0005's size is not
excluded, and the same arms ran 0.7 to 1.4 ms slower than in EXP-0005, so
compare across sessions only with care.

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

### R17 - An online selection profile

**EXP-0006, Adopt** (opt-in `LLAMA_MOE_PIN_ONLINE=<N>`). Warm (tokens 387-1022)
within 1 ms of the file profile at both caps: on64 +0.65, on128 -0.16, on64x15k
+0.83, on128x15k +0.90 ms, slower in both rounds in three of four arms, with a
deficit after the registration that fades by about token 640 (split post hoc;
consistent with first touches, not shown directly). Over tokens 3-1022 it beats
address order by about 1% (-0.47, -0.54 ms) and, at the default cap, trails the
file profile by about 3.5 ms per token (2.7-3.2 s per 1024-token run end to
end), mostly the pageable passes before the registration; at 15000 MiB by 7.05
to 7.6 ms per token (6-7 s). On one out-of-domain prompt it is 6.2% faster than
the file profile warm and 0.64 s ahead end to end. The register calls take
0.54-0.59 s at the default cap; at 15000 MiB one token stalls for 1.4-1.8 s.
KNOWLEDGE_BASE.md Finding 12.

### R16 - Hold RAM headroom fixed (done in part)

**EXP-0006, the reproducibility half holds.** With a balloon holding 19968 MiB
available at every launch, a discarded warm-up run and a new arm set in ABBA
order, reps
agree within 0.8% warm and the slope of r1 - r2 against the rep gap is +0.011
ms per position (95% CI -0.036 to +0.058; EXP-0005: 0.13). Several things
changed at once, so the result is not attributed to the headroom alone. The
prefill clause needs two or more levels: R16b.

### R12 - Fault in each layer's missed experts with one batched prefetch

**EXP-0007, Adopt** (opt-in `LLAMA_MOE_PREFETCH=2`, probed; mode 1 prefetches
every missed range). Gate G: one `PrefetchVirtualMemory` call over cold expert
ranges plus the copies took 0.24 to 0.32 of the faulting copies' time, and
cost 0.12 to 0.26 us per 4 KiB page already in RAM. Warm, mode 2 against no
prefetch: -2.59 ms (-6.2%) under the profile, -1.74 (-3.9%) under address
order; mode 1 -0.96 and +2.03. Cold unpinned (tokens 3-126), mode 1 -12.91 ms
(-16.5%). Same sign in both rounds everywhere; outputs unchanged. Mode 2 was
not run unpinned or at 15000 MiB (R12b). KNOWLEDGE_BASE.md Finding 13.

## Priority 1

### R12b - Probed prefetch where it has not run, then the default

| | |
|---|---|
| Motivated by | EXP-0007: mode 2 won warm in both pinning orders and both rounds, but ran only pinned at the default cap on one prompt; the cold unpinned arms ran mode 1 only (-16.5%) |
| Hypothesis | Mode 2 is no slower than mode 1 cold and unpinned, and no slower than no prefetch anywhere (pageable warm, 15000 MiB, the story prompt), so it can be on by default whenever the cache is. |
| Falsifiable by | Any arm where mode 2 is slower than no prefetch by more than 0.3 ms per token warm in both rounds, or slower than mode 1 cold. |
| Expected gain | None new warm; it decides the default. Cold, unpinned, about mode 1's -12.9 ms per token over tokens 3-126 (pageable warm was not run: unknown, bounded by its slow-copy idle, 6.7 ms in EXP-0004). |
| Complexity | None (arms only), plus the default switch. |
| Risk | Low; B3 already covers mode 2 around pinned experts. Add a pageable mode-2 row. |
| Cost | Low: EXP-0007's harness, about 10 runs. |

### R20 - Read the never-read pageable experts ahead of need, off the decode thread

| | |
|---|---|
| Motivated by | EXP-0007 (Finding 13): 99.9% of the remaining faults are first reads since the load, and an expert once read essentially stays; mode 2 still blocks 0.87 ms per token warm and 3.1 ms over tokens 3-130 in its prefetch calls, reading only such experts |
| Hypothesis | A background thread that, after the prefill, reads the pageable experts the decode has not yet read (into the working set, most selected first by the profile, in the lost regions first: the file's tail and the head behind the front) removes most of that blocking time without slowing the decode. |
| Falsifiable by | No-API idle not falling by at least half against mode 2, or the decode slower in tokens 3-130 (disk contention with the decode's own reads), or the lowest available RAM below the 300 MiB floor. |
| Expected gain | At most about 0.9 ms per token warm and 3 ms over tokens 3-130 under the profile (the no-API idle mode 2 adds); about 1.4 s per 1024-token run. |
| Complexity | Small to medium: a thread, the pageable ranges (the pinned table already gives them), a touch or a prefetch plus touch, stopping at teardown. |
| Risk | Medium: the budget is the limit. The 18.3 GiB of experts and the process do not fit in what is available here, so reading the lost experts back takes pages from somewhere (the front would move faster, or other processes page out). Settle Q21 first; R21 may be the better first step. |
| Cost | Low to measure with EXP-0007's harness and copy log. |

### R21 - Release the mapped pages of the tensors uploaded to the GPU

| | |
|---|---|
| Motivated by | EXP-0007 (Finding 13): the file's last 2.1 GiB never stays resident after the load at about 19.5 GiB available, and those experts' first reads fault 82 to 100% of the time (23 to 54% of all slow first reads over tokens 3-1022). The non-expert tensors uploaded to the GPU (2.0 GiB of the 2.49 GiB; the 0.5 GiB token embedding stays in a CPU mapping) are read through the mapping and stay mapped: the loader unmaps only the file before the first and after the last CPU-mapped tensor (here about 0.5 GiB), and on Windows `llama_mmap::unmap_fragment` does nothing (level 2, code) |
| Hypothesis | After the load those pages sit in the working set (level 4), where nothing trims them; releasing them (unmapping and remapping only the CPU-side ranges, or removing them from the working set) gives about 2.0 GiB back to the experts, so the tail stays resident and the head front moves later. |
| Falsifiable by | `QueryWorkingSetEx` after the load showing the non-expert pages not in the working set (then there is nothing to release), or slow first reads per token not falling by at least a third with them released. |
| Expected gain | Bounded by the faults' cost: 3.5 ms per token of slow-copy idle warm under the profile without a prefetch, 0.87 ms of blocking reads with mode 2. The tail alone is 28% of the slow first reads over tokens 3-1022 but only about 10% warm (its experts are first read early); the freed RAM should also hold back the head's front, which is most of the rest. Also the prefill (H4: headroom costs it), and 2.0 GiB of RAM for every user who offloads a model like this one with mmap (on Linux the kernel can reclaim clean mapped pages, so the gain there is untested), which is upstream material. |
| Complexity | Small to measure (one query after the load); medium to fix (Windows has no partial unmap of a view: remap the expert ranges as separate views, or use the working-set trim calls on the ranges). |
| Risk | Low to measure; medium to fix (the CPU-side tensors, such as the token embedding, must stay mapped). |
| Cost | Low. |

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

### R18 - A hybrid profile: the file profile at load, re-ranked online

| | |
|---|---|
| Motivated by | EXP-0006: the online profile matches a same-domain file profile warm but pays about 3 s up front at the default cap (6-7 s at 15000 MiB); on an out-of-domain prompt the file profile kept only a 1.5% warm gain over address order and the online profile was 6.2% faster than it |
| Hypothesis | Pinning the file profile at load and, after N passes, swapping its least selected experts for the run's own most selected ones keeps the file profile's start and reaches the online profile's warm speed out of domain. |
| Falsifiable by | Warm out of domain more than 1 ms slower than the online profile, or in domain more than 1 ms slower than the file profile, or a whole-run cost above 1 s against the better of the two. |
| Expected gain | Out of domain about 2 ms per token warm without the online profile's pageable start; in domain none. |
| Complexity | Medium: unregistering a range while no copy reads from it (the stream is idle at the registration point), and the range table and edge split updated in place. Registering in chunks across passes would also spread the 15000 MiB stall (R19). |
| Risk | Medium: unregister under WDDM is untested here; the swap must not race uploads. |
| Cost | Low to measure once written; reuses EXP-0006's harness and prompts. |

### R19 - The deferred residency cost of registration

| | |
|---|---|
| Motivated by | EXP-0006: registering 7.66 GiB of evicted pages takes 2-3 ms, but a GPU-idle stream sync of 0.9 to 2.9 s follows every load-time registration before the prompt (at most 53 ms in arms that pin nothing at load); mid-run at 15000 MiB one token stalls 1.4 to 1.8 s, inside one register call (N = 64) or at the next sync (N = 128) |
| Hypothesis | The cost is paging the non-resident registered pages into RAM (level 4; the balloon's pressure is the other candidate), in the register call or at the next submission. Prefetching the range before registering it (as the load's prefetch does) moves the cost off the critical path. |
| Falsifiable by | The idle sync unchanged when the registered pages are resident first (no eviction, or a prefetch before registering). |
| Expected gain | Time to first token: up to 1.2 s at the default cap (1.5 s on the story prompt), 2.9 s at 15000 MiB; the online profile's 15000 MiB stall. |
| Complexity | Small to measure (a microbenchmark with `cudaHostRegister` on an evicted file mapping); the fix is a prefetch call. |
| Risk | Low. |
| Cost | Low. |

### R16b - Prefill against launch headroom (the rest of R16)

R16's reproducibility half is done (EXP-0006). EXP-0007 added one level 1 GiB
lower (18944 MiB): prefill 3.22-3.25 s against 2.59-2.74, slow copies over
tokens 3-1022 5.00 per token against 3.70, +0.93 ms per token warm, all in
both rounds. A slope needs a third level; 2 GiB below 19968 would cross the 300 MiB floor
under both pinnings (`prof-lo`, 1 GiB below, already reached 919 to 1015 MiB
during the decode), so place it above 19968 (for example 20992 MiB) or at
most about half a GiB below 18944, with profile pinning.
Originally: whether prefill time and slow copies rise as the launch headroom
falls needs two or more balloon levels. It may also explain why EXP-0006's arms ran 0.7 to 1.4
ms slower warm than EXP-0005's (mostly slow-copy idle, at 0.2 to 1.7 GiB less
headroom); the page cache is an untested alternative. Fold it into the next
timing experiment as a second balloon level.

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
