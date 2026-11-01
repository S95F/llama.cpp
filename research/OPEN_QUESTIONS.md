# OPEN QUESTIONS

Unresolved questions, missing evidence, and what it would take to settle each.
Ordered by value, not by age. Q11 to Q14 are new from EXP-0004, Q15 to Q17 from
EXP-0005, Q18 and Q19 from EXP-0006, Q20 from the 2026-11-01 merge check, Q21
from EXP-0007.

## Q21 - Why does the file's tail never stay resident, and what takes the head?

**New, from EXP-0007.** 99.9% of the hard faults left under pinning are a
pageable expert's first read since the load, and whether it faults depends on where the
expert sits in the 20.82 GiB file. The last 2.1 GiB (from about 18.7 GiB,
layers 35-39) was not resident after the load at about 19.5 GiB available: its
prefill uploads were already slow, and 82 to 100% of its first reads are slow
all run. At the head, pages are lost in file order: under the profile, layers 0-4 were already slow at the prefill and layers
5-9 early in the decode, and during the decode a front moves
forward (layers 10-14 from 0% slow first reads over tokens 3-50 to 99% over
387-1022), further at 1 GiB less headroom. The middle layers almost never
fault. The balloon's size does not move it (0 or 1344 MiB held, the same
counts); the run's own working set grew from 12.9 to 17.3 GiB.

Level 4 candidates: the load's prefetch reads the file in order into a budget
smaller than the file, and its last pages do not stay (or are never read);
Windows repurposes the oldest standby pages first as the process grows; under
profile pinning, registering 7.8 GiB at load displaces the head; and the pages of the non-expert tensors uploaded to the GPU at load (2.0 GiB)
stay mapped (the loader unmaps only the file outside the CPU-mapped span, and
on Windows `unmap_fragment` does nothing), probably in the working set,
taking about 2.0 GiB of the budget.

**To settle:** right after the load, query the working-set state of the
non-expert tensors' pages (`QueryWorkingSetEx`) and time a read of a few pages
spread over the file (as `evict.py`'s probe does) to map what is resident;
repeat at the end of the prefill and every 128 passes; then launch with more
headroom than the file needs (24 GiB or more available), where the prediction
is near-zero slow first reads in every layer.

**Why it matters:** it decides between R20 (read the lost experts off the
decode thread, within the same budget) and R21 (enlarge the budget by
releasing pages nothing reads again), and whether the load's prefetch order
should change.

## Q18 - Why does the pageable remainder of expert-level pinning fault more per byte? SETTLED

**Settled by EXP-0007 (level 1 for the attribution, level 4 for the
mechanism): the faults are first reads, and the profile's remainder has more of
them.** 99.9% of the slow pageable uploads are an expert's first read since the
load (44-45% of first reads are slow under both orders over tokens 3-1022,
depending strongly on the expert's place in the file); re-reads essentially
never fault, none after a gap of 500 passes or more. The profile pins the most
selected experts, so more of its pageable uploads are first reads (10.4%
against 4.0% over tokens 3-1022), and standardizing on that explains 103% of
the difference. The level 4 hypothesis below (pages leave RAM between uses) is
wrong. The probed batch prefetch (R12, mode 2) removes the faults' slow-copy
idle. Why the file's tail and head are not resident at a first read is Q21.

**New, from EXP-0006.** Every slow copy (an expert copy below 1.5 GB/s, a hard
fault) is pageable. The share of pageable expert copies that are slow is 0.9%
under address order, 2.0% for the file profile on the story prompt, 2.7% at the
default cap on prompt a, 3.1% for the online profile at N = 128, and at 15000
MiB 12% for the file profile (4.7 to 9.2% for the online profiles). So slow-copy
idle does not fall with the pageable share: 3.48 ms per token under the file
profile against 2.59 under address order, which leaves 28 points more pageable.
Level 4 hypothesis: a profile pins the hot experts, so its pageable remainder is
the cold tail, selected rarely, whose pages leave RAM between uses; address
order leaves whole layers pageable, hot experts included, and those stay
resident. The copy records carry no host address, so which experts fault is not
known.

**To settle:** log the layer and expert of every copy with its host time (Q11's
method), and count how often each slow copy's expert was selected before it
faulted. Then test keeping the pageable experts resident (a batched prefetch,
ROADMAP R12, or a touch) against the RAM it costs.

**Why it matters:** slow-copy idle is 3.5 to 3.8 ms of a 42 ms warm token under
profile pinning at the default cap, the largest piece left that is not DMA or
compute (1.6 to 2.2 ms at 15000 MiB, below the 2.3 ms of sync idle), and it is
one of the two reasons the offline share does not convert to milliseconds at one
rate.

## Q19 - Why does a GPU-idle sync of up to 2.9 s follow a load-time registration?

**New, from EXP-0006.** `cudaHostRegister` over 7.66 GiB of evicted pages
returned in 2-3 ms at load, but the next stream sync before the prompt is GPU
idle for 0.9 to 1.1 s (address order), 1.0 to 1.5 s (file profile; 1.2 to 1.5 on
the story) and 2.8 to 2.9 s (file profile at 15000 MiB); arms that pin nothing
at load wait at most 53 ms there. Mid-run at 15000 MiB the online registration
stalls one token for 1.4 to 1.8 s, with 1.3 to 1.6 million page faults and 1.5
to 2.1 GiB of disk reads: inside one register call for N = 64 (1.5-1.6 s), at
the next stream sync for N = 128 (1.2-1.3 s). An untraced run without eviction
or balloon logged 14.57 GiB registered in 164 ms; whether a stall followed was
not measured. Level 4: the cost is paging in the registered pages that are not
resident; where it lands, and why, is not known. Other candidates: the balloon's
memory pressure, or the load's prefetch still reading the file.

**To settle:** a microbenchmark that registers an evicted file mapping and times
the register call and the first submission, with and without a prefetch of the
range first (ROADMAP R19).

**Why it matters:** up to 2.9 s before the first token at 15000 MiB, and the
online profile's mid-run stall. If a prefetch moves it off the critical path,
it is a small fix.

## Q20 - Why do llama-bench's tg reps step up within a process?

**New, from the 2026-11-01 merge check.** At 2048 MiB, fresh process,
B2's `-p 0 -n 64 -r 4`, all eight runs on both builds had rep 1 slowest and
rep 4 fastest: rep 1 7.33 to 10.05 tok/s, reps 2 and 3 14.17 to 16.69, rep 4
20.55 to 21.02 (`results/baseline/rebase-2ca33f0d7/bench_r3.jsonl`).
llama-bench never seeds `std::rand`, so rep k decodes the same tokens in every
process, and its generation warm-up is one token. Rep 1 is slow even in a
process launched right after one that decoded the same tokens. Candidates,
level 4: first-touch faults that map pages already in RAM into the new
process; per-rep token content (rep 4's tokens routing to fewer distinct or
cheaper experts); the expert cache warming. The last is unlikely to be enough:
the cache (one per context, so empty at each test's start) holds about 3.5
tokens' worth of experts at 2048 MiB (B1's slots per token), and `bench_r2.sh`'s
tg64 also began with an empty cache yet averaged 24.72 to 27.26 tok/s. Among
what differed there: the pp512 test before it in the same process (which
touched the experts its random tokens selected, plausibly most of them; level
4, not checked) and the rand position.

**Correction and a new candidate (EXP-0007's pre-run review).** These runs were
recorded as pageable, but `bench_r3.sh` does not set `GGML_CUDA_REGISTER_HOST`,
and the user's environment sets it to 1 (ENVIRONMENT.md). So they most likely
pinned the default cap (7.8 GiB) in address order at load, as did B2's
references from EXP-0002 and EXP-0003, whose scripts did not set it either.
That adds a candidate, also level 4: Q19's deferred residency cost of the
registration (0.9 to 2.9 s of GPU idle after a load-time registration in
EXP-0006), paid in the first reps. Settle it with the variable set explicitly to 0
and to 1.

**To settle:** no code needed. `-r 8` in one process (a warm-up flattens; content
repeats by rep index across processes), the same with `LLAMA_MOE_CACHE_TRACE` for
fills per rep, `-p 0 -n 64,64` (bench_r2.sh's shape without pp512: does a
tg-only first test remove the second's staircase? its tokens differ, since the
rand stream runs on), per-process page-fault counts, and each of these with
`GGML_CUDA_REGISTER_HOST` set explicitly to 0 and to 1.

**Why it matters:** B2 means average over reps that, at 2048 MiB in a fresh
process, differ up to 2.8x (other sizes untested), in proportions set by the rep
count and by what ran earlier in the process. It may explain Q12, and why a
fresh process now measures 14.1 to 15.9 tok/s per run at 2048 MiB against B2's
21.24 to 24.25 references.

## Q15 - How much do the small-cache hardware rows depend on CPU-computed layers?

**New, from EXP-0005.** At 256, 528 and 586 MiB one cache group gets no slots,
and decode computes that layer's experts on the CPU. CPU and CUDA round
differently, so from about token 110 those runs decode a different text than
2048 MiB (and the cache-off run a different one from about token 65). Their
hardware hit rates are therefore over different routing than the simulator's
replay, and than the larger sizes. Comparisons between policies at one size are
unaffected: B3 shows every policy at 528 MiB generating the same text.

**To settle:** re-run B1 at 256 to 586 MiB with `GGML_OP_OFFLOAD_MIN_BATCH=1`
(the same text as 2048 MiB) and compare hit rates with the default rows and
with the simulator.

**Why it matters:** the small-cache hardware rows of Findings 1, 3 and 7 rest
on it (Finding 6 is simulated from the traces). Probably small (routing
statistics are similar across texts), but unmeasured.

## Q16 - Does prefill depend on available RAM, and what made EXP-0004's prefill slow?

**EXP-0007 added a second level** (18944 MiB, profile pinning): prefill 3.22
and 3.25 s against 2.59 and 2.74, slow copies over tokens 3-1022 5.00 per token
against 3.70, both rounds. So prefill and early faults do rise as headroom
falls by 1 GiB; a slope needs a third level (R16b).

**Still open after EXP-0006**, which held one headroom level (19968 MiB) and so
tested only R16's reproducibility half. Its repeated arms ran 0.7 to 1.4 ms
slower warm than EXP-0005's, mostly in slow-copy idle, at 0.2 to 1.7 GiB less
launch headroom; that may be the same effect, untested.

**New, from EXP-0005.** With the model evicted before launch, prefill took 1.55
to 3.02 s at 19.7 to 21.1 GiB available, and its slow copies sat in the file
tail that the load's prefetch left unread. EXP-0004's pageable runs after other
pageable runs took 13.4 to 13.8 s with every layer faulting, at 16.5 to 17.3
GiB. Available RAM, the eviction procedure, the code and other processes all
differ between the two sessions (level 4 hypothesis: prefill faults scale with
available RAM through how much of the file's tail the load leaves unread; that
alone does not explain EXP-0004, where all 40 layers faulted and runs after a
pinned run at about the same RAM prefilled in 3.7 to 7.3 s).

**To settle:** ROADMAP R16b, two or more balloon levels (0 to 6 GiB) with the
EXP-0006 harness, with per-layer fault positions at prefill.

**Why it matters:** it decides whether short requests on a host with less spare
RAM pay 2 s or 13 s of prefill, and it is R10's missing control.

## Q17 - Does a selection profile from one domain pin well for another? SETTLED IN PART

**EXP-0006, for one prompt: worse, and the online profile recovers it.** On a
children's story, answered outside the think block, the file profile left 34.5%
of the warm upload bytes pageable against 22.3% on prompt a, and gained 1.5%
over address order against 7.9% on prompt a. The online profile (N = 128) left
11.6% and was 2.25 ms (6.2%) faster than the file profile warm. Part of the file
profile's shortfall is not the share but extra hard faults in its pageable
remainder (slow-copy idle 2.79 against 1.69 ms per token; Q18). The story
changes topic and register together, so which of the two matters is not known,
and it is one prompt with two reps.

**Still to settle:** more out-of-domain prompts that separate topic from
register (code, non-English, an answer without thinking on the profile's
topic), at both caps.

**From EXP-0005.** The profile (prompts b, c, d) was held out from the run
prompt but shares its domain, and the run prompt is the best covered of four
offline (2.1% pageable at 15000 MiB against 4.6 to 5.7%; 20.7% at the default
cap against 26.8 to 29.3%).

**Why it matters:** the default-cap gain (-8.1%) depends most on it, and it
decides whether a hybrid profile (ROADMAP R18) is worth building.

## Q11 - Are the faults left in a warm run first touches or re-faults? SETTLED IN PART

**New, from EXP-0004.** Once warm, a 2048 MiB pageable token still makes 2 to 5
slow copies per token (3.8 on average over tokens 387 to 1022), in bursts, worth
3.8 to 9.1 ms of GPU idle per 128-token window (6.7 on average). They could be
the last first touches of rarely used experts, or pages the OS dropped from the
page cache under memory pressure and has to read again. The trace does not say
which expert each copy carried. The 256 MiB arm suggests re-faults, at least there: its partly
warm second rep faulted **more** than its cold first rep (37.2 against 25.1 per
token), and 1636 copy positions were slow in the second rep only.

**To settle:** log the expert id and layer of every copy alongside its host time
(the cache already knows both; `LLAMA_MOE_CACHE_TRACE` is the precedent), then
count how many slow copies hit an expert already copied earlier in the run.

**Why it matters:** first touches are a startup cost that pinning, prefetch or a
persistent process removes; re-faults are a residency problem that only pinning
(R11) or more free RAM removes. It decides between R11 and R12 for warm runs.

**EXP-0007, for 2048 MiB with pinning at the default cap, and the first 126
tokens unpinned: first touches.** With the expert of every upload logged, 99.9%
of the slow copies are an expert's first read since the load, and re-reads
essentially never fault (Q18). Not rerun: the unpinned warm 2048 MiB case and
EXP-0004's 256 MiB arm, whose second rep faulted more than its first, which
first touches do not explain.

**EXP-0006 narrows it for pinned runs:** every slow copy is pageable, and a
larger share of pageable copies fault under profile pinning than under address
order (Q18). Level 4: they are the cold tail re-faulting. Same method to settle
both.

## Q12 - What does a persistent process look like, and why does B2 disagree?

**New, from EXP-0004.** Every EXP-0004 run was a fresh CLI process, and a
cold 128-token one is 2.1 to 2.5x slower per token than warm. A server keeps the
process, but nothing here has measured whether its page cache then stays warm
between requests. Separately, warm 2048 MiB here (52 to 56 ms, traced) is 10 to
36% slower per token than the B2 `llama-bench` references (41.2 to 47.1 ms), while at 586
MiB the two agree.

**To settle:** two generations in one process, and an untraced run plus
`llama-bench` at the same configuration and token window. EXP-0005 did R10's
page-cache part but not this; the persistent-process run belongs with R16's
fixed headroom.

**Why it matters:** every timing from R11 on is quoted against this baseline.
EXP-0005's 52.95 ms pageable (traced, fresh process) is the current one.

## Q13 - Can a second stream overlap a copy with a kernel on this card?

**New, from EXP-0004.** The RTX 4060 Ti reports one copy engine and runs under
WDDM, where the driver batches submissions. One copy engine still allows a
host-to-device copy beside a kernel in principle; whether WDDM delivers it is
unmeasured.

**To settle:** a standalone microbenchmark, a pinned copy on one stream and a
kernel on another, with the timeline. No llama.cpp code.

**Why it matters:** it is gate 1 of ROADMAP R13. If it fails, the 9.4 ms bound of
Finding 10 is unreachable on this host whatever the predictor.

## Q14 - Why does pageable DMA slow to 8.2 GB/s while the host stages?

**New, from EXP-0004.** Pageable copies run at 12.9 to 13.1 GB/s when the host is
not staging the next pageable copy and 8.2 GB/s when it is, in every copy-size
class. Host memory bandwidth and the driver's staging path are the candidates.

**To settle:** ROADMAP R15, a microbenchmark of pageable and pinned DMA with and
without a concurrent host `memcpy`.

**Why it matters:** modestly. About 8 ms per token at 2048 MiB pageable today,
but pinning removes most of it. EXP-0005 adds that the pageable rate depends on
the arm, not on how many pageable copies there are (prof7842 makes more per
token than addr15000 or none): 8.5 to 8.9 GB/s in off, address order and none,
10.5 to 11.3 in the profile arms, and fast-copy idle may include soft faults
(788 to 831 transition faults per token warm when pageable).

## Q10 - Should the distance term also run above the crossover?

**New, from EXP-0002.** The cyclic-distance victim (Finding 7) is switched off at
and above the crossover along with the sampling it rides on. But it *would* help
there: it turns the pure-frequency policy's clear loss against upstream
(-1.75 to -3.08 points) into a wash - **+0.42** at 645 MiB and **+0.35** at 1024,
then -1.43 at 2048 and -1.17 at 3072.

It was not enabled because the screening rule requires the sign of the paired
delta to agree on all four traces, and it does not at any of those sizes: 645
goes +1.73, +0.96, -0.54, -0.48. Keeping it off also preserves the
byte-identical control rows that every throughput claim here is read against,
which is worth more than a wash.

**EXP-0003 added the first hardware datapoint**, incidentally: because
`LLAMA_MOE_CACHE_SAMPLE` also defeats the crossover switch, its `s16` arm at 1024
and 2048 MiB *is* this configuration.

| MiB | forced sampling + distance | inert (shipping) | delta | simulated |
|---:|---:|---:|---:|---:|
| 1024 | 43.46% | 41.36% | **+2.10** | +2.12 |
| 2048 | 61.23% | 61.38% | -0.15 | -0.36 |

The 1024 MiB row matches the simulator to **0.02 points**, so the effect is real
on hardware and not a simulator artifact. That raises the value of settling this.

**To settle:** more traces. Four is too few to resolve a 0.4-point effect when
the between-trace spread is 4 to 7 points. Ten to twenty traces across more topic
types would do it, and collecting them is cheap - EXP-0001 took 4 minutes for
four.

**Why it matters:** modestly. The upside is ~0.4 points at two cache sizes. The
real reason to ask is that it would remove a special case rather than add one: if
the distance term were unconditionally safe, the policy would be "always sample,
always weight by distance" with the crossover switch only selecting
`max_protected`. That is a simpler story to take upstream than the current
two-regime one.

## Q2 - How much of the Belady gap is actually reachable online? SETTLED IN PART

**EXP-0002 answered the shape question.** The gap to Belady is **near-uniform
across layers** (14.49 to 27.50 points at 528 MiB, stdev 3.23; the worst 8 of 39
layers hold 24.7% of it against 20.5% for a uniform spread), while the per-layer
hit rate varies hugely (1.84% to 49.04%) and is almost perfectly explained by
routing skew (r = +0.979 against top8 share).

That ruled out per-layer slot budgets and pointed at a better victim score, which
EXP-0002 then built: see KNOWLEDGE_BASE.md Findings 7 and 8.

**Still open:** how much of what remains is reachable. After EXP-0002 the policy
closes 58 to 62% of the gap at 256 to 528 MiB, but Belady still gets 32.01%
against 18.68% at 256 MiB. Both structural facts this project has found - routing
skew and the cyclic sweep - are now exploited, so the next step needs a new
source of information. Q7 is the only candidate with one.

## Q3 - Is the pinning gain real once page residency is controlled for? SETTLED IN PART

Finding 4's +88% is the sum of two effects. **EXP-0004 split them on this host**
at 42% pinned, from the timeline: in cold 128-token runs pinning saves 42.1 ms
per token at 2048 MiB, 86% of it avoided page faults; warm, the 9.6 ms saved is
5.4 DMA rate (the removal of staging contention), 2.8 faults and 1.5 staging. So
pinning pays without faults too, **on this host's memory system**. EXP-0005
repeated the split with 80% of the expert bytes pinned (98% of warm upload
bytes): 86% of the 16.52 ms warm gain is DMA rate and fast-copy idle, 13%
faults. What still needs a host with more RAM is whether that DMA-rate part
survives where the page cache never misses and memory is less contended.

**To settle:** ROADMAP R4, which needs **a smaller MoE model** whose expert
working set fits comfortably in RAM - something in the 4 to 8 GiB range. The only
MoE model on disk is the 20.82 GiB one; the other three models are dense.

**Blocked on:** acquiring such a model. This is the cheapest unblock in the
project and would also serve Q4.

**Why it matters:** it decides whether the pinning commit is worth proposing
upstream at all. Its +88% headline is still not defensible as a general number,
but the warm split now gives it a mechanism that is not page faults.

## Q4 - Does the frequency victim help a model with different routing?

Every policy result depends on routing skew (hottest 8 of 256 take 31 to 33%).
A model with fewer experts per layer, a different top-k, or flatter routing
should show a smaller effect - possibly none.

**This is the largest external-validity gap in the project.**

**To settle:** ROADMAP R7 - one more MoE model, ideally with a different expert
count and top-k. Needs `moe-bench/layout-*.json` regenerated from its GGUF expert
tensor types and shapes, which the harness already supports.

**Blocked on:** the same missing model as Q3.

## Q5 - Would zero-copy (device access to host memory) beat copying?

Named in the project directive, **never investigated here**: no measurement, no
code, no literature pass.

On this hardware the prior is poor. EXP-0004 measured the copies themselves at
12.8 GB/s pinned and 8.2 to 13 GB/s pageable, and a zero-copy MoE matmul would
read expert weights over the same link at matmul access patterns rather than as
one large sequential DMA. Reading each weight exactly once with no reuse is the
best case, which is what a 0% hit rate already does, but at worse granularity.
It also needs mapped pinned memory: the device reports
`pageableMemoryAccess = 0` and `canUseHostPointerForRegisteredMem = 0`, so the
WDDM pinning cap would bound it too. Its one advantage is that it overlaps by
construction, which EXP-0004 shows the copy path does not.

**Confidence: level 4** - this is reasoning, not evidence.

**To settle cheaply:** a microbenchmark of a single MoE matmul reading its experts
from host memory versus from VRAM, which needs no integration. Worth one
iteration before dismissing it.

## Q7 - Can expert activation be predicted better than by frequency?

Previous-token prefetch was rejected on trace evidence: token-to-token overlap
within a layer is 31 to 37%, below what the hot set already delivers. But that
tests **one** predictor, the weakest plausible one.

Untried: prediction from the hidden state before the router runs, from the
previous *layer's* selection in the same token, or from a learned per-layer
transition matrix built offline from the traces.

**To settle:** entirely offline from existing traces - measure the mutual
information between layer `il`'s selection and layer `il+1`'s in the same token,
and the accuracy of a transition-matrix predictor. **No GPU time, no code in
llama.cpp.** The traces are committed in
`research/results/experiments/EXP-0001/`, so this is now purely analysis.

EXP-0001 re-measured the one predictor that was already rejected: token-to-token
overlap is **31.4 to 38.7%** at a lag of 1 token, slightly above the old 31 to
37% range but still far below the 79 to 85% the hottest 64 experts deliver. The
rejection stands and the untried predictors above are unaffected.

**Why it matters:** a predictor that beats frequency would attack the Belady gap
(Q2) from a different direction, and since EXP-0004 it is also the predictor a
copy stream needs (ROADMAP R13). Its value there is bounded: perfect next-layer
prediction could hide at most 9.4 ms per token (17%), and 70 to 90% recall still
8.9 to 9.2 ms (KNOWLEDGE_BASE.md Finding 10). The traces hold selections, not
hidden states, so they can test selection-based predictors only; the
literature's router-on-current-hidden-state predictor needs a new dump.

## Q8 - What does the prefill actually cost end to end?

Finding 5 records 5.7 GiB of PCIe traffic for a 54-token prompt, but the `-p 512`
throughput sweep (BENCHMARKS.md B4) was never tabulated. For long-context use
the prefill could dominate, and this project has no number for it.

**To settle:** run B4. Cheap. Upstream #30594 is the fix, so this is about sizing
the problem, not solving it.

## Settled questions

Kept so they are not reopened without new evidence.

| question | answer | where |
|---|---|---|
| Does the upstream cache thrash below one token's experts? | Yes, 0.00% hit rate at 0.66 slots per token | Finding 1, level 1 |
| Where is the crossover? | 1.0 slots per token, per cache group | Finding 3, level 1 |
| Should frequency replace recency everywhere? | No - it costs 2 to 3 points above the crossover | Finding 3, level 1 |
| Is counter aging optional? | No - 6 to 9 points over a multi-topic session | FAILED_APPROACHES.md |
| Sample the LRU end or uniformly? | Uniformly: 14.2% vs 1.1% at 256 MiB | Finding 3, level 1 |
| Does the cache change generated output? | No, byte-identical in every configuration **with every expert computed on the GPU**. With the cache off, or a group with no slots, those experts run on the CPU and the text diverges after about 65 to 110 tokens (EXP-0005) | Correctness invariant, level 1 |
| **Q1** - do the simulated rankings hold on `d854d98b`? | **Yes.** Simulator exact to hit and fill counts on all four traces; every figure within +0.45 points; no ranking changed; crossover unmoved | EXP-0001, level 1 |
| **Q6** - is the host-to-device copy overlapped with compute? | **No.** 0 ns of overlap in every token of 15 traced runs; one stream and two host waits per MoE layer. A warm 2048 MiB token is 12.8 compute + 27.1 DMA + 0.1 read-back + 16.1 host waits; overlap could hide at most 9.4 ms of it | EXP-0004, level 1 (level 2 for the mechanism); Findings 4, 9, 10 |
| **Q9** - does `--load-mode none` already do what the pinning commit does? | **Partly, on this host.** It beats address-order pinning at 15000 MiB by 1.14 ms but not profile order. It is about 80% pinned (a 4 GiB chunk silently falls back to pageable at the budget), 38.21 ms warm against 39.35 for address-order pinning and 36.65 for profile order at 15000 MiB (tokens 387-841), and it loads in 15 s instead of 6.3 to 7.5 | EXP-0005, level 1; Finding 11 |
