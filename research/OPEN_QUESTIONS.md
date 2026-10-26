# OPEN QUESTIONS

Unresolved questions, missing evidence, and what it would take to settle each.
Ordered by value, not by age.

## Q1 - Do the simulated policy rankings still hold on `d854d98b`?

**Status: blocking.** Every simulated table in KNOWLEDGE_BASE.md (Findings 2 and
6, and the whole of FAILED_APPROACHES.md's policy comparisons) rests on four
traces collected on `78a8ae1`. Upstream's own commits changed the router's top-8
at some positions between the two bases, which is measured (level 1) even though
its attribution to specific commits is not (level 4).

**To settle:** ROADMAP R1 - re-collect the four traces, re-validate the simulator
bit-exactly, re-run the policy sweep. This is EXP-0001.

**What hangs on it:** the ranking could in principle change, but the *mechanism*
(routing skew plus a cyclic sweep) is base-independent, so the expected outcome
is "unchanged within the 0.8-point agreement margin". The point of asking is that
until it is checked, nothing simulated is quotable.

## Q2 - How much of the Belady gap is actually reachable online?

Below the crossover the branch gets 15.8 to 32.0% where Belady gets 31.9 to
53.9%. Belady is offline and therefore not a target, but the *shape* of the gap
is informative: is the remaining loss concentrated in a few cold layers, or
spread evenly?

**To settle:** per-layer hit-rate and per-layer reuse-distance breakdown from an
existing trace. `trace_stats.py` reports skew averaged over layers; a per-layer
version is a small addition and needs **no GPU time**.

**Why it matters:** it decides between ROADMAP R2's three candidates. If the loss
is concentrated, per-layer budgets (candidate 1) are the lever; if it is spread,
a better victim score (candidate 2) is.

## Q3 - Is the pinning gain real once page residency is controlled for?

Finding 4's +88% is the sum of two effects and **cannot be separated on this
host** (31.11 GiB RAM against an 18.32 GiB expert working set; 14 GiB free at
last check).

**To settle:** ROADMAP R4, which needs **a smaller MoE model** whose expert
working set fits comfortably in RAM - something in the 4 to 8 GiB range. The only
MoE model on disk is the 20.82 GiB one; the other three models are dense.

**Blocked on:** acquiring such a model. This is the cheapest unblock in the
project and would also serve Q4.

**Why it matters:** it decides whether the pinning commit is worth proposing
upstream at all. Right now its headline number is not defensible.

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

On this hardware the prior is poor. Finding 4 measures the pageable copy path at
3.4 to 5.5 GB/s against ~14 GB/s of link, and a zero-copy MoE matmul would read
expert weights over the same link at matmul access patterns rather than as one
large sequential DMA. Reading each weight exactly once with no reuse is the best
case, which is what a 0% hit rate already does, but at worse granularity.

**Confidence: level 4** - this is reasoning, not evidence.

**To settle cheaply:** a microbenchmark of a single MoE matmul reading its experts
from host memory versus from VRAM, which needs no integration. Worth one
iteration before dismissing it.

## Q6 - Is the host-to-device copy actually overlapped with compute?

Finding 4 notes that compute overlaps the copies, which is why the derived GB/s
is a lower bound - but **the overlap was never measured**. If it is poor, there
may be a win in the scheduling rather than in the bytes.

**To settle:** CUDA stream timeline for one generation step, or the cheap version
first: compare measured time per token against
`max(compute_time, bytes/measured_bandwidth)` and see which bound is active.

**Why it matters:** this is the one remaining way the bandwidth-bound conclusion
of Finding 4 could be incomplete. If transfers are serialized with compute, the
fix is overlap, not fewer bytes, and that would reprioritize the whole roadmap.
**This is the highest-value question that needs no new hardware.**

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
llama.cpp.** The traces needed are the ones R1 collects anyway.

**Why it matters:** a predictor that beats frequency would attack the Belady gap
(Q2) from a different direction. The cost of asking is low enough that it should
ride along with EXP-0001.

## Q8 - What does the prefill actually cost end to end?

Finding 5 records 5.7 GiB of PCIe traffic for a 54-token prompt, but the `-p 512`
throughput sweep (BENCHMARKS.md B4) was never tabulated. For long-context use
the prefill could dominate, and this project has no number for it.

**To settle:** run B4. Cheap. Upstream #30594 is the fix, so this is about sizing
the problem, not solving it.

## Q9 - Does `--load-mode none` already do what the pinning commit does?

RESULTS.md leaves this open explicitly. `--load-mode none` puts all experts in a
pinned host buffer, at the cost of pinning the full 18.3 GiB and reading the whole
file at startup.

**To settle:** ROADMAP R5. One existing flag, no code.

**Caution:** pinning 18.3 GiB of 31.11 GiB with 14 GiB free may drive the host
into swap. Check free RAM first and be ready to abort.

**Why it matters:** if it matches capped in-place pinning, the pinning commit has
no upstream value.

## Settled questions

Kept so they are not reopened without new evidence.

| question | answer | where |
|---|---|---|
| Does the upstream cache thrash below one token's experts? | Yes, 0.00% hit rate at 0.66 slots per token | Finding 1, level 1 |
| Where is the crossover? | 1.0 slots per token, per cache group | Finding 3, level 1 |
| Should frequency replace recency everywhere? | No - it costs 2 to 3 points above the crossover | Finding 3, level 1 |
| Is counter aging optional? | No - 6 to 9 points over a multi-topic session | FAILED_APPROACHES.md |
| Sample the LRU end or uniformly? | Uniformly: 14.2% vs 1.1% at 256 MiB | Finding 3, level 1 |
| Does the cache change generated output? | No, byte-identical in every configuration | Correctness invariant, level 1 |
