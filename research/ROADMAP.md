# ROADMAP

Prioritized optimization opportunities. Reordered after every experiment, not
followed blindly. Each item carries an estimate of gain, complexity, risk and
cost, and names the evidence that motivates it.

Last reprioritized: 2026-10-26, after EXP-0002.

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

## Priority 1

### R6 - Raise the sample count below the crossover

| | |
|---|---|
| Motivated by | EXP-0002's screen: `dist s32` beats `dist s16` by +0.39 to +0.87, sign-consistent on all four traces at 256 to 645 MiB |
| Hypothesis | Sampling 32 slots instead of 16 is a free gain below the crossover, where the policy is already active. |
| Falsifiable by | The hardware gain not matching the simulated one within the 0.7 to 0.8 point RNG margin, or a measurable cost in the eviction path. |
| Expected gain | +0.39 to +0.87 points of hit rate, so roughly another 0.5 to 1% of traffic. Small but already measured. |
| Complexity | **One constant.** `n_sample = 16` becomes 32 in the sub-crossover branch. |
| Risk | Low. The eviction loop does twice the sampling work; EXP-0002 argued that cost is negligible against the traffic but did not measure it, and at 32 samples it is worth a look. |
| Cost | Very low - the simulator screen is already done. |

The cheapest remaining item, and the only one whose gain is already measured.
Ranked first for that reason, not because it is the largest.

### R2b - Close the rest of the Belady gap

| | |
|---|---|
| Motivated by | Finding 6 after EXP-0002: the policy now closes 58 to 62% of the gap at 256 to 528 MiB, but Belady still gets 32.01% where it gets 18.68% at 256 MiB |
| Hypothesis | None formed yet. The two structural facts available (routing skew, cyclic sweep) are both now exploited, so the next candidate needs a new source of information. |
| Falsifiable by | n/a until a hypothesis exists. |
| Expected gain | 13 points of hit rate remain at 256 MiB, but the easy structure is spent. |
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
   committed traces. The only remaining candidate with an unexplored information
   source.
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

Deliberately ranked below R2: the band is narrow and the gain is small, and the
present design's main virtue with reviewers is that it introduces **no** new
constant.

EXP-0001 made this slightly less attractive. At 586 MiB (0.99 slots per token)
the sampled victim already wins by 7.88 points, sign-consistent on all four
traces, and at 645 MiB (1.09) it loses by 1.75. The transition is sharp rather
than gradual, which is what a hard switch handles well. The remaining question is
only whether anything useful sits between 0.99 and 1.09, a band this model's
group arithmetic barely resolves.

## Priority 3

### R4 - Separate the pinning gain from page residency

| | |
|---|---|
| Motivated by | Finding 4: the +88% is the sum of two effects and cannot be split on this host |
| Hypothesis | With the page cache fully warm and enough free RAM that the mmap working set never misses, pinning still gives a double-digit gain from removing the staging copy alone. |
| Falsifiable by | Pinning giving no gain once page residency is controlled for. |
| Expected gain | None new - this is **attribution**, which decides whether the pinning commit is worth proposing upstream at all. |
| Complexity | Low in code, high in method: this machine has 31.11 GiB against an 18.32 GiB working set, and only 14 GiB was free at session start. |
| Risk | Low. |
| Cost | Medium. Needs a smaller MoE model whose expert working set fits comfortably in RAM, which is **not currently on disk** - the only MoE model available is the 20.82 GiB one. |

Blocked on acquiring a smaller MoE model. Recorded in OPEN_QUESTIONS.md as Q3.

### R5 - Benchmark `--load-mode none` against capped in-place pinning

| | |
|---|---|
| Motivated by | RESULTS.md section 9: this comparison is explicitly still open |
| Hypothesis | Capped in-place pinning beats `--load-mode none` on this machine, because the latter pins the full 18.3 GiB and reads the whole file at startup. |
| Falsifiable by | `--load-mode none` matching or beating it. |
| Expected gain | None new; it tells us whether the pinning commit duplicates an existing upstream option, which decides its upstream value. |
| Complexity | None - it is an existing flag. |
| Risk | Medium at the machine level: pinning 18.3 GiB of 31.11 GiB with 14 GiB free may drive the host into swap. Check free RAM first and be ready to abort. |
| Cost | Low. |

## Priority 4 - not yet justified

### R6 - Populate the cache from the prefill

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

S3-FIFO is a partial exception: rejected as a whole-cache policy, but its
admission idea survives as R2 candidate 3.
