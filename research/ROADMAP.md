# ROADMAP

Prioritized optimization opportunities. Reordered after every experiment, not
followed blindly. Each item carries an estimate of gain, complexity, risk and
cost, and names the evidence that motivates it.

Last reprioritized: 2026-10-26, after EXP-0001.

## Done

### R1 - Re-collect the four traces on `d854d98b` and re-validate the simulator

**EXP-0001, Adopt.** Simulator validated to exact hit and fill counts on all four
traces; every simulated figure reproduces within +0.45 points and no ranking
changed. Raw results committed. This unblocked R2 and R3, and it added the paired
per-trace screening rule that both must now use (BENCHMARKS.md).

## Priority 1

### R2 - Close more of the Belady gap below the crossover

| | |
|---|---|
| Motivated by | Finding 6, re-measured by EXP-0001: the branch closes 50 to 58% of the Belady gap at 256 to 528 MiB but only **27% at 586 MiB**, and 0% above the crossover |
| Hypothesis | Below one token's experts, a victim that combines frequency with the known cyclic structure of the sweep beats frequency alone, because the access pattern is not an arbitrary skewed stream - the next access to layer `il` is almost exactly one token away. |
| Falsifiable by | No candidate whose paired per-trace delta against `ramp-s16-age8` has the same sign on all four traces at 256 to 586 MiB. |
| Expected gain | Belady leaves 16 to 27 points on the table below the crossover. Half of that would be a large win at the measured near-linear bytes-to-tokens conversion. Realistically less: Belady is offline. |
| Complexity | Medium. Sampling already exists; a structure-aware score is a change inside the same function. |
| Risk | Medium. Must stay inert above the crossover and must not change output. |
| Cost | Low - the simulator is validated and screens candidates with no GPU time. |

**Screening rule, from EXP-0001:** compare paired per trace and require the sign
of the delta to agree on all four traces. One policy varies by 4.2 to 7.3 points
between traces, so a positive mean is not evidence.

Concrete candidates, reordered after EXP-0001:

1. **Per-layer slot budgets.** The budget is currently split by cache group, not
   by layer. **Do the Q2 per-layer breakdown first** - it is pure offline
   analysis on the committed traces and it decides between this candidate and the
   next. Promoted to first because EXP-0001 found the gap is *not* uniform across
   cache sizes (27% closed at 586 against 58% at 528), which hints the loss is
   structured rather than spread.
2. **Frequency plus position in the sweep.** Prefer evicting an expert of a layer
   the sweep has just passed, since it will not be read again for a whole token.
3. **Admission control.** Do not admit a one-off expert at all. EXP-0001 sharpens
   the case: S3-FIFO is a **wash** with the branch from 528 to 1024 MiB and only
   beats it at 2048 MiB, where the branch is inert anyway and `slru-ramp` beats
   S3-FIFO. So admission is worth isolating from the FIFO structure, but the
   expected gain is smaller than EXP-0000 implied.

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
