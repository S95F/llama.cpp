# ROADMAP

Prioritized optimization opportunities. Reordered after every experiment, not
followed blindly. Each item carries an estimate of gain, complexity, risk and
cost, and names the evidence that motivates it.

Last reprioritized: 2026-10-25, during setup (iteration 0). No experiment has run
under this roadmap yet, so the ordering rests entirely on the findings
consolidated from `moe-bench/RESULTS.md`.

## Priority 1

### R1 - Re-collect the four traces on `d854d98b` and re-validate the simulator

| | |
|---|---|
| Motivated by | KNOWLEDGE_BASE.md "Open caveat on the current base" |
| Hypothesis | The simulator still matches llama.cpp to the digit on the current base, and the policy rankings of Findings 2, 3 and 6 are unchanged by upstream's router perturbation. |
| Falsifiable by | A simulated hit rate that misses the measured one by more than the 0.7 to 0.8 points already attributed to RNG differences, or any reordering of the policy table. |
| Expected gain | None directly. This is **infrastructure**: it unblocks R2 and R3, which both need the simulator to screen policies without GPU time. |
| Complexity | Low. The harness exists; this is four trace collections, one simulator validation run, and one re-run of the policy sweep. |
| Risk | Low. No source change. |
| Cost | High in wall clock, low in risk: four 256-token greedy decodes plus a 2048 MiB validation run. |

Why first: every simulated table in the knowledge base is currently unquotable
for the base we are actually on, and the two experiments below are policy work
that would be screened by the simulator. Doing R2 first would mean validating
each candidate on hardware one at a time, which is far more expensive than fixing
the screen.

This is also the first chance to **commit raw results**. The prior work left no
machine-readable artifacts in the repository: `moe-bench/.gitignore` excludes
`out/` and `*.csv`, so no trace, no JSON and no CSV from any earlier measurement
survives. `research/results/` must hold them from now on.

## Priority 2

### R2 - Close more of the Belady gap below the crossover

| | |
|---|---|
| Motivated by | Finding 6: branch 15.8 to 32.0% against Belady 31.9 to 53.9% |
| Hypothesis | Below one token's experts, a victim that combines frequency with the known cyclic structure of the sweep beats frequency alone, because the access pattern is not an arbitrary skewed stream - the next access to layer `il` is almost exactly one token away. |
| Falsifiable by | No candidate beating `ramp-s16-age8` by more than the simulator's agreement margin at 256 to 586 MiB. |
| Expected gain | Up to 10 to 20 points of hit rate below the crossover if half the remaining gap is reachable; at the measured near-linear bytes-to-tokens conversion that is a large throughput win on this machine. Realistically less: Belady is offline. |
| Complexity | Medium. Sampling already exists; a structure-aware score is a change inside the same function. |
| Risk | Medium. Must stay inert above the crossover and must not change output. |
| Cost | Low once R1 is done - the simulator screens candidates with no GPU time. |

Concrete candidates, in the order they should be screened:

1. **Per-layer slot budgets.** The budget is currently split by cache group, not
   by layer. Layers differ in routing entropy (not yet measured per layer), so a
   budget proportional to each layer's measured reuse could beat a uniform one.
2. **Frequency plus position in the sweep.** Prefer evicting an expert of a layer
   the sweep has just passed, since it will not be read again for a whole token.
3. **Admission control.** Do not admit a one-off expert at all; S3-FIFO's gain at
   large caches came from admission, and FAILED_APPROACHES.md rejected it on
   small-cache erraticness rather than on the idea.

### R3 - Measure whether the crossover switch should be a ramp

| | |
|---|---|
| Motivated by | Finding 3: frequency wins below 1.0 slots per token, loses 2 to 3 points above it; the switch is currently a hard step at 1.0 |
| Hypothesis | Between 1.0 and about 1.2 slots per token a blend beats both pure policies, because thrash decays gradually rather than stopping at exactly one token. |
| Falsifiable by | No blend beating `max(frequency, recency)` at any size in 1.0 to 1.3 slots per token. |
| Expected gain | Small, 1 to 3 points over a narrow band of cache sizes. |
| Complexity | Low. |
| Risk | Low, but it adds a tuning constant, which upstream dislikes and which the current hard switch deliberately avoids. |
| Cost | Low after R1. |

Deliberately ranked below R2: the band is narrow and the gain is small, and the
present design's main virtue with reviewers is that it introduces **no** new
constant.

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
