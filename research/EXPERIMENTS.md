# EXPERIMENTS

Chronological index. One row per experiment, newest last. Detailed records in
`research/experiments/EXP-XXXX.md`.

Outcome classes: **Adopt** (reproducible improvement, no unacceptable
regression), **Refine** (promising, justifies a focused follow-up), **Reject**
(no benefit or unacceptable trade-off), **Inconclusive** (insufficient evidence,
with the missing test named).

| ID | Date | Hypothesis | Outcome | Measured | Record |
|---|---|---|---|---|---|
| EXP-0000 | 2026-10-24 to 10-25 | Retrospective: the work that predates this research system | Adopt (eviction policy), Inconclusive (pinning) | hit rate +8.7 to +26.5 points below the crossover, PCIe traffic -11 to -27% | [EXP-0000.md](experiments/EXP-0000.md) |
| EXP-0001 | 2026-10-26 | The simulator still matches llama.cpp on `d854d98b`, and the policy rankings are unchanged | **Adopt** (infrastructure; no perf claim) | simulator exact to hit and fill counts on 4/4 traces; every simulated figure within +0.45 points; no ranking changed | [EXP-0001.md](experiments/EXP-0001.md) |
| EXP-0002 | 2026-10-26 | Weighting the sampled victim by the known cyclic distance to its layer's next visit beats use count alone | **Adopt** | hit rate +1.74 to +2.94 points on hardware, PCIe traffic a further -2.3 to -3.4%, sign-consistent on 4/4 traces at 8/8 sizes, byte-identical above the crossover, correctness 8/8 | [EXP-0002.md](experiments/EXP-0002.md) |
| EXP-0003 | 2026-10-26 | Sampling 32 slots instead of 16 is a free gain below the crossover | **Adopt** | hit rate +0.53 to +0.86 points, traffic -0.76 to -1.12%, within 0.18 of the simulated prediction, byte-identical above the crossover, correctness 8/8 on a new base | [EXP-0003.md](experiments/EXP-0003.md) |

Next: **EXP-0004**, OPEN_QUESTIONS.md **Q6** - whether the host-to-device copy is
actually overlapped with compute. Not started; awaiting authorization.

## Dependency graph

Which experiment each later one rests on, so that invalidating one shows what
else falls with it.

```
EXP-0000  (frequency victim below the crossover; pinning inconclusive)
   |
   +-- EXP-0001 (R1: simulator revalidated on d854d98b; traces committed)  DONE
          |
          +-- EXP-0002 (R2: cyclic-distance victim score)  DONE
          |      ^-- Q2 per-layer breakdown chose the candidate, ruled out budgets
          |      |
          |      +-- EXP-0003 (R6: 32 sampled slots)  DONE
          |      +-- Q10 (distance term above the crossover too?)  <- needs more traces
          +-- R3 (ramp the crossover switch)
          +-- Q7 (layer-to-layer prediction)  <- uses EXP-0001's traces
```

R4 (separate pinning from page residency) and R5 (`--load-mode none`) depend only
on EXP-0000, not on the simulator. R4 is additionally blocked on hardware, see
OPEN_QUESTIONS.md Q3.

**Everything downstream of EXP-0001 inherits its traces.** If the branch is
rebased again, the traces must be re-collected and the simulator re-validated
before any simulated figure from R2 or R3 can be quoted - that is what EXP-0001
had to do for EXP-0000's work.

## Validity conditions

Each finding's dependence on this particular setup, so that no result is carried
to a context it was never tested in.

| Finding | Depends on |
|---|---|
| Frequency victim wins below the crossover | **Skewed routing** (hottest 8 of 256 experts take 25.9 to 33.2% of selections, hottest 64 take 79.1 to 85.1%). A model with fewer experts per layer or flatter routing should show less. EXP-0001's `prompt_d`, the multilingual trace, is the flattest and still gives the largest margin over full LFU, which is weak evidence that the effect survives flatter routing. Also depends on the cyclic layer sweep, a property of generation, not of prefill. |
| The crossover sits at 1.0 slots per token | The cyclic sweep. Holds per cache group, not globally. |
| Counter aging is required | Multi-topic sessions. A single-topic session would tolerate no aging; a four-topic 1024-token session loses 6 to 9 points without it. |
| Generation is bandwidth-bound | **This link** (PCIe 4.0 x8) against **this compute** (RTX 4060 Ti) and **this quantization**. A wider link or a slower GPU moves the balance. |
| Pinning gains | **This host's RAM/working-set ratio** (31.11 GiB against 18.32 GiB). Permanently confounded here. |
| Byte-identical inertness above the crossover | Construction, not measurement luck - the branch sets `n_sample = 0` there. Holds for any model. |
| The cyclic-distance victim score (Finding 7) | **The layer sweep being cyclic**, which is a property of generation in any MoE model, so this is the least model-specific result here. It still rides on the same skew as Finding 3, since the score is `dist/use`. It does nothing for prefill, which bypasses the cache. |
| Per-layer budgets being pointless (Finding 8) | **A shared slot pool per cache group** plus a frequency-scored victim. A model whose layers were in separate pools, or a policy without use counters, could still gain from explicit budgets. |

## Conventions

- One experiment per hypothesis, on its own commit or checkpoint.
- Raw results to `research/results/experiments/EXP-XXXX/`, JSON or CSV, never
  overwritten and never silently replaced.
- Correctness gate (BENCHMARKS.md B3) before any performance claim is accepted.
- Negative results get the same write-up as positive ones; see
  FAILED_APPROACHES.md.
- An experiment that changes generated output is a bug, not a trade-off.
