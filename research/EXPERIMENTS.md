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

Next: **EXP-0002**, ROADMAP.md item R2 - close more of the Belady gap below the
crossover, starting with the per-layer breakdown that OPEN_QUESTIONS.md Q2 needs.
Not started; awaiting authorization.

## Dependency graph

Which experiment each later one rests on, so that invalidating one shows what
else falls with it.

```
EXP-0000  (frequency victim below the crossover; pinning inconclusive)
   |
   +-- EXP-0001 (R1: simulator revalidated on d854d98b; traces committed)  DONE
          |
          +-- R2 (close the Belady gap)   <- screens with the validated simulator
          |      +-- Q2 per-layer breakdown decides which R2 candidate
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

## Conventions

- One experiment per hypothesis, on its own commit or checkpoint.
- Raw results to `research/results/experiments/EXP-XXXX/`, JSON or CSV, never
  overwritten and never silently replaced.
- Correctness gate (BENCHMARKS.md B3) before any performance claim is accepted.
- Negative results get the same write-up as positive ones; see
  FAILED_APPROACHES.md.
- An experiment that changes generated output is a bug, not a trade-off.
