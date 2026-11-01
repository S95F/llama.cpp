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
| EXP-0004 | 2026-10-31 | The expert uploads overlap compute, so the copy is the bottleneck (Q6) | **Adopt** (measurement; no code change). H1 falsified | overlap 0 ns in every token of 15 traced runs; warm 2048 MiB token 56.2 ms = 12.8 compute + 27.1 DMA + 0.1 read-back + 16.1 idle; cold short pageable runs 55-66% page-fault waits; overlap bound 9.4 ms/token (17%) | [EXP-0004.md](experiments/EXP-0004.md) |
| EXP-0005 | 2026-10-31 | Pinning about 15000 MiB, most-selected experts first, cuts the warm 2048 MiB token to 42 ms or less, and beats address order at the same MiB (R11); a fixed page-cache state makes reps agree within 3% (R10); `--load-mode none` (R5) | **Adopt** (opt-in `LLAMA_MOE_PIN_PROFILE`, upload split at pinned-range edges). R10 falsified in part | warm 52.95 -> 36.43 ms (-31%) at 14924 MiB by profile; profile vs address -2.69 ms (-6.8%) at 15000 MiB, -3.55 ms (-8.1%) at the default cap, same sign both rounds; `--load-mode none` about 80% pinned, 38.21 ms (matched window); correctness 10/10 at 256 tokens with every expert on the GPU, and the old gate found to compare CPU with GPU | [EXP-0005.md](experiments/EXP-0005.md) |
| EXP-0006 | 2026-10-31 | Pinning by the run's own first 64 or 128 decode passes is within 1 ms of the file profile warm and beats address order over the whole run (R17); the file profile transfers worse to an unlike prompt (Q17); a fixed launch headroom makes reps agree (R16, its reproducibility half) | **Adopt** (opt-in `LLAMA_MOE_PIN_ONLINE=<N>`; the fixed-headroom harness). H1-H4 hold, H1 and H2 narrowly | warm vs file profile +0.65, -0.16, +0.83, +0.90 ms; over tokens 3-1022 -0.47 and -0.54 ms vs address order (predicted -2, -1), about 3 s per run behind the file profile at the default cap (6-7 s at 15000 MiB); out of domain 6.2% faster than the file profile warm, 0.64 s ahead end to end; reps within 0.8%, drift slope +0.011; 2.0-12% of the pageable expert copies fault under profile pinning (0.9% address order); correctness 11/11 | [EXP-0006.md](experiments/EXP-0006.md) |
| EXP-0007 | 2026-11-01 | One `PrefetchVirtualMemory` call per layer over the missed experts' pages beats faulting them in one by one (R12); the pageable copies that fault are the ones not read for long (Q18); less launch headroom slows prefill and raises faults (R16b) | **Adopt** (opt-in `LLAMA_MOE_PREFETCH=2`, probed by working set). H1 falsified; H2-H4 hold | gate G: prefetch then copy 0.27x the faulting copy (cold, 3 experts); warm profile -2.59 ms (-6.2%), address order -1.74 (-3.9%), mode 1 -0.96 and +2.03; cold unpinned mode 1 -12.91 ms (-16.5%); 99.9% of the remaining faults are first reads since the load, re-reads essentially never fault; 1 GiB less headroom +0.5-0.6 s prefill, +0.93 ms warm; correctness 14/14 | [EXP-0007.md](experiments/EXP-0007.md) |

Next: **EXP-0008**, not chosen until the next `go`. ROADMAP priority 1 is
R12b (the probed prefetch unpinned, at 15000 MiB and on the story prompt, then
the default) and R20 (read the never-read pageable experts off the decode
thread). Not started; awaiting authorization.

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

EXP-0004  (timeline: no overlap; faults, staging, DMA)  DONE
   |      ^-- rests on EXP-0003's binary and the B1 rebase check only
   +-- EXP-0005 (R11 profile pinning to the budget; R10 in part; R5 as an arm)  DONE
          |      ^-- profile and offline coverage from EXP-0001's traces
          |
          +-- EXP-0006 (R17 online profile; R16 reproducibility half; Q17 one prompt)  DONE
          |      ^-- coverage replay from EXP-0005's simulator on two new traces
          |      |
          |      +-- EXP-0007 (R12 probed batch prefetch; Q18 attribution; R16b one level)  DONE
          |      |      ^-- copy log joined with the selection trace
          |      |      |
          |      |      +-- R12b (probed prefetch unpinned, 15000 MiB, story; the default)
          |      |      +-- R20 (read never-read pageable experts off the decode thread)
          |      |      +-- Q21 (what takes the load's pages)
          |      +-- R18 (hybrid profile: file at load, online re-rank)
          |      +-- R19 (deferred residency cost of registration)
          |      +-- R16b (prefill against launch headroom; one level in EXP-0007)
          +-- R13 (copy stream + next-layer prediction) <-- Q7 supplies the predictor
          +-- R14 (per-layer round trip)
```

R4 (separate pinning from page residency) depends on EXP-0000 and on the
EXP-0004/0005 splits of the pinning gain, not on the simulator, and is blocked
on hardware, see OPEN_QUESTIONS.md Q3. R5 (`--load-mode none`) ran as an arm of
EXP-0005.

**Everything downstream of EXP-0001 inherits its traces.** If the branch is
rebased again, run the B1 2048 MiB router check first (KNOWLEDGE_BASE.md,
Standing practice). Only if it does not reproduce must the traces be
re-collected and the simulator re-validated before any simulated figure from R2
or R3 can be quoted - that is what EXP-0001 had to do for EXP-0000's work. The
check decodes the `prompt` trace only. It reproduced on `a4699400e`,
`5df437a57` and `2ca33f0d7`.

## Validity conditions

Each finding's dependence on this particular setup, so that no result is carried
to a context it was never tested in.

| Finding | Depends on |
|---|---|
| Frequency victim wins below the crossover | **Skewed routing** (hottest 8 of 256 experts take 25.9 to 33.2% of selections, hottest 64 take 79.1 to 85.1%). A model with fewer experts per layer or flatter routing should show less. EXP-0001's `prompt_d`, the multilingual trace, is the flattest and still gives the largest margin over full LFU, which is weak evidence that the effect survives flatter routing. Also depends on the cyclic layer sweep, a property of generation, not of prefill. |
| The crossover sits at 1.0 slots per token | The cyclic sweep. Holds per cache group, not globally. |
| Counter aging is required | Multi-topic sessions. A single-topic session would tolerate no aging; a four-topic 1024-token session loses 6 to 9 points without it. |
| Uploads never overlap compute (Finding 4) | **The code**, not the hardware: one stream and two host waits per MoE layer. Holds for any model and GPU until the scheduling changes. |
| Warm, DMA is the largest part of a token (Finding 9) | **This link** (PCIe 4.0 x8) against **this compute** (RTX 4060 Ti) and **this quantization**, and a cache large enough to be warm. A wider link or a slower GPU moves the balance. |
| Cold short runs are dominated by page faults (Finding 9) | **This host's RAM/working-set ratio** (31.11 GiB against 18.32 GiB) and its disk (about 175 us per 64 KiB fault). A host that keeps the model cached would not show it. Also on the page-cache state left by the previous process. |
| Pageable DMA at 8.2 GB/s when the host stages concurrently | **This host's memory system and driver** under WDDM. Untested elsewhere. |
| Pinning gains | **This host's RAM/working-set ratio**. EXP-0004 split them (cold: 78 to 86% avoided faults; warm: mostly DMA contention), EXP-0005 at 98% pinned (warm: 86% DMA rate and fast-copy idle, 13% faults), but whether pinning pays when nothing faults still needs a host with more RAM. |
| Profile order beats address order (Finding 11) | **A pin budget below the expert bytes** (about 15 of 18.3 GiB here, the Windows pinned budget). With room to pin everything, order does not matter. Also on a profile that covers the run's routing: held out by prompt but from the same domain, and the run prompt is the best covered of four. |
| The online profile matches the file profile warm (Finding 12) | **A file profile from the same kind of text.** Out of domain it was 6.2% faster than the file profile (one prompt, which changed topic and register together). Its up-front cost (about 3 s per 1024 tokens at the default cap, 6-7 s at 15000 MiB) depends on N and on how slow the pageable passes are, so on the page-cache state and the headroom. |
| The pageable remainder under profile pinning faults more per byte (Finding 12) | **This host's RAM/working-set ratio** and the headroom left after pinning. With RAM to keep the whole file cached, the pageable experts would stay resident and the difference should vanish (level 4). |
| Faults left under pinning are first reads; a probed batch prefetch removes them (Finding 13) | **This host's RAM/working-set ratio and the launch headroom** (the file's last 2.1 GiB never stays resident at about 19.5 GiB available, and its head is lost in file order as the run grows), **the file's tensor order**, and **Windows**: `PrefetchVirtualMemory` blocks until its reads finish and the probe uses `QueryWorkingSetEx`; elsewhere `llama_prefetch` is `madvise(MADV_WILLNEED)` and the probe is skipped (untested). With RAM to keep the file cached there would be little to prefetch. One prompt, the default pin cap. |
| Byte-identical output with the cache (B3) | **The same compute device for every expert.** The cache off, or a cache group with no slots, computes those experts on the CPU, whose output diverges from the GPU's within about 65 to 110 tokens. |
| Overlap bound 9.4 ms/token (Finding 10) | The per-layer compute and upload times of this model on this GPU and link; scales with them. |
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
- An experiment that changes generated output, with every expert computed on
  the same device, is a bug, not a trade-off.
