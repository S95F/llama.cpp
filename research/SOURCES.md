# SOURCES

Upstream commits, PRs and references, with dates where known and a note on
applicability. **Confidence levels** are those of KNOWLEDGE_BASE.md.

## Caveat on currency

`git fetch upstream` **failed in the 2026-10-25 session** (exit 53, no network).
Local `upstream/master` is `d854d98ba`, which equals the branch base. Every PR
state below is as recorded during the 2026-10-24 survey and was **not re-checked
since**. Treat the states as of 2026-10-24 and re-verify before relying on any
of them.

## In-tree, already merged

| ref | what | applicability |
|---|---|---|
| `src/llama-moe-cache.cpp`, `--moe-cache-mib` | The MoE expert cache itself: per-device SLRU, host slot map per layer, uploads on the scheduler's slot-map copy callback | **This is the thing being optimized.** Level 2, and level 1 for everything measured about it |
| #30112 | multi-GPU expert cache | Merged. Cannot be exercised here - one GPU |
| `c49edf096` | CUDA: fuse FFN gate/up matmuls and GLU in MMQ | In the `78a8ae1`..`d854d98b` range. **Suspected** cause of the router top-8 perturbation between the two bases (level 4 - intermediate commits were not built) |
| `a55c16e4e` | ggml-cuda: fuse qwen MRoPE + set-rows | Same range, same suspicion, same confidence |

## Open upstream PRs surveyed before writing code

Checked 2026-10-24 to avoid duplicating existing work. States not re-verified.

| PR | what | relation to this work |
|---|---|---|
| #30848 | small-cache protected-set ramp, by another contributor | **Applied locally** as `0955c91b2` so the policy work could be measured on top of it. **Must not be resubmitted.** It explicitly leaves the small-cache policy alone ("performance may suffer") - that sentence is what motivated EXP-0000's main hypothesis |
| #30594 | profile prefilling and zero-upload inference | Attacks Finding 5 (prefill is uncached) and the Belady gap of Finding 6. **Prefer this over a local reimplementation** - ROADMAP R9 |
| #30966 | run missed experts on CPU | Orthogonal: changes what happens on a miss rather than the miss rate |
| #29963 | pipeline parallelism with host experts | Orthogonal, needs multiple devices |
| #30205 | per-device cache sizes | Orthogonal here - one GPU |
| #31139, #31095 | cache group splitting and allocation limits | **Directly touches the group budget split** that ARCHITECTURE.md describes, and therefore the "slots per token" arithmetic the crossover test uses. Watch for conflicts |
| #21067 | tensor prefetch | Adjacent to the rejected previous-token prefetch of FAILED_APPROACHES.md, but at a different level |

## This project's own closed PRs

| PR | what | why closed |
|---|---|---|
| #31121 | `-mcmoe` llama-bench axis (now `111d8372f`) | Closed by the automated checker on **process grounds**: PR template, two open PRs from a new contributor, AI-written description. **Never reviewed on technical grounds** |
| #31122 | host expert page pinning (now `c71dee331`) | Same |

Recorded so that neither is mistaken for a technical rejection, and neither for a
reason to resubmit without fixing the process problems. `AGENTS.md` forbids
automated PR submission in this repository.

## Techniques named in the project directive

Status of each on this machine, so that none is mistaken for untried-but-easy.

| technique | status |
|---|---|
| MoE expert offloading | In use (`-ncmoe`), level 1, the basis of everything here |
| Adaptive GPU caching / eviction policies | **The main line of work.** EXP-0000, level 1 |
| Pinned memory | Implemented, measured, **confounded** (Finding 4). ROADMAP R4 |
| Zero-copy access | **Not investigated.** No measurement, no code. OPEN_QUESTIONS.md Q5 |
| PCIe DMA / async transfers | Only indirectly: pinning is the prerequisite for a genuinely asynchronous copy, but no overlap measurement was made. Q6 |
| Expert activation prediction / prefetch | Previous-token prefetch **rejected on trace evidence** (31 to 37% overlap). Learned prediction untried, Q7 |
| Reducing redundant copies | The staging copy is the one identified; see pinning |
| Constrained PCIe x4/x8 | x8 only, **cannot be varied on this hardware** |
| Overlapping CPU exec, GPU compute, transfers | **Not measured.** #30966 is the upstream angle. Q6 |
| Multi-GPU / heterogeneous placement | **Impossible here** - one GPU. ROADMAP R8 |

"MoE-L2" was named in the directive. **No source was located for it** during the
survey, and it does not appear in the llama.cpp tree or in the PRs above. Not
pursued; if the user has a specific reference, it would be worth adding.

## External literature

**None consulted.** Every finding in KNOWLEDGE_BASE.md is level 1 (locally
reproduced) or level 2 (upstream code). The classical results the work leans on
are textbook rather than cited:

- **LRU thrash under cyclic access** - the standard pathological case for LRU,
  and exactly what Finding 1 measures.
- **Belady's optimal replacement** - used as the offline upper bound in Finding 6,
  implemented in `moe-bench/moe_sim.py`.
- **Sampled approximate LFU** - the same idea as Redis's `maxmemory-policy
  allkeys-lfu` sampling, arrived at independently here; no implementation was
  consulted.
- **S3-FIFO** - tried and rejected as a whole-cache policy
  (FAILED_APPROACHES.md). Its admission idea survives as a ROADMAP R2 candidate.

A literature pass on expert-activation prediction would be worth an iteration;
recorded as OPEN_QUESTIONS.md Q7.

## How to re-verify the upstream state

```sh
git fetch upstream
git log --oneline d854d98ba..upstream/master -- src/llama-moe-cache.cpp src/llama-moe-cache.h
gh pr view 30848 --json state,title
```

Network was unavailable in the 2026-10-25 session, so this is the first thing to
run when it returns.
