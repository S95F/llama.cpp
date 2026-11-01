# SOURCES

Upstream commits, PRs and references, with dates where known and a note on
applicability. **Confidence levels** are those of KNOWLEDGE_BASE.md.

## Caveat on currency

`git fetch upstream` **failed in the 2026-10-25 session** (exit 53, no network)
and worked on 2026-10-31, when the branch merged upstream `a91eab67d` (merge
`2ca33f0d7`). The PR states in the first table were recorded during the
2026-10-24 survey.
**EXP-0004 re-checked them on 2026-10-31** through the GitHub web pages, not
through `git fetch`; states that changed are marked. Re-verify before relying on
any of them.

## In-tree, already merged

| ref | what | applicability |
|---|---|---|
| `src/llama-moe-cache.cpp`, `--moe-cache-mib` | The MoE expert cache itself: per-device SLRU, host slot map per layer, uploads on the scheduler's slot-map copy callback | **This is the thing being optimized.** Level 2, and level 1 for everything measured about it |
| #30112 | multi-GPU expert cache | Merged. Cannot be exercised here - one GPU |
| `c49edf096` | CUDA: fuse FFN gate/up matmuls and GLU in MMQ | In the `78a8ae1`..`d854d98b` range. **Suspected** cause of the router top-8 perturbation between the two bases (level 4 - intermediate commits were not built) |
| `a55c16e4e` | ggml-cuda: fuse qwen MRoPE + set-rows | Same range, same suspicion, same confidence |
| `f5222f25c` (#31188) | CUDA: fuse MoE (MUL_MAT_ID) gate/up and GLU in MMQ | In the `63b4b303b`..`a91eab67d` range. B1 and B3 unchanged across it, and its op tests pass here (`research/results/baseline/rebase-2ca33f0d7/`); whether it fires for this model was not checked |
| `ggml_backend_cuda_host_buffer_type_get_max_size`, `ggml_cuda_host_malloc` | The CUDA host buffer type caps a buffer at 4 GiB, and a failed `cudaMallocHost` falls back to a pageable CPU buffer, logged at debug level | Why `--load-mode none` ends up about 80% pinned here (EXP-0005). Level 2 for the code, level 1 for the effect |
| `GGML_OP_OFFLOAD_MIN_BATCH` (`ggml-cuda.cu`, default 32) | Ops on host weights are offloaded to the GPU only at this batch size or above | With the cache off, decode computes the experts on the CPU; B3 sets it to 1 so every configuration computes on the GPU (EXP-0005) |

## Open upstream PRs surveyed before writing code

Checked 2026-10-24 to avoid duplicating existing work; states re-checked on
2026-10-31 (EXP-0004), changes marked.

| PR | what | relation to this work |
|---|---|---|
| #30848 | small-cache protected-set ramp, by another contributor | **Applied locally** as `0955c91b2` so the policy work could be measured on top of it. **Must not be resubmitted.** It explicitly leaves the small-cache policy alone ("performance may suffer") - that sentence is what motivated EXP-0000's main hypothesis. **Closed unmerged on 2026-10-27**, so upstream master still has the thrashing behaviour of Finding 1 |
| #30594 | profile prefilling and zero-upload inference | Attacks Finding 5 (prefill is uncached) and the Belady gap of Finding 6. **Prefer this over a local reimplementation** - ROADMAP R9 |
| #30966 | run missed experts on CPU | Orthogonal: changes what happens on a miss rather than the miss rate |
| #29963 | pipeline parallelism with host experts | Orthogonal, needs multiple devices |
| #30205 | per-device cache sizes | Orthogonal here - one GPU |
| #31139, #31095 | cache group splitting and allocation limits | **Directly touches the group budget split** that ARCHITECTURE.md describes, and therefore the "slots per token" arithmetic the crossover test uses. Watch for conflicts |
| #21067 | tensor prefetch | Adjacent to the rejected previous-token prefetch of FAILED_APPROACHES.md, but at a different level. Still an open draft on 2026-10-31: whole-tensor prefetch on a second backend instance, needs pinned memory, helps only at large ubatch because it cannot know the next layer's experts |

Added by the EXP-0004 survey, states as seen on 2026-10-31:

| ref | state | what | relation to this work |
|---|---|---|---|
| #29887 | merged 2026-10-07 | the MoE expert cache itself (LRU, misses only, ubatch <= 32) | The base of everything here. No prefetch |
| #30637 | merged 2026-10-19 | SLRU replaces LRU | The policy EXP-0000 builds on |
| #31421 | open issue | "hit rate drops to 0.6% at one token of slots since #30637" (Metal, another model) | **Independent upstream report of Finding 1.** The natural place to propose EXP-0000 to EXP-0003 |
| #27861 | open | GPU-resident LRU, hits on GPU, misses on CPU, asynchronous lagged inserts | Author reports **synchronous uploads ate the entire win**, consistent with EXP-0004 |
| #28414 | closed 2026-10-14 | lookahead H2D expert prefetch on a second same-device backend with events | Prefill only. The closest existing design for a copy stream |
| #25859 | open issue | offloaded-MoE prefill leaves the GPU idle on serial H2D copies | Same serialization EXP-0004 measures, for prefill. Reports +21% from page-locking |
| #25319 | open | asynchronous scheduler memory copies (backend and event API changes) | Earlier async-copy PRs #17795 and #20793 were merged and then reverted. Any copy-stream change here should build on this rather than around it |
| #31469 | open | out-of-core expert streaming with a VRAM slot pool; overlaps shared-expert compute with miss servicing | The only upstream design found that overlaps anything with the upload during decode |
| #21609 | withdrawn 2026-04-08, never reviewed | N-slot LFRU cache with a FATE-style next-layer-gate prefetch | The only upstream attempt at router-based prediction |

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
| Pinned memory | Implemented and measured (Findings 4 and 11). Near the Windows budget (about 15 GiB), pinning the experts a selection profile uses most takes a warm token from 52.95 to 36.43 ms, 6.8% better than address order (EXP-0005). Whether it pays where nothing faults: ROADMAP R4 |
| Zero-copy access | **Not investigated.** No measurement, no code. OPEN_QUESTIONS.md Q5 |
| PCIe DMA / async transfers | Measured by EXP-0004 (BENCHMARKS.md B7): copies run at 12.8 GB/s pinned and 8.2 to 13 GB/s pageable, and never overlap compute (one stream, two host waits per MoE layer). Q6 settled; a copy stream is ROADMAP R13 |
| Expert activation prediction / prefetch | Previous-token prefetch **rejected on trace evidence** (31 to 37% overlap). Learned prediction untried, Q7 |
| Reducing redundant copies | The staging copy is the one identified; pinning removes it for 98% of the warm upload bytes in EXP-0005 |
| Constrained PCIe x4/x8 | x8 only, **cannot be varied on this hardware** |
| Overlapping CPU exec, GPU compute, transfers | **Measured: none** (EXP-0004). Overlap could hide at most 9.4 ms of a warm 56.2 ms token (KNOWLEDGE_BASE.md Finding 10). #30966 is the upstream angle; ROADMAP R13 |
| Multi-GPU / heterogeneous placement | **Impossible here** - one GPU. ROADMAP R8 |

"MoE-L2" was named in the directive. **No source was located for it** during the
survey, and it does not appear in the llama.cpp tree or in the PRs above. Not
pursued; if the user has a specific reference, it would be worth adding.

## External literature

**First consulted in EXP-0004 (2026-10-31)**, on hiding or overlapping the
host-to-GPU expert transfer during batch-1 decode. Every entry is **level 3**:
published, not reproduced here. Each citation was checked by a second reader
against the paper; corrections it found are applied below. Several arXiv URLs
now resolve to later versions with different titles or headline numbers; the
figures here are from the version named.

| work | mechanism | result as stated | next-layer prediction accuracy | relevance here |
|---|---|---|---|---|
| Eliseev and Mazur 2023, [arXiv 2312.17238](https://arxiv.org/abs/2312.17238) | LRU expert cache plus speculative loading: the next layer's gate applied to the current hidden state, 1-2 experts fetched | Mixtral-8x7B. Pre-loading on top of LRU: +5% A100, +11% RTX 3060, +19% RTX 3080 Mobile, +34% T4 (ratios of their Table 2) | figure only | **Highest.** Same regime, training-free predictor, cleanest ablation of what prefetch adds over a cache |
| Pre-gated MoE, ISCA 2024, [arXiv 2308.12066](https://arxiv.org/abs/2308.12066) | gate of block N trained to pick block N+1's experts | 1.7x lower MoE block latency than on-demand (A100, Switch) | n/a, routing is the prediction | Low: needs fine-tuning. Shows one block of compute can hide one block of migration on A100 |
| ProMoE, [arXiv 2410.22134](https://arxiv.org/abs/2410.22134) | learned per-layer MLP predictor, chunked preemptible transfers, cached experts computed first | **Inside llama.cpp: 1.09x over an LRU cache on average, up to 1.17x** (RTX 4090, PCIe 4.0) | 84.7% (learned) | **Very high.** The sobering number: the gain over a plain cache inside llama.cpp is small |
| AdapMoE, ICCAD 2024, [arXiv 2408.10284](https://arxiv.org/abs/2408.10284) | next 2-3 layers' gates on current activations; tile-wise copy stream | prefetch alone 1.22x per token (Mixtral 4-bit, RTX 4090) | about 90% | High for the prefetch half; its adaptive gating changes outputs |
| HOBBIT, [arXiv 2411.01433](https://arxiv.org/abs/2411.01433) | llama.cpp-based; next-layer gate prefetch plus low-precision or skipped misses | up to 3.29x vs MoE-Offloading on RTX 4090 | 96% top-1 next layer (Mixtral) | Very high, but most of the gain is from inexact miss handling |
| Fate, [arXiv 2502.12224](https://arxiv.org/abs/2502.12224) | next block's gate run on the CPU in parallel; transfers capped at `n = T / t_expert` | prefetch alone about +30%; 3.7x is the full stack incl. quantization | **78.79% exact** (top-6 DeepSeekMoE), 97.15% only when over-fetching | High: makes the time budget explicit |
| Speculating Experts, [arXiv 2603.19289](https://arxiv.org/abs/2603.19289) | next router on a quasi-hidden state; mispredictions **executed**, not refetched | **Qwen3-30B-A3B, A6000, PCIe 4.0: 9-14% TPOT** vs on-demand | recall@k about 90% beyond early layers | **Very high**: same model class and regime. Even with no refetch the gain is small, and the inexact variant costs accuracy badly |
| SeqMoE, [arXiv 2609.12978](https://arxiv.org/abs/2609.12978) | GPU-resident sequence predictor several tokens and layers ahead; fixed slot pool plus id-to-slot table, copy stream, graph-capturable miss path | **Qwen3.6-35B-A3B** at 40% residency: 75.69% of full-load decode speed; reports llama.cpp at no more than 28% | 88.39% recall@(k+3), Qwen3.6-35B-A3B | **Very high**: this exact model, llama.cpp as a measured baseline, a design that maps onto the existing cache |
| Pre-Attention Expert Prediction, [arXiv 2511.10676](https://arxiv.org/abs/2511.10676) | per-layer predictor on the same layer's pre-attention activation | no end-to-end result | 94.69% exact top-k, Qwen3-30B (first layer, Table 4) | Medium-high for prediction quality; the attention window is about one expert transfer long |
| PROBE, [arXiv 2602.00509](https://arxiv.org/abs/2602.00509) | frozen next router plus a tiny distilled residual | multi-GPU, large batch | **frozen router 70-80% top-k** on 128-expert models; 87-94% distilled | Low for the setting; the accuracy data point matters |
| Fiddler, ICLR 2025, [arXiv 2402.07033](https://arxiv.org/abs/2402.07033) | avoid the transfer: run misses on the CPU | 1.26x vs llama.cpp single-batch | n/a | The alternative strategy upstream #30966 follows |
| SpecMD, [arXiv 2602.03921](https://arxiv.org/abs/2602.03921) | benchmark of prefetch policies; Least-Stale eviction (layer-ordered, stale first) | OLMoE, Mixtral on A100 at ~5 GB/s measured host-to-device | figure only | Medium: also builds eviction on the sequential layer order, like Finding 7. Whether its rule agrees with Finding 7's in direction was not checked; it was evaluated mainly on TTFT |

Also surveyed and judged low relevance for batch-1 decode on one consumer GPU:
MoE-Infinity, MoE-Lightning, ExpertFlow, DuoServe-MoE, SP-MoE, MoE-SpeQ, FloE,
DAOP, HybriMoE, KTransformers, fMoE, DoMoE, EdgeMoE, SiDA, BuddyMoE, cache-
conditional experts, Read-ME, MoEpic, PreScope. The full list with URLs and the
second reader's corrections is in
`research/results/experiments/EXP-0004/literature.json`.

**What the literature says, as it bears on this project.** At batch 1 the
measured value of prefetch alone is single digits to about 30%, because one
layer of compute is about one expert transfer long and a top-8 model misses
several experts per layer. Training-free next-layer prediction is 70 to 96%
accurate depending on the metric (about 70 to 90% for top-k on many-expert
models), worst in the early layers. Pinned host memory
is treated as a prerequisite everywhere. None of the papers measures the
page-fault and pageable-staging costs that EXP-0004 finds dominant in short runs
on this host and still about a quarter of a warm token, probably because those
that report their host run with RAM far larger than the model (ProMoE 128 GB,
HOBBIT 256 GB). The two closest measurements of prefetch alone (ProMoE inside
llama.cpp, Speculating Experts on Qwen3-30B-A3B), 8 to 15% of token time, are
about half to most of the 17 to 18% that EXP-0004 bounds overlap at for this
model (KNOWLEDGE_BASE.md Finding 10).

**First consulted in EXP-0006 (2026-10-31)**, on deciding what to pin or keep
from a profile of expert selections. All **level 3**, read once from the
project pages, not checked by a second reader and not reproduced here:

| work | mechanism | relevance here |
|---|---|---|
| MoE-Infinity, [GitHub](https://github.com/TorchMoE/MoE-Infinity), [arXiv 2401.14361v2](https://arxiv.org/html/2401.14361v2) | traces expert activation per request (an Expert Activation Matrix) and uses it to prefetch and to choose what stays cached | Closest published idea to an online profile: the request's own early activations steer what is kept. Built for serving, not for page pinning |
| neurall's llama.cpp fork, [releases](https://github.com/neurall/llama.cpp/releases) | lifetime per-expert activation counts saved at server stop; `cudaHostRegister` pinning | A persistent file profile, the counterpart of `LLAMA_MOE_PIN_PROFILE` across sessions |
| moe-l2, [PyPI 0.7.0](https://pypi.org/project/moe-l2/0.7.0/) | pins the top 100 experts per layer in advance, from a router map learned from gating traces | A per-layer quota where this project ranks globally under one cap |
| GenerelSchwerz's llama.cpp wiki, [Notable Runs](https://github.com/GenerelSchwerz/llama.cpp/wiki/Notable-Runs) | partial pinning of host expert memory, measured runs | Anecdotal runs of the same kind of partial pinning |

As far as these pages show, none of them reports how well a profile from one
prompt domain transfers to another, which is what EXP-0006's H3 measures.

The classical results the work leans on are textbook rather than cited:

- **LRU thrash under cyclic access** - the standard pathological case for LRU,
  and exactly what Finding 1 measures.
- **Belady's optimal replacement** - used as the offline upper bound in Finding 6,
  implemented in `moe-bench/moe_sim.py`.
- **Sampled approximate LFU** - the same idea as Redis's `maxmemory-policy
  allkeys-lfu` sampling, arrived at independently here; no implementation was
  consulted.
- **S3-FIFO** - tried and rejected as a whole-cache policy
  (FAILED_APPROACHES.md). Its admission idea survives as a ROADMAP R2 candidate.

EXP-0004's survey above covers next-layer prediction accuracy (70 to 96%
depending on the metric). What remains is measuring selection-based predictors
on this model's traces, recorded as OPEN_QUESTIONS.md Q7.

## How to re-verify the upstream state

```sh
git fetch upstream
git log --oneline a91eab67d..upstream/master -- src/llama-moe-cache.cpp src/llama-moe-cache.h
gh pr view 30848 --json state,title
```

`a91eab67d` is the upstream commit of the last merge (`2ca33f0d7`).
