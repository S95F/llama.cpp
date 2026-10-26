# KNOWLEDGE BASE

Consolidated technical reference for running MoE models larger than VRAM on this
machine. Every claim carries an evidence level; nothing below level 1 may be
quoted as a fact.

## Evidence levels

1. **Reproduced locally** under controlled benchmarks on this machine.
2. **Upstream or published** - implemented in llama.cpp master, or in a
   published technical source.
3. **Third-party report**, not independently reproduced here.
4. **Theoretical** - reasoning or extrapolation only.

Provenance: findings 1 to 6 consolidate `moe-bench/RESULTS.md`, which is the
primary record and holds the full tables. Section numbers below refer to it.
Measurements exist on two bases: `78a8ae1` (RESULTS.md sections 1-11) and
`d854d98b` (section 12, the current base). Where they differ, the `d854d98b`
figure is the one to use.

---

## Finding 1: the upstream expert cache returns nothing below one token's experts

**Level 1.** Reproduced on hardware on both bases.

The MoE expert cache already in master (`src/llama-moe-cache.cpp`,
`--moe-cache-mib`) keeps host-resident experts in a per-device SLRU. Expert
selection during generation is a cyclic sweep over layers 0..39, then back to
layer 0 for the next token. When the cache holds fewer slots than one token
needs, LRU evicts each expert shortly before the next token asks for it: textbook
LRU thrash.

Generation-only hit rate, 255 greedy tokens on `moe-bench/prompt.txt`, from the
cache's own counters on base `d854d98b` (RESULTS.md section 12):

| cache MiB | slots per token | upstream hit | upstream uploaded |
|---:|---:|---:|---:|
| 384 | 0.66 | **0.00%** | 145030.88 MiB |
| 528 | 0.90 | 1.04% | 143422.10 MiB |
| 586 | 1.00 | 20.93% | 114692.98 MiB |

The cache still occupies its VRAM while returning nothing. Upstream already
detects this condition and logs a warning, so the threshold needs no new tuning
constant.

**Key scale for this model: the experts of one token across all 40 layers are
586 MiB.** Cache sizes are therefore also quoted as "slots per token" =
cache slots / experts used by one token.

## Finding 2: routing is strongly skewed, so a small resident hot set pays

**Level 1.** From `moe-bench/trace_stats.py` over four traces, averaged over
layers. Re-measured on `d854d98b` by EXP-0001; the `78a8ae1` ranges are kept for
comparison.

| hottest k experts of a layer (of 256) | `78a8ae1` | **`d854d98b`** | `prompt_d` alone |
|---:|---:|---:|---:|
| 8 | 31 - 33% | **25.9 - 33.2%** | 25.9% |
| 16 | 46 - 47% | **39.1 - 46.8%** | 39.1% |
| 32 | 64 - 66% | **57.0 - 65.7%** | 57.0% |
| 64 | 83 - 85% | **79.1 - 85.1%** | 79.1% |

The hot set exists; recency cannot find it under thrash, frequency can. This is
the mechanism behind Finding 3.

**The wider range is entirely `prompt_d`**, the multilingual translation task;
the other three traces sit inside the old ranges. Even `prompt_d` is strongly
skewed - 79.1% from the hottest 64 of 256 against 25% for uniform routing - and
it is where the sampled victim beats full LFU by the widest margin. The cause is
that greedy decode diverges between the two bases, so the traces cover
**different generated text**, and a multilingual continuation drifts most. Do
not requote the old narrow ranges.

Token-to-token overlap within a layer, which is what rules out a previous-token
prefetch, also shifted up slightly: **31.4 to 38.7%** at a lag of 1 token
(`78a8ae1`: 31 to 37%) and **19.6 to 25.3%** at a lag of 8 (19 to 23%). Still far
below the 79 to 85% the hottest 64 deliver, so FAILED_APPROACHES.md's rejection
stands.

## Finding 3: a frequency-aware victim recovers 8.7 to 26.5 points below the crossover

**Level 1.** Reproduced on hardware on both bases. It must be switched off above
the crossover. **Superseded in part by Finding 7**, which adds a distance term to
the same victim and gains a further 1.74 to 2.94 points; the switch and
everything else below are unchanged.

The branch change (`3db731656`, about 45 lines inside the existing
`moe_cache_slru`): the victim becomes **the least used of 16 uniformly sampled
slots**, with one `uint32_t` use counter per (layer, expert) (40 KiB for this
model) halved every `8 * n_slots` fills. Enabled **only** when
`n_slots < experts of one token`; above that the code path is byte-for-byte the
old one.

On base `d854d98b`, 255 greedy tokens (RESULTS.md section 12):

| cache MiB | upstream hit | branch hit | PCIe traffic |
|---:|---:|---:|---:|
| 384 | 0.00% | **21.32%** | **-21.3%** |
| 528 | 1.04% | **27.53%** | **-26.7%** |
| 586 | 20.93% | **29.64%** | **-11.0%** |

Above the crossover the two arms are **byte-identical** - same hits, same
misses, same uploaded bytes to the hundredth of a MiB at 645, 1024 and 2048 MiB.
That inertness is measured, not assumed, and those rows serve as the control for
every throughput claim.

Three sub-results, all level 1:

- **Sampling must be uniform over the whole cache.** Sampling the LRU end
  recovers 1.1% at 256 MiB against 14.2% for uniform sampling, because the LRU
  end is exactly where the entries about to be reused sit.
- **The crossover is real and sits at 1.0 slots per token.** Forcing frequency on
  everywhere costs 2 to 3 points above it (33.84% vs 35.50% at 1.10, 58.71% vs
  61.65% at 3.49).
- **Counter aging is not optional.** Without it, a 1024-token four-topic session
  loses 6 to 9 points, and at 0.99 slots per token it is *worse* than upstream.

## Finding 4: generation is bandwidth-bound on the host-to-device copy

**Level 1** for the rate figures; **level 4** for the attribution of the pinning
gain, which is confounded on this machine. The pageable path wastes most of the
link.

Dividing the known per-token upload volume by measured time per token gives the
effective rate of the pageable `cudaMemcpyAsync` path. On `d854d98b`, using
measured rather than theoretical bytes (RESULTS.md section 12):

| cache MiB | MiB uploaded per token | tok/s | GB/s |
|---:|---:|---:|---:|
| 384 | 568.7 | 5.71 | 3.40 |
| 528 | 562.4 | 7.24 | 4.27 |
| 586 | 449.8 | 10.07 | 4.75 |
| 645 | 398.4 | 12.15 | 5.08 |
| 1024 | 344.6 | 15.02 | 5.43 |
| 2048 | 227.6 | 22.10 | 5.27 |

So **3.4 to 5.5 GB/s against roughly 14 GB/s of usable PCIe 4.0 x8**. The rate is
near-constant across a 2.6x range of transferred bytes, which is why cutting
bytes converts almost linearly into tokens per second. These are derived lower
bounds, not bus measurements: compute overlaps the copies.

Registering the mmap'd expert pages with `cudaHostRegister` (`c71dee331`,
opt-in, capped at RAM/4, `LLAMA_MOE_PIN_MIB` overrides) targets the driver
staging copy. Sweep at a 1024 MiB cache where the eviction change is inert, on
`78a8ae1` (RESULTS.md section 7):

| pin cap | actually pinned | share of 18760 MiB | tok/s | vs pageable |
|---|---:|---:|---:|---:|
| off | 0 | 0% | 9.32 | - |
| 2048 MiB | 1954 MiB | 10% | 8.47 | **-9%** |
| 4096 MiB | 3986 MiB | 21% | 9.70 | +4% |
| 8192 MiB | 8162 MiB | 44% | 13.61 | **+46%** |
| 16384 MiB | 15266 MiB | 81% | 17.53 | **+88%** |

**Read the +88% as confounded.** Locking the pages both removes the staging copy
and stops the mmap page-cache misses that this 31 GiB machine suffers against an
18.32 GiB working set. Separating the two needs a host with enough RAM to keep
the model cached without pinning, which this one does not have. Low coverage is
mildly counterproductive: it spends RAM without covering enough to pay for
itself. Spread here was +-25 to 45%, the widest of any measurement in the
project.

## Finding 5: prompt processing is uncached by design

**Level 1** for the measurement, **level 2** for the intent.

A 54-token prompt pushed **5696.85 MiB** over PCIe with zero cache reuse
(5707.74 MiB on `d854d98b`). Ubatches above `max_batch = 32` bypass the slot map,
and nothing populates the cache from the prefill. This is deliberate upstream:
large batches touch most experts of a layer and would evict the generation hot
set. Upstream PR #30594 attacks exactly this.

EXP-0001 recorded the large-batch counters for all four prompts, which are all
32 to 41 words. The cost is **not** proportional to prompt length alone:

| prompt | words | large-batch uploads | MiB |
|---|---:|---:|---:|
| prompt (technical prose) | 41 | 6315 | 5707.74 |
| prompt_b (C++ code) | 32 | 5904 | 5153.51 |
| prompt_c (history summary) | 35 | 6462 | 6065.68 |
| prompt_d (translation) | 37 | 6978 | 7003.48 |

`prompt_d` costs 36% more than `prompt_b` for 5 more words, which is the same
flatter routing that widened Finding 2's range: a prompt that touches more
distinct experts pays more in the uncached prefill. The `prompt` row reproduces
the separately measured 5707.74 MiB exactly.

These are generation-run prefills of a short prompt, not the `-p 512` sweep,
which is still untabulated (OPEN_QUESTIONS.md Q8).

## Finding 7: the layer sweep is cyclic, so the next use of a layer is known, not predicted

**Level 1.** EXP-0002, hardware and simulator.

Generation sweeps the layers of a cache group in order, so a layer visited `ago`
calls back comes again in `m - ago` calls, where `m` is the group's layer count.
The layer the sweep just passed is therefore **furthest** from its next use and
the layer just ahead of the sweep is **nearest** - and the nearest is what plain
LRU evicts first, which is the thrash of Finding 1 stated exactly.

Weighting the sampled victim's use count by that distance (maximise `dist/use`,
approximating Belady's "furthest next use" as
`dist(layer) / P(expert picked at that layer)`) beats use count alone:

| MiB | upstream | EXP-0000 policy | **with distance** | gain | traffic vs EXP-0000 |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.00% | 13.40% | **16.34%** | +2.94 | -3.4% |
| 384 | 0.00% | 21.32% | **23.16%** | +1.84 | -2.3% |
| 528 | 1.04% | 27.53% | **29.30%** | +1.77 | -2.4% |
| 586 | 20.93% | 29.64% | **31.38%** | +1.74 | -2.5% |
| 645, 1024, 2048 | - | - | byte-identical | 0 | 0% |

Sign-consistent on all four traces at all eight simulated sizes (+1.06 to +2.73).
Costs 320 bytes for this model and no new tuning constant: the distance comes
from the group's layer count, which the cache already knows.

It is kept **off above the crossover** even though it helps there, because the
gain is a wash rather than a sign-consistent win and inertness above the
crossover is what the control rows depend on. See OPEN_QUESTIONS.md Q10.

Correctness is independent of the cyclic assumption: in a non-cyclic order
(prefill) `dist` becomes an arbitrary but bounded weight, and the "cannot evict
what this request needs" rule and the LRU fall back are untouched. Only the
quality of the choice degrades.

## Finding 8: layers differ enormously in cacheability, and the policy already tracks it

**Level 1.** EXP-0002, from the per-layer breakdown at 528 MiB over four traces.

Per-layer hit rate under the shipping policy ranges **1.84% to 49.04%** (stdev
10.69) and is almost perfectly explained by that layer's routing skew:

| pair | Pearson r |
|---|---:|
| top8 share vs hit rate | **+0.979** |
| top64 share vs hit rate | +0.944 |
| distinct experts vs hit rate | **-0.924** |
| top64 share vs gap to Belady | -0.218 |

The earliest layers are nearly uncacheable: layer 0 routes over 236 of 256
experts with 13.3% from its hottest 8 and gets 1.84%, while layer 20 takes 95.0%
from its hottest 64 and gets 49.04%.

**The gap to Belady, by contrast, is near-uniform** (14.49 to 27.50, stdev 3.23;
the worst 8 of 39 layers hold 24.7% of it against 20.5% for a uniform spread).
Two consequences:

1. **Per-layer slot budgets are not the lever.** A shared pool with a
   frequency-scored victim already allocates slots by cacheability - that is what
   `r = +0.979` means - so there is little left for an explicit budget to win.
2. The residual loss is a **timing** loss shared by every layer, not an
   allocation loss concentrated in some. That is what pointed at Finding 7 and is
   why it worked.

## Finding 6: a large gap to offline optimal remains everywhere

**Level 1** (simulated; the simulator is validated to exact hit and fill counts
against llama.cpp on both bases).

Belady at the same slot count, re-measured on `d854d98b` by EXP-0001:

Simulated, mean of four traces. "branch" is EXP-0000's policy, "+dist" adds
Finding 7:

| cache MiB | upstream | branch | **+dist** | Belady | gap closed by +dist | Belady on `78a8ae1` |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.00% | 15.94% | **18.68%** | 32.01% | 58% | 31.86% |
| 384 | 0.00% | 24.17% | **26.21%** | 43.51% | 60% | 43.35% |
| 528 | 1.18% | 30.42% | **32.33%** | 51.65% | 62% | 51.53% |
| 586 | 24.54% | 32.42% | **34.29%** | 54.07% | 33% | 53.93% |
| 645 | 35.89% | 35.89% | 35.89% | 56.31% | 0% | not tabulated |
| 1024 | 44.67% | 44.67% | 44.67% | 64.96% | 0% | 64.74% |
| 2048 | 61.85% | 61.85% | 61.85% | 77.48% | 0% | 77.30% |

With the distance term the policy closes 58 to 62% of the gap at 256 to 528 MiB,
33% at 586, and none of it above the crossover where it is inert by design.
A large gap remains everywhere - Belady still gets 32.01% where the policy gets
18.68% at 256 MiB. This is the headroom a profile-driven resident hot set
(#30594) aims at.

---

## Architectural facts

Level 2 unless marked. Detail in ARCHITECTURE.md.

- **Master already has the cache.** Per-device SLRU, host slot map per layer,
  uploads on the scheduler's slot-map copy callback. Generation (ubatch <= 32)
  reads through it; larger ubatches bypass it but may still read out of it.
- **The mixed `UD` quantization splits this model into three cache groups**
  (36 layers at 1.812 MiB per expert, 1 at 2.195, 3 at 1.945). Each group gets
  its own slot budget and its own LRU, so a small total budget can leave a group
  with **zero** slots, which falls back to the bypass path. Level 1: this is why
  the 384 MiB row gains less than its traffic saving.
- **18.32 GiB of the 20.81 GiB model is expert weights.** The experts, not the
  dense weights, are what cannot fit in 8 GiB.
- **`-ngl 99 -ncmoe 40` is the only configuration this model fits in** on 8 GiB:
  all experts in host memory, dense weights and attention on the GPU.

## Measurement discipline

Level 1, all learned the hard way. FAILED_APPROACHES.md has the detail.

- **Hit rate and uploaded bytes carry every claim.** Both are exactly
  reproducible for a fixed model, prompt and greedy decode. Throughput is
  corroboration only, and must be reported with its control rows.
- **Never run two benchmark processes at once.** Each maps the whole 20.81 GiB
  model; two exhaust 31 GiB, the run goes disk bound, and every cache size
  returns the same ~6 tok/s, making the cache look inert.
- **Sweeps must be order-balanced.** The second arm inherits a warmer page cache.
  Run in sequence, the frequency policy appeared to gain 4 to 12% even at sizes
  where its hit rate is *lower* by construction.
- **Free host RAM changes the answer.** At ~18 GiB free the control rows span
  +-1.3%; at 16.1 GiB they span +3.7% to -3.9% and absolute throughput is 10 to
  15% lower. Record free RAM with every throughput number.
- **Do not time with `llama-completion -v`.** The per-token debug logging hides
  the effect. Use `llama-bench`.
- **The simulator is trustworthy, and now validated on the current base.**
  `moe-bench/moe_sim.py` matched llama.cpp to the digit on `78a8ae1` (60.82% hit,
  58867.3 vs 58867.26 MiB uploaded, 2044.1 vs 2044.11 MiB VRAM). EXP-0001
  re-validated it on `d854d98b` on **all four traces**, to the integer hit and
  fill counts (prompt: 50092 / 31516 in both). Validation must be redone on any
  future rebase: the traces are base-specific even though the code is not.
- **Compare policies paired per trace, and require the sign to agree on all four
  traces.** Level 1, from EXP-0001. The spread of one policy **between** traces
  is 4.2 to 7.3 points, which is larger than several of the differences being
  compared, so a positive mean is not evidence. Full LFU at 256 and 384 MiB
  would pass a mean test while three of four traces disagree with it.

## Resolved: the simulated tables now hold on the current base

**Level 1.** EXP-0001 settled this. Upstream's own commits between `78a8ae1` and
`d854d98b` changed the router's top-8 at some positions: with
`--moe-cache-mib 0`, where the cache is never constructed, the two bases generate
different text from the same greedy decode. Consistent with `c49edf096` (fused
FFN gate/up matmuls and GLU in MMQ) and `a55c16e4e` (fused MRoPE and set-rows)
changing reduction order and therefore rounding, but intermediate commits were
not built, so the attribution is **level 4**.

The four traces were re-collected on `d854d98b` and the simulator re-validated.
**Every simulated figure in this document reproduces within +0.45 points**, every
delta in the same direction, and no policy ranking changed. Traffic reductions
reproduce within 0.4 points. The crossover did not move.

Simulator agreement is now established more strongly than before: at 2048 MiB,
where the branch is inert and the policy fully deterministic, the simulator
reproduces llama.cpp's **integer hit and fill counts** on all four traces
(prompt: 50092 / 31516 in both), not merely the ratios.

Note that an expert-selection change does not move uploaded bytes at a 0% hit
rate - every selected expert is uploaded, 8 per layer per token, and expert size
depends on the layer not on which expert. That is why the 384 MiB upstream row is
identical to the last digit across the two bases.

## Correctness invariant

**Level 1.** The cache only moves expert weights into VRAM; the kernels and the
values are unchanged, so generation must be bit-identical.
`moe-bench/check-correctness.sh` greedily decodes the same prompt under every
configuration and diffs. All configurations agree byte for byte on both bases:
policy on and off, above and below the crossover, host page pinning, and (on
`78a8ae1`) upstream master with and without the cache. Any experiment that
changes output is a bug, not a trade-off.
