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
Findings 7 and 8 come from EXP-0002 and 0003, findings 9 and 10 and the rewrite
of Finding 4 from EXP-0004, Finding 11 and the rewritten correctness invariant
from EXP-0005, Finding 12, Finding 11's online figures and three
measurement-discipline entries from EXP-0006.
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

## Finding 4: uploads are serialized with compute, and each copy runs below the link rate

**Level 1** (EXP-0004's CUDA timeline), **level 2** for the mechanism (the
code). **Rewritten by EXP-0004.** The earlier version read the effective-rate
table below as link rates and said compute overlaps the copies; **both were
wrong.**

**Nothing overlaps.** Copy/compute overlap is **0 ns in every token** of 15
traced runs. Expert copies, CUDA graphs and kernels share one stream, and per MoE
layer the host runs graph, sync, 32-byte router read-back, sync, expert copies,
next graph. That is 80 syncs per token, 109 `cudaStreamSynchronize` in all with
the input and logits syncs (114 at 256 and 586 MiB). ARCHITECTURE.md "One
generated token, in order" has the code path.

**Measured copy rates** (CUPTI, GPU side of each copy):

| path | GB/s | note |
|---|---:|---|
| pinned | **12.8** | 81% of the 15.75 GB/s raw PCIe 4.0 x8 rate; returns to the host at once |
| pageable, host not staging the next copy | 12.9 to 13.1 | |
| pageable, host staging the next copy | **8.2** | 21.7 of 26.3 ms of DMA per token at 2048 MiB |
| pageable, page not resident | 0.32 to 0.50 (host side) | a hard page fault, about 175 us per 64 KiB; Finding 9 |

So a pageable copy is not slow because of the staging copy as such but because
the host's staging of the next copy slows the current DMA by about 37%; whether
through host memory bandwidth or the driver's path is untested. The "~14 GB/s
usable" figure quoted in earlier documents was never measured.

**Effective rate, uploaded bytes over whole-token time.** On `d854d98b`
(RESULTS.md section 12):

| cache MiB | MiB uploaded per token | tok/s | GB/s |
|---:|---:|---:|---:|
| 384 | 568.7 | 5.71 | 3.40 |
| 528 | 562.4 | 7.24 | 4.27 |
| 586 | 449.8 | 10.07 | 4.75 |
| 645 | 398.4 | 12.15 | 5.08 |
| 1024 | 344.6 | 15.02 | 5.43 |
| 2048 | 227.6 | 22.10 | 5.27 |

The denominator includes compute, page faults, staging and syncs, so these are
**not link rates, only loose lower bounds on the copy rate**. They remain right
as what they are: the rate is near-constant across a 2.5x range of bytes, which
is why cutting bytes converts almost linearly into tokens per second. EXP-0004 confirms
the shape - warm, DMA is the largest single part of a token (Finding 9).

Registering the mmap'd expert pages with `cudaHostRegister` (`c71dee331`,
opt-in, capped at RAM/4, `LLAMA_MOE_PIN_MIB` overrides) makes them pinned. Sweep
at a 1024 MiB cache where the eviction change is inert, on `78a8ae1`
(RESULTS.md section 7):

| pin cap | actually pinned | share of 18760 MiB | tok/s | vs pageable |
|---|---:|---:|---:|---:|
| off | 0 | 0% | 9.32 | - |
| 2048 MiB | 1954 MiB | 10% | 8.47 | **-9%** |
| 4096 MiB | 3986 MiB | 21% | 9.70 | +4% |
| 8192 MiB | 8162 MiB | 44% | 13.61 | **+46%** |
| 16384 MiB | 15266 MiB | 81% | 17.53 | **+88%** |

**Read the +88% as confounded.** Locking the pages both removes the staging copy
and stops the mmap page-cache misses that this 31 GiB machine suffers against an
18.32 GiB working set. Low coverage is mildly counterproductive: it spends RAM
without covering enough to pay for itself. Spread here was +-25 to 45%, the
widest of any measurement in the project.

**EXP-0004 split the gain at 42% pinned** (level 1; cold rows: means of two
128-token reps per arm; warm row: one 1024-token rep per arm):

| regime | pinning saves | page faults | DMA rate | staging |
|---|---:|---:|---:|---:|
| cold, 2048 MiB, 128-token runs | 42.1 ms/token | 36.3 | 3.7 | 2.0 |
| cold, 1024 MiB | 44.6 | 35.0 | 5.6 | 3.8 |
| warm, 2048 MiB, tokens 387-1022 | 9.6 | 2.8 | 5.4 | 1.5 |

Cold, **78 to 86% of the gain is avoided page faults**; warm, most of it is the DMA
rate, which is the removal of staging contention. What still needs a host with
more RAM (OPEN_QUESTIONS.md Q3) is whether pinning pays when nothing faults.
**Pinning is capped near half of RAM on Windows**: the driver charges pinned
pages to a shared budget of **about 15 GiB here** (EXP-0005: `cudaMallocHost`
refused at 15360 MiB after granting 14848 in 512 MiB chunks; 15243 to 15266 MiB
were granted at other times). That is 81% of the expert bytes.

**EXP-0005 split it again with 80% of the expert bytes pinned** (98% of warm
upload bytes) (warm, tokens 387-1022, profile order at 14924 MiB against
pageable): of the 16.52 ms saved, DMA plus fast-copy idle is 14.17 ms (86%) and
slow-copy (fault) idle 2.08 ms (13%). Pinned copies ran at 12.75 to 12.85 GB/s
(11.9 to 12.0 in prof7842); pageable ones at 8.5 to 8.9 GB/s in off, the
address-order arms and none and 10.5 to 11.3 in the profile arms, although
prof7842 makes more pageable copies per token (120, 49 MiB) than addr15000 (112,
45 MiB) or none (107, 41 MiB). Which pages to pin mattered as much as how many:
Finding 11.

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

| MiB | upstream | EXP-0000 policy | with distance | **+ 32 samples** | total gain | traffic vs EXP-0000 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.00% | 13.40% | 16.34% | **17.00%** | +3.60 | -4.1% |
| 384 | 0.00% | 21.32% | 23.16% | **24.03%** | +2.71 | -3.4% |
| 528 | 1.04% | 27.53% | 29.30% | **29.96%** | +2.43 | -3.4% |
| 586 | 20.93% | 29.64% | 31.38% | **31.91%** | +2.27 | -3.2% |
| 645, 1024, 2048 | - | - | byte-identical | byte-identical | 0 | 0% |

Sign-consistent on all four traces at all eight simulated sizes (+1.06 to +2.73).
Costs 320 bytes for this model and no new tuning constant: the distance comes
from the group's layer count, which the cache already knows.

**Sample count (EXP-0003).** Raising the sampled slots from 16 to 32 adds +0.53
to +0.86 of the above. The curve saturates near 48 to 64 - marginal gain per
doubling is +1.40 (8->16), +0.62 (16->32), +0.21 (32->64), **-0.04** (64->96) -
and 32 was chosen as three quarters of the gain for half the work. Sampling
nearly the whole cache still beats exact LFU by over 3 points at 256 MiB, so the
distance term, not sampling breadth, is what avoids the full-LFU behaviour
FAILED_APPROACHES.md rejects.

The default is **not** clamped to `n_slots`. The small cache groups hold only 9
to 22 slots below the crossover, and oversampling them with repeats raises
coverage rather than wasting work (16 draws from 9 slots cover about 83% of them
against 63% for 9 draws). Clamping is the obvious optimization and would make the
policy slightly worse. Note that `LLAMA_MOE_CACHE_SAMPLE` *does* clamp, so the
A/B toggle is not exactly the shipping policy on a small group; EXP-0003 bounded
that asymmetry at 0.02 points.

**Above the crossover the distance term would help, and is still switched off.**
Forced on, hardware gives **+2.10** points at 1024 MiB and -0.15 at 2048
(simulator: +2.12 and -0.36). It is not shipped because the gain is not
sign-consistent across the four traces and because inertness there is what the
control rows depend on. OPEN_QUESTIONS.md Q10.

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

## Finding 9: a generated token is mostly the host waiting - on page faults when cold, on DMA once warm

**Level 1.** EXP-0004, CUDA timelines of every generated token.

GPU compute is **12.7 to 12.9 ms per token** in every configuration, whatever the
cache size or pinning. Everything else is DMA and GPU idle while the host thread
waits:

| 2048 MiB cache | ms/token | compute | DMA | idle | of which page faults |
|---|---:|---:|---:|---:|---:|
| pageable, 128-token fresh process | 139.6 | 12.9 | 26.3 | 100.3 | 91.7 |
| pinned 42%, 128-token fresh process | 97.5 | 12.9 | 22.6 | 62.0 | 55.4 |
| pageable, warm (tokens 387-1022) | **56.2** | 12.8 | 27.1 | 16.1 | 6.7 |
| pinned 42%, warm | **46.6** | 12.8 | 21.8 | 11.9 | 3.9 |
| 586 MiB pageable, warm | **77.7** | 12.8 | 43.5 | 21.3 | 6.3 |

(Every row also carries 0.1 ms of logits read-back.)

**The slow copies are hard page faults on the mmap'd expert pages.** Their host
rate is bimodal with the slow peak at about 175 us per 64 KiB cluster, read
synchronously; their bytes match the disk reads during decode (6.18 against
6.29 GiB at 2048 MiB pageable, per-second correlation 0.999); they fall at the
same positions in repeated runs; and pinned runs fault in proportion to the bytes
left pageable.

**They behave like first touches and die away over about 400 tokens**: in the
1024-token run, 2048 MiB pageable goes from 118.8 ms per token in tokens 3 to
130 to 52.1 to 59.7 from token 387 on. So **a cold 128-token CLI generation is
2.1 to 2.5x slower per token than tokens 387 to 1022 of the same configuration**
(2.5x at 2048 MiB pageable, 2.1x at 586 MiB and with 42% pinned), and in
EXP-0004's state (no eviction, 16.5 to 18 GiB available) the decode of a short
request is dominated by disk, not by PCIe; after an eviction at 19.7 to 21.1 GiB
available it is not (EXP-0005: DMA 26.1 ms against 15.2 ms slow-copy idle,
pageable tokens 3-130).

**Warm, DMA is the largest component (47 to 56%)**, compute 16 to 28%, idle 25 to
29%. The faults never quite stop (2 to 5 per token late in a run, in bursts),
and whether those are first touches or re-faults after the OS dropped a page is
not known.

**The OS keeps part of the model cached between processes** after a pinned run:
pageable runs that followed one prefilled in 3.7 to 7.3 s instead of 13.4 to
13.8 s. That is a measurement hazard (see "Measurement discipline").

**EXP-0005 refines this** (level 1 for the observations, level 4 for the
mechanism). Even with the model evicted before launch (0 to 7 of 256 probed
pages cached), the load's whole-file prefetch read 19.05 to 20.38 GiB before
prefill, and 19.5 to 20.8 GiB had been read in all before decode, so tokens
3-130 were early, not cold: 0.04 to 1.47 GiB of disk reads, and the excess time
there (pageable: slow-copy idle 15.22 against 3.00 ms warm) is not yet split
between hard faults and soft (transition) faults, about 8600 per token pageable
against 135 to 142 with no file-backed experts. Prefill took 1.55 to 3.02 s with
37 to 604 slow copies at 19.7 to 21.1 GiB available, and the slow experts were
the file's unread tail (layers 36 to 39). Hypothesis: prefill faults scale with
available RAM through how much of the file the load leaves unread. It does not
by itself explain EXP-0004's 13.4 s regime, where all 40 layers faulted (ROADMAP
R16 tests it).

## Finding 10: bounds on the remaining levers, warm 2048 MiB

**Level 1** for the per-layer times these rest on, **level 4** for the
projections, which are arithmetic over them. EXP-0004. Base: tokens 387 to 1022
(56.2 ms). On the best window (tokens 771 to 898, 52.1 ms) the overlap bound is
9.3 ms (18%) and full pinning about 34 ms (+53%).

| lever | bound on the gain | condition |
|---|---:|---|
| pin every expert | about **-22 ms** (56.2 -> about 34.6) | needs 18.3 GiB locked; **not possible on this host** |
| pin up to the WDDM cap (about 15000 MiB) | about **-17 to -18 ms** (-> about 38 to 39) | possible now; address order leaves layers 33 to 39 (and layer 32's up tensor) pageable, about 18% of warm traffic |
| overlap upload with compute, perfect next-layer prediction | **-9.4 ms** (17%); -10.4 with unlimited lookahead | copy stream plus prediction; at 70-90% recall still 8.9 to 9.2 |
| the same, after full pinning | about -9 of about 35 (26%) | the levers compound toward about 25 ms |
| per-layer host round trip (sync, read-back, plan, slot-map copy, launch) | **-2.6 to -3.4 ms** | 65 to 86 us per layer; about 10% of a pinned token |

**Overlap's ceiling is not compute's 12.8 ms**: split `S(39)`, which ends in the
LM head, takes 2.1 ms after the token's last upload, 3.3 to 3.5 of 40 layers per
token upload nothing, and a typical layer computes for 270 us against a median
upload of 588 to 654 us, so compute is the short side in about three layer
visits of four. The two closest measurements of prefetch alone (ProMoE 1.09 to
1.17x inside llama.cpp, Speculating Experts 9 to 14% on Qwen3-30B-A3B), 8 to 15%
of token time, are about half to most of this ceiling.

**So on this host the host's memory is the first lever and overlap the second**:
faults and staging contention cost more than overlap could hide, and pinning
also raises overlap's share.

EXP-0005 measured the cap lever: 36.43 ms with profile order at 14924 MiB and
39.35 with address order at 14946 MiB (matched window 387-841, where profile
order is 36.65), against a pageable 52.95 (53.52 on 387-841) in that session.
Profile order beat the projected 38 to 39 ms; address order about met it
(39.35). In gain neither reached the projected 17 to 18 ms (16.52 and 14.17 ms).
The projection was made against another session's 56.2 ms (Finding 11).

## Finding 11: pin the experts a profile selects most, not whole tensors in address order

**Level 1** (EXP-0005: 2048 MiB cache, 1024-token traced runs, two reps per arm
in ABBA order, the model evicted before every run). Level 4 where marked.

Warm, tokens 387-1022, ms per token (addr15000 on the matched window 387-841,
because one of its reps was stopped at token 841):

| arm | pinned | pageable share of upload bytes | ms/token | vs pageable |
|---|---:|---:|---:|---:|
| pageable | 0 | 100% | 52.95 | - |
| address order, default cap (RAM/4) | 7842 MiB | 50.7% | 43.97 | -17.0% |
| profile order, default cap | 7943 MiB | 22.3% | 40.42 | -23.7% |
| address order, 15000 MiB | 14946 MiB | 20.2% | 39.35 (387-841) | -26.5% (387-841) |
| profile order, 15000 MiB | 14924 MiB | **2.3%** | **36.43** (387-841: 36.65) | **-31.2%** |
| `--load-mode none` | about 15240 MiB (14.9 GiB) | 18.5% | 37.93 (387-841: 38.21) | -28.4% |

- **Profile order wins at the same MiB**: -3.55 ms (-8.1%) at the default cap,
  -2.69 ms (-6.8%) at 15000 MiB, the same sign in both rounds and in every
  128-token window. `LLAMA_MOE_PIN_PROFILE=<trace>` counts selections per
  (layer, expert) in a `LLAMA_MOE_CACHE_TRACE` file and pins expert slices, all
  tensors of an expert together, most selected first. Address order pins whole
  tensors from layer 0 up, so it leaves the top layers pageable whatever their
  traffic.
- **The offline replay reproduces the share left pageable** (a replay of the run
  prompt's own routing over tokens 16-255, not a prediction for unseen text).
  Replaying the EXP-0001 traces through the simulator and mapping each miss to
  its GGUF byte range gives the measured share to 0.1 points over the traces'
  255 tokens. Over tokens 387-1022 it drifts by up to 3.1 points.
- **At about 15 GiB pinned the desktop is near its limit.** Address order left
  243 and 814 MiB available (median, warm), and one rep was killed at 205 MiB.
  Profile order left 1764 and 2336 MiB. The pageable experts it leaves are the
  rarely selected ones, so fewer of them are in the working set at any point.
  This is cumulative, not a lower fault rate.
- **Profile order at the default cap does not match address order at 1.9x the
  memory**: +1.35 ms (+3.4%), driven by a tail of slow pageable copies.
- **Costs**: registration takes 0.26 to 0.64 s for 4881 to 7299 ranges, against
  3 to 4 ms for 50 to 96 whole-tensor ranges. Splitting copies at range edges
  adds 12 to 70 small copy calls per token, about 1% of a token.
- **It needs the copy split** (Architectural facts: the copy-edge rule), because
  every expert tensor starts off a page boundary, and neighbouring experts share
  a page.
- **`--load-mode none` is about 80% pinned, not 100%** (Architectural facts),
  and loads in 15 s instead of 6.3 to 7.5.

**Limits.** One run prompt. The profile is held out by prompt but shares its
domain, and the run prompt is the best covered of the four offline (2.1%
pageable at 15000 MiB against 4.6 to 5.7%; at the default cap 20.7% against 26.8
to 29.3%). At 15000 MiB the profile pins every expert its 765 tokens selected
at least twice, but only 207 of the 606 it selected once (8126 of its 8525
selected experts; all of them would need about 15650 MiB); at the default cap
the cutoff is 16 selections, so the default-cap result depends more on the
profile. With room to pin every expert, order would not matter. An online
profile from a run's own first 64 or 128 decode passes leaves 6.7% and 4.2%
pageable at 15000 MiB on the run prompt, tokens 387-1022 (measured in EXP-0006;
the level 4 estimate from four traces was 5.5 to 7.6% and 2.8 to 4.2%). Finding
12 has what that costs and what it gains.

## Finding 12: an online profile matches a matching file profile warm, pays up front, and wins warm out of domain

**Level 1** (EXP-0006: 2048 MiB cache, 1024-token traced runs, two reps per arm
in ABBA order, 19968 MiB available at every launch held by a balloon, a
discarded warm-up run). Level 4 where marked.

`LLAMA_MOE_PIN_ONLINE=<N>` counts the selections of the run's first N decode
passes and registers the most selected experts at the start of pass N+1, up to
the same cap as the other modes. ms per token, mean of two reps:

| arm | pinned share, warm | 387-1022 | 3-1022 |
|---|---:|---:|---:|
| address order, default cap | 49.3% | 45.32 | 48.59 |
| file profile, default cap | 77.7% | 41.73 | 44.53 |
| online, N = 64 | 75.2% | 42.38 | 48.11 |
| online, N = 128 | 79.8% | 41.57 | 48.05 |
| file profile, 15000 MiB | 97.7% | 37.11 | 37.54 |
| online, N = 64, 15000 MiB | 93.3% | 37.94 | 44.59 |
| online, N = 128, 15000 MiB | 95.8% | 38.01 | 45.15 |
| story: address order | 50.9% | 36.71 | 37.68 |
| story: file profile | 65.5% | 36.17 | 37.56 |
| story: online, N = 128 | 88.4% | **33.92** | 37.69 |

- **Warm, it is within 1 ms of a file profile from the same kind of text**
  (+0.65, -0.16, +0.83, +0.90 ms), slower in both rounds in three of the four
  arms, partly a deficit after the registration that fades by about token 640
  (split post hoc; consistent with first touches, not shown directly).
- **It pays up front.** Every pass before the registration is pageable (75-92 ms
  per token), and the register calls take 0.54-0.59 s at the default cap (the
  registering pass 0.62-0.66 s). Over tokens 3-1022 it beats address order by
  about 1% (-0.47, -0.54 ms) and trails the file profile by about 3.5 ms per
  token, 2.7 to 3.2 s per 1024-token run end to end. At 15000 MiB it trails by
  7.05 to 7.6 ms per token (6 to 7 s), with one token stalled for 1.4 to 1.8 s.
- **Out of domain it wins warm.** On a children's story (one prompt, which
  changes topic and register together) the file profile leaves 34.5% of the warm
  bytes pageable against 22.3% on prompt a and gains 1.5% over address order
  (7.9% on prompt a). The online profile leaves 11.6% and is 2.25 ms (6.2%)
  faster than the file profile warm; over tokens 3-1022 the three story arms
  tie, and end to end the online run is 0.64 s ahead of the file profile's
  (break-even near token 1080 on decode alone, extrapolated) and 0.27 s behind
  address order's.
- **The pageable remainder of a profile faults more per byte.** Every slow copy
  (an expert copy below 1.5 GB/s) is pageable. The share of pageable expert
  copies that are slow is 0.9% under address order, 2.0% for the file profile
  on the story, 2.7% at the default cap on prompt a and 12% at 15000 MiB. So
  slow-copy idle does not fall with the pageable share: 3.48 ms per token for
  the file profile against 2.59 for address order, which leaves 28 points more
  pageable. **Level 4**: a profile pins the hot experts, so what stays pageable
  is the cold tail, selected rarely, whose pages leave RAM between uses;
  address order leaves whole layers pageable, hot experts included, and those
  stay resident. Which experts fault was not identified.
- **So the offline replay gives the share but not the milliseconds.** The replay
  reproduces every arm's measured pageable share within 0.2 points (0.1 once a
  one-pass offset in its windows was fixed). Converting share to ms with
  EXP-0005's slopes overestimated every gain: by 0.1 to 0.35 ms for three of the
  H1 arms and by 0.5 to 1.5 ms for H2, on128x15k and the story's file profile.
  Two causes: the extra faults above, and passes before the registration slower
  than the rebuild from EXP-0005's timelines assumed.
- **Registration has a cost that can come later.** `cudaHostRegister` on evicted
  pages can return fast (7.66 GiB in 2-3 ms in address order at load), but a
  GPU-idle stream sync of 0.9 to 2.9 s follows every load-time registration
  before the prompt (at most 53 ms in arms that pin nothing at load). Mid-run at
  15000 MiB one token stalls 1.4 to 1.8 s, inside one register call for N = 64
  (1.5-1.6 s) and at the next stream sync for N = 128 (1.2-1.3 s), with 1.3 to
  1.6 million page faults and 1.5 to 2.1 GiB of disk reads. An untraced run
  without eviction or balloon logged 14.57 GiB registered in 164 ms; whether a
  stall followed was not measured. **Level 4**: paging in the non-resident
  registered pages is the leading cause, the balloon's pressure the second; the
  driver mechanism is unknown.
- **Memory.** At the default cap it pins only experts the run uses, so it uses
  less RAM: warm working set 16457-16539 MiB against the file profile's
  17013-17025 and address order's 18289-18295. At 15000 MiB it uses more (warm
  median available 465-774 MiB against 859-1060), because the pageable experts
  it leaves are selected later and stay resident.
- **Upload fragments are cheap.** Expert-level pinning splits neighbouring
  experts' uploads at page edges: 67 to 84 pinned copies under 4 KiB per token
  at the default cap (30 for the story's online profile, 19 to 27 at 15000 MiB;
  7 under address order), about 0.2 ms of DMA and 0.1 ms of idle per token.

**Limits.** One in-domain and one out-of-domain prompt, two reps, one headroom
level. Five of the six runs at 15000 MiB had a warm median below 1 GiB
available, so those comparisons are headroom-limited. The three arms repeated
from EXP-0005 ran 0.7 to 1.4 ms slower here, mostly in slow-copy idle, at 0.2
to 1.7 GiB less launch headroom; the cause was not tested.

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
- **Uploads are serialized with compute by construction** (EXP-0004): one
  stream, and two host waits per MoE layer (the scheduler's sync before the
  slot-map input, and the cache's sync after the 32-byte router read-back).
  Overlap needs a second stream and, across layers, prediction.
- **CUDA graphs are used** during generation: 41 graph launches per token. A
  token every 256 re-captures and runs most graphs eagerly.
- **The GPU has one copy engine** (`asyncEngineCount = 1`) and runs under WDDM.
- **llama.cpp prefetches the whole model file at load** (`src/llama-mmap.cpp`,
  about lines 611-627), more than the free RAM here, and **the MoE cache never
  prefetches**: `llama_prefetch` and `llama_prefetch_rows` have no caller there.
- **At 256 and 586 MiB one layer gets no cache slots** and runs its MoE on the
  CPU, which costs about 2 ms per token of GPU idle (level 1). 528 MiB too
  (EXP-0005).
- **With the cache off, decode computes the experts on the CPU** (level 1,
  EXP-0005). They are host weights, and the CUDA backend offloads an op on host
  weights only at a batch of 32 or more (`GGML_OP_OFFLOAD_MIN_BATCH`, default
  32). `GGML_OP_OFFLOAD_MIN_BATCH=1` computes them on the GPU instead, uploading
  the selected experts each token.
- **The copy-edge rule** (level 1, EXP-0005, cudart 12.6): a host-to-device copy
  that starts in a `cudaHostRegister`'d range and runs past its end fails with
  `cudaErrorInvalidValue`, synchronous or async. A copy that starts in pageable
  memory and ends inside or runs through a registered range succeeds. Expert
  tensors start off page boundaries, so pinning single experts needs every
  upload split at the ranges' edges (`llama_moe_cache_set_async`).
- **The CUDA host buffer type caps a buffer at 4 GiB, and a chunk that
  `cudaMallocHost` refuses silently becomes a pageable CPU buffer** (level 1 for
  the effect, level 2 for the code). The failure is logged only at debug level,
  and the multi-buffer keeps the `CUDA_Host` name. `--load-mode none` asks for
  19275 MiB in five chunks; here the fourth (layers 25 to 32) fell back.

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
- **A short single-process run measures page faults, not steady state.**
  EXP-0004: a cold 128-token CLI generation is 2.1 to 2.5x slower per token than
  tokens 387 to 1022 of the same configuration. Quote warm windows for steady
  state, and say which one.
- **Page-cache state carries over between processes.** A pageable run after a
  pinned run starts partly warm (prefill 3.7 to 7.3 s instead of 13.4 to 13.8 s,
  and up to 60% fewer page faults). Record the prefill time with every timed run
  together with available RAM; it is not a coldness check by itself (EXP-0005:
  cold-evicted pageable runs prefilled in 2.1 to 3.0 s at 19.7 to 21.1 GiB
  available), and order runs by cache state, not only by arm.
- **Evicting the model before launch is not enough** (EXP-0005). `evict.py`
  (memory pressure, no admin rights) makes the launch state reproducibly cold,
  but the load's prefetch then refills the cache as far as available RAM allows,
  and RAM drifted 1.5 GiB through one session, plausibly (not measured) as the
  evictions trimmed other processes. Reps differed by up to 3.9% warm and 28%
  in tokens 3-130, and r1 was slower than r2 in every arm. Hold headroom fixed
  too, and record available RAM before every run.
- **With a fixed launch headroom and a discarded warm-up run, reps agreed**
  (EXP-0006). A balloon held 19968 MiB available at every launch: warm reps
  within 0.8% (0.32 ms), and the slope of r1 - r2 against the rep gap fell from
  0.13 to +0.011 ms per position (95% CI -0.036 to +0.058; a drift half
  EXP-0005's size is not excluded). The arm set and order changed too and there
  was no control, so neither the headroom nor the harness is shown to be the
  cause. Check every guard the plan names: EXP-0006's warm-up went below the 300
  MiB floor for about 5 s and the script checked only for a kill.
- **Pageable share does not convert to milliseconds at one rate across pinning
  orders** (EXP-0006, Finding 12). A profile's pageable remainder faults more
  per byte (2.0 to 12% of its pageable copies against 0.9% under address order;
  level 4: because it is the cold tail), so a slope taken from one session's
  address-against-profile gap (EXP-0005: 0.125 and 0.163 ms per point) does not
  transfer to other profiles or prompts; in EXP-0006 it overestimated every
  gain.
- **Align replay tokens to timeline tokens before scoring windows.** In a
  `LLAMA_MOE_CACHE_TRACE` file, replay token 0 is the prompt's last-layer call
  and replay token t is decode pass t, which is timeline token t. EXP-0006's
  first coverage script was one pass early.
- **ABBA cancels a linear drift, not a memory-state effect** that decays with
  token index. Report the per-round pair differences next to the means.
- **System-wide paging and disk counters are not llama-only** after an eviction:
  trimmed processes page back in during the run (0.10 to 0.65 GiB of reads in
  tokens 387-1022 of a run with no file-backed experts).
- **typeperf stalls for up to 5.5 s while a model loads.** Integrate each rate
  over its real interval; summing rows as 1 s each undercounted disk reads 1.8
  to 4.3x.
- **Bytes over token time is not a link rate.** It folds compute, faults, staging
  and syncs into one number; Finding 4 read it as a link rate for three
  iterations. Measure copies on a timeline instead.
- **Tracing with Nsight Systems costs +0.6 to +2.2% per token** in the 128-token
  runs (unmeasured in the warm regime), host-side, so it lands in GPU idle and
  leaves compute and DMA unaffected (EXP-0004).
- **The B2 reference and a traced CLI run disagree at 2048 MiB**: the traced CLI
  is 10 to 36% slower per token (they agree at 586). Unexplained; do not mix the
  two without reconciling them.
- **Compare policies paired per trace, and require the sign to agree on all four
  traces.** Level 1, from EXP-0001. The spread of one policy **between** traces
  is 4.2 to 7.3 points, which is larger than several of the differences being
  compared, so a positive mean is not evidence. Full LFU at 256 and 384 MiB
  would pass a mean test while three of four traces disagree with it.

## Standing practice: check the router after every rebase

**Level 1.** A base change can perturb the router's top-8, which changes the
generated text and therefore invalidates the traces - EXP-0001 spent a whole
iteration establishing that after the `78a8ae1` to `d854d98b` move.

There is a one-run check. **Run B1 at 2048 MiB and compare against
`61.38% / 58033.62 MiB`.** Nothing samples at that size, so the figure is fully
deterministic; if it reproduces, the traces are still valid.

Done after the move to `a4699400e` (33 upstream commits, EXP-0003): **exact at
both 1024 and 2048 MiB**, so that base did *not* move the router and no
re-collection was needed. Two bases have now been checked this way, with
opposite answers, which is why the check is worth its one run.

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

**Level 1. Rewritten by EXP-0005.** The cache only moves expert weights into
VRAM, so **with every expert computed on the same device**, generation must be
bit-identical. `moe-bench/check-correctness.sh` greedily decodes 256 tokens
under every configuration with `GGML_OP_OFFLOAD_MIN_BATCH=1`, so that every
expert is computed on the GPU, and diffs. On `5df437a57` plus EXP-0005, all ten
configurations agree byte for byte:
- the cache off;
- every eviction policy at 528 and 2048 MiB, and 256 MiB;
- whole-tensor pinning, and profile pinning at the default cap.

The reference also equals what the default 2048 MiB cache generates. Any
experiment that changes this output is a bug, not a trade-off.

**What the earlier statement got wrong.** It compared every configuration with
the cache off at 64 tokens. With the cache off (and in a cache group with no
slots) decode runs the experts on the CPU, whose rounding differs from CUDA's.
At 256 tokens the cache-off text diverges from every cache configuration a few
tokens after token 64 (byte 535, where 64 tokens end at byte 524), and 528 MiB
diverges from 2048 MiB at about token 110. So "byte-identical with the cache
off" held only for short outputs. Comparisons between policies at one cache size
are unaffected, because they compute the same layers on the same device. But
hardware rows at 256 to 586 MiB decode a different text than 2048 MiB after
about 100 tokens, so their hit rates are not over the same routing
(OPEN_QUESTIONS.md Q15).
