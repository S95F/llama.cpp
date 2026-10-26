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
layers (RESULTS.md section 4). Traces were collected on `78a8ae1`.

| hottest k experts of a layer (of 256) | share of that layer's selections |
|---:|---:|
| 8 | 31 - 33% |
| 16 | 46 - 47% |
| 32 | 64 - 66% |
| 64 | 83 - 85% |

The hot set exists; recency cannot find it under thrash, frequency can. This is
the mechanism behind Finding 3.

## Finding 3: a frequency-aware victim recovers 8.7 to 26.5 points below the crossover

**Level 1.** Reproduced on hardware on both bases. It must be switched off above
the crossover.

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

## Finding 6: a large gap to offline optimal remains everywhere

**Level 1** (simulated, validated bit-exact against llama.cpp on `78a8ae1`).

Belady at the same slot count (RESULTS.md section 8):

| cache MiB | upstream | branch | Belady |
|---:|---:|---:|---:|
| 256 | 0.00% | 15.77% | 31.86% |
| 384 | 0.00% | 23.90% | 43.35% |
| 528 | 1.17% | 29.98% | 51.53% |
| 586 | 24.27% | 31.97% | 53.93% |
| 1024 | 44.34% | 44.34% | 64.74% |
| 2048 | 61.65% | 61.65% | 77.30% |

The branch closes roughly half the gap below the crossover and none of it above.
This is the headroom a profile-driven resident hot set (#30594) aims at.

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
- **The simulator is trustworthy but base-specific.** `moe-bench/moe_sim.py`
  matched llama.cpp to the digit on `78a8ae1` (60.82% hit, 58867.3 vs
  58867.26 MiB uploaded, 2044.1 vs 2044.11 MiB VRAM). Its traces have **not**
  been re-collected on `d854d98b`.

## Open caveat on the current base

**Level 1.** Upstream's own commits between `78a8ae1` and `d854d98b` changed the
router's top-8 at some positions: with `--moe-cache-mib 0`, where the cache is
never constructed, the two bases generate different text from the same greedy
decode. Consistent with `c49edf096` (fused FFN gate/up matmuls and GLU in MMQ)
and `a55c16e4e` (fused MRoPE and set-rows) changing reduction order and therefore
rounding, but intermediate commits were not built, so the attribution is **level
4**. Consequence: every simulated table above still rests on `78a8ae1` traces.
Re-collecting them is ROADMAP item R1.

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
