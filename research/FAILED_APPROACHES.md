# FAILED APPROACHES

Negative results, preserved deliberately. For each: what was tried, what it
measured, **why** it failed, and under what conditions revisiting it would be
justified. The directive forbids repeating a rejected approach without new
evidence or materially different conditions - this file is the record that makes
that checkable.

All figures from EXP-0000 unless stated; primary record
`moe-bench/RESULTS.md` section 8. Policy comparisons were simulated on `78a8ae1`
traces; **EXP-0001 re-ran them on `d854d98b`** and the entries below carry its
numbers where they differ. No rejection was overturned, but the S3-FIFO and full
LFU characterisations were imprecise and are corrected.

---

## Policy failures

### Previous-token expert prefetch

**Dropped before implementation**, on trace evidence.

Token-to-token expert overlap within a layer is only **31 to 37%**, and 19 to 23%
at a lag of 8 tokens. The resident hot set already delivers more than that
(hottest 8 of 256 experts take 31 to 33% of selections, hottest 64 take 83 to
85%), so a previous-token prefetch would spend PCIe bandwidth to do **worse than
the cache it feeds**.

EXP-0001 re-measured both on `d854d98b`: overlap **31.4 to 38.7%** at a lag of 1
token and **19.6 to 25.3%** at a lag of 8, against **79.1 to 85.1%** for the
hottest 64. Both ends shifted up slightly; the gap that decides the question is
unchanged.

**Failure cause: the workload.** Routing is skewed but not temporally sticky. This
is a property of the model's router, not of the implementation.

**Revisit if:** a model shows high token-to-token stickiness (measure it with
`trace_stats.py` before writing any code), or a *different* predictor is used -
the previous layer's selection in the same token, or the hidden state before the
router. Those are untried and are OPEN_QUESTIONS.md Q7. Upstream #21067 (tensor
prefetch) operates at a different level and is not covered by this rejection.

### Frequency sampling from the LRU end

Sampling candidate victims from the LRU end instead of uniformly over the cache:
**1.1% / 5.4%** hit rate at 256 / 512 MiB, against **14.2% / 25.6%** for uniform
sampling.

**Failure cause: wrong place to look.** Under a cyclic sweep the LRU end is
exactly where the entries that are about to be reused sit. Biasing the sample
towards them evicts precisely what is needed next.

**Revisit if:** never, for a cyclic access pattern. The result is structural, not
tuning.

### Pure random eviction

The cheapest possible fix for thrash. It does help - **3.2% / 14.0%** at
256 / 512 MiB - but it is far behind frequency sampling (14.2% / 25.6%) and it
costs **8 to 10 points at large caches**.

**Failure cause: it discards usable information.** Random avoids LRU's
pathological case without exploiting the routing skew that Finding 2 measures.

**Revisit if:** the use counters ever become a measurable cost. They are 40 KiB
for this model, so this is unlikely.

### S3-FIFO

Marginally the best at large caches (**86.94% vs 86.22% at 6144 MiB**) but
**erratic at the small sizes that matter here**, and it needs a new data structure.

**EXP-0001 sharpened this on `d854d98b`.** It is worse than "erratic" at the
smallest size and weaker at the largest than EXP-0000 implied:

| MiB | s3fifo | branch | slru-ramp | paired sign-consistent vs branch |
|---:|---:|---:|---:|---|
| 256 | **0.00** | 15.94 | 0.00 | yes, branch wins by 15.94 |
| 384 | 7.43 | 24.17 | 0.00 | yes, branch wins by 16.74 |
| 528 | 29.96 | 30.42 | 1.18 | no - a wash |
| 586 | 32.31 | 32.42 | 24.54 | no - a wash |
| 645 | 34.38 | 35.89 | 35.89 | no - a wash |
| 1024 | 42.64 | 44.67 | 44.67 | no - a wash |
| 2048 | 60.97 | 58.77 | 61.85 | yes, s3fifo beats the forced-on policy by 2.20 |

**0.00% at 256 MiB** is a complete failure, identical to plain LRU. And its win at
2048 MiB is against `ramp-s16-age8`, which is *not* what the branch runs there -
the branch is inert at 2048 and `slru-ramp` beats S3-FIFO 61.85 to 60.97. The
6144 MiB figure above was not re-tested.

**Failure cause: wrong operating point, bad complexity trade.** A wash across the
middle of the range, a failure at the bottom, and no win against the shipping
policy at the top, for a new data structure, in a codebase whose contribution
guide explicitly prefers "a simpler change that does 90% of the job".

**Revisit if:** working on a machine where the cache sits well above the
crossover. **Its admission-control idea is still not rejected** and survives as
ROADMAP R2 candidate 3, though EXP-0001 lowers the expected gain: if admission
were doing the heavy lifting, S3-FIFO should have beaten the branch somewhere in
528 to 1024 MiB, and it does not.

### Full LFU everywhere

Recorded by EXP-0000 as "wins below the crossover but loses up to 5 points above
it". **EXP-0001 found that too generous to LFU.** Paired per-trace against the
branch's sampled victim:

| MiB | prompt | prompt_b | prompt_c | prompt_d | mean | sign-consistent |
|---:|---:|---:|---:|---:|---:|---|
| 256 | -2.25 | -1.03 | -1.51 | +2.29 | -0.63 | **NO** |
| 384 | -0.74 | +1.45 | +1.01 | +4.83 | +1.64 | **NO** |
| 528 | +0.09 | +2.69 | +2.31 | +5.81 | +2.73 | yes |
| 586 | +0.34 | +2.68 | +2.79 | +6.08 | +2.97 | yes |
| 645 | +0.06 | +2.75 | +2.63 | +6.02 | +2.87 | yes |
| 1024 | +0.60 | +3.18 | +3.03 | +5.78 | +3.15 | yes |
| 2048 | +1.20 | +2.87 | +2.98 | +4.02 | +2.77 | yes |

So LFU wins only at **256 MiB**, by 0.63 points on the mean, and **not
sign-consistently** - three of the four traces have the sampled victim ahead. At
384 MiB it is a wash. From 528 MiB upward it loses, including below the
crossover. Against the policy that actually ships above the crossover it loses
5.84 points at 2048 MiB (56.01 vs 61.85).

**Failure cause: two of them.** Recency genuinely is better above the crossover
(the same result as forcing frequency on everywhere: 34.14% vs 35.89% at 1.09
slots per token, 58.77% vs 61.85% at 3.49). And below the crossover, exact
min-frequency is not better than sampling 16 slots - it is slightly worse from
528 MiB up - while being more machinery.

**Revisit if:** never as a global policy. The switch is the finding, and sampling
is not a compromise against exact LFU but an improvement on it over most of the
range.

### No counter aging

Omitting the halving of use counters every `8 * n_slots` fills: **loses 6 to 9
points** over a 1024-token four-topic session, and at 0.99 slots per token it is
**worse than upstream**.

**Failure cause: stale popularity.** Without aging, experts that were hot early
in a session keep their counters and resist eviction after the topic changes, so
the cache fills with a hot set that no longer exists.

**Revisit if:** never. Aging is load-bearing, and a single-topic benchmark would
hide that - which is why the four-prompt trace set exists.

---

## Measurement failures

Each produced confident nonsense before being caught. They are the reason
KNOWLEDGE_BASE.md's "Measurement discipline" section exists.

### Two benchmark processes at once

Each instance maps the whole 20.81 GiB model, so two exhaust 31 GiB of RAM, the
run becomes disk bound, and **every cache size returns the same ~6 tok/s** - the
cache appears to do nothing at all.

**How it presents:** a suspiciously flat sweep.

**Guard:** check for stray processes and free RAM before believing a flat sweep.

### Arms run in sequence rather than interleaved

The second arm inherits a warmer page cache. Run in sequence, the frequency
policy appeared to gain **4 to 12% even at cache sizes where its hit rate is
lower by construction** - which is how the bias was spotted.

**How it presents:** a gain in the control rows, where the change is provably
inert.

**Guard:** order-balance every sweep (arm A then B, then B then A, mean of the
two passes), and **always report the control rows**. A sweep without inert
control configurations cannot detect this.

### Timing the policy with `llama-completion -v`

Per-token debug logging is large enough to hide the effect: it showed 6.61 vs
6.53 tok/s where `llama-bench` on the same configurations shows a clear
difference.

**Guard:** `llama-completion -v` is for **telemetry and traces** (exact counters);
`llama-bench` is for **timing**. Never the other way round.

### Reading uploaded bytes over token time as a link rate

Found by EXP-0004, after three iterations. Finding 4 divided the uploaded bytes
per token by the time per token, got 3.4 to 5.5 GB/s, and read that as the rate
of the copy path - "the pageable path wastes most of the link" - on the premise
that "compute overlaps the copies". Nothing overlaps (0 ns in every token of 15
traced runs), and the time per token includes compute, page faults, staging and
syncs. The figure is a lower bound on the copy rate, but so loose that it said
nothing about the link: measured on a timeline, the copies run at 8.2 to 13 GB/s
pageable and 12.8 GB/s pinned.

**How it presents:** a plausible rate that is near-constant across cache sizes,
which looks like a bandwidth limit and is really a fixed ratio of time to bytes.

**Guard:** never derive a link rate from token time. Take copy rates from the
copies themselves (BENCHMARKS.md B7), and check an overlap assumption before
building on it.

### Reading a short fresh-process run as steady state

Found by EXP-0004. A pageable 128-token CLI generation (586 to 2048 MiB) spends
55 to 66% of its mean token waiting on hard page faults of the mmap'd expert
pages (49 to 57% with 42% pinned). They die away over about 400 tokens: in a
1024-token run, 2048 MiB pageable goes from 118.8 to 52.1 to 59.7 ms per token.
The short run is 2.1 to 2.5x slower than tokens 387 to 1022, and it also inherits whatever the
previous process left in the page cache: pageable runs after a pinned run started
partly warm (prefill 3.7 to 7.3 s instead of 13.4 to 13.8 s, up to 60% fewer
faults).

**How it presents:** two reps that agree well with each other (faults are
deterministic) and disagree with anything run in a different order or for longer.

**Guard:** quote a stated warm window for steady state, record the prefill time
as a coldness check, and order runs by page-cache state as well as by arm.

### A short correctness gate against the cache-off output

Found by EXP-0005. B3 decoded 64 tokens and required every cache configuration to
match `--moe-cache-mib 0` byte for byte, and it passed in every iteration. At 256
tokens every cache configuration differs from cache-off. The cache is not wrong:
with the cache off, decode computes the experts on the CPU (host weights, batch
under the offload threshold of 32), and CPU and CUDA round differently. The
cache-off text first differs at byte 535; 64 tokens end at byte 524. A cache
group with no slots (256 to 586 MiB) does the same for its layer, from about
token 110. With `GGML_OP_OFFLOAD_MIN_BATCH=1`, all ten configurations agree.

**How it presents:** a gate that passes and seems to prove a strong invariant
("bit-identical with the cache off"), but compares two compute devices and
passes only because it stops early.

**Guard:** compute every expert on the same device in a correctness gate (the
script now exports `GGML_OP_OFFLOAD_MIN_BATCH=1`), and decode long enough for
rounding to show (256 tokens).

### Fixing only the launch page-cache state (R10 as first framed)

Found by EXP-0005. R10 proposed that reps would agree within 3% if every run
started from the same page-cache state. Eviction by memory pressure did make the
launch state reproducible (0 to 7 of 256 probed pages cached before all 12
runs), but reps still differed by up to 3.9% warm and 28% in tokens 3-130, r1
was slower than r2 in all six arms, and the prefill coldness check did not
reproduce (604 against 44 slow copies). The load's prefetch refills the page
cache as far as available RAM allows (a hypothesis, confounded with run order),
and available RAM rose 1.5 GiB through the session, plausibly (not measured)
because each eviction trimmed other processes.

**How it presents:** a verified-cold start, then a drift that looks like an arm
effect when arms are not interleaved, and that ABBA averages away only if it is
linear.

**Guard:** hold RAM headroom fixed through each run (ROADMAP R16), record
available RAM before every run, and report per-round pair differences.

### Summing typeperf rows as one second each

Found by EXP-0005. During a model load typeperf rows stall for up to 5.5 s, and
each row's cooked rate is the average over its real interval. Summing the rates
as if every row were 1 s undercounted the mmap runs' disk reads 1.8 to 4.3x (the
`--load-mode none` runs, whose rows never stalled, 1.0x).

**Guard:** integrate each rate over the timestamp difference to its previous
row (`analyze.py`), and calibrate the counter against a known read
(`calib_disk_counter.txt`).

---

## Code failures

### Pinning single experts without splitting the copies

Found by EXP-0005. The first profile-pinned smoke run died in prefill (only
EXP-0005.md records it; the rerun overwrote its log) with `CUDA error: invalid
argument` in `ggml_backend_cuda_set_tensor_async`. Every expert tensor starts
off a page boundary, so a pinned expert's page-rounded range covers the first
partial page of its neighbour. A copy of that neighbour, or of a run of
consecutive experts, then starts inside a registered range and runs past it, and
the driver rejects such copies (`cross_copy.txt`). Address order never hit this,
because it pins whole tensors and no copy spans two tensors.

**Guard:** route every host upload of expert data through
`llama_moe_cache_set_async`, which splits at the edges of every range pinned in
the process (a second context on the same model included), and keep the
scheduler's whole-tensor fallback on it while anything is pinned.

---

## Not a failure, but recorded to prevent a wasted iteration

### Populating the cache from the prefill

A 54-token prompt pushes 5.7 GiB over PCIe with zero cache reuse, which looks
like obvious low-hanging fruit. It is **deliberate upstream**: ubatches above
`max_batch = 32` bypass the slot map because a large batch touches most experts
of a layer and would evict the generation hot set.

Upstream PR #30594 is already attacking this properly (profile prefilling and
zero-upload inference). ROADMAP R9 says to **prefer the upstream solution** and
revisit only if it stalls. Duplicating it would be both wasted work and a
guaranteed upstream conflict.
