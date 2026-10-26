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

Both produced confident nonsense before being caught. They are the reason
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

---

## Not a failure, but recorded to prevent a wasted iteration

### Populating the cache from the prefill

A 54-token prompt pushes 5.7 GiB over PCIe with zero cache reuse, which looks
like obvious low-hanging fruit. It is **deliberate upstream**: ubatches above
`max_batch = 32` bypass the slot map because a large batch touches most experts
of a layer and would evict the generation hot set.

Upstream PR #30594 is already attacking this properly (profile prefilling and
zero-upload inference). ROADMAP R6 says to **prefer the upstream solution** and
revisit only if it stalls. Duplicating it would be both wasted work and a
guaranteed upstream conflict.
