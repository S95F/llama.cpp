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

**Guard:** hold RAM headroom fixed, record available RAM before every run, and
report per-round pair differences. EXP-0006 held the launch headroom (a balloon
to 19968 MiB) with a warm-up run: warm reps within 0.8%, no detectable drift (a
drift half EXP-0005's size is not excluded).

### Summing typeperf rows as one second each

Found by EXP-0005. During a model load typeperf rows stall for up to 5.5 s, and
each row's cooked rate is the average over its real interval. Summing the rates
as if every row were 1 s undercounted the mmap runs' disk reads 1.8 to 4.3x (the
`--load-mode none` runs, whose rows never stalled, 1.0x).

**Guard:** integrate each rate over the timestamp difference to its previous
row (`analyze.py`), and calibrate the counter against a known read
(`calib_disk_counter.txt`).

### Converting the pageable share to milliseconds at one rate across orders

Found by EXP-0006. The offline replay predicted every arm's pageable share to
0.2 points, and EXP-0005's slopes turned share into time. Every gain was
overestimated, by 0.5 to 1.5 ms per token in four cases: the online arms were
predicted -2 and -1 ms against address order over the whole run and measured
-0.47 and -0.54; on128x15k was predicted +0.3 ms against the file profile and
measured +0.90; the file profile on the story was predicted about -1.3 ms
against address order and measured -0.54. Under a profile 2.0 to 12% of the
pageable copies are hard faults against 0.9% under address order, so a pageable
byte costs more there (EXP-0007: because more of the profile's pageable
uploads are first reads since the load, which fault; re-reads do not). For the online arms the passes before the registration were also slower
than the rebuild from EXP-0005's timelines assumed.

**How it presents:** an exact share prediction that lends false confidence to
the time prediction built on it.

**Guard:** predict share and time separately, and fit the share-to-time rate
per pinning order (or model the fault rate of what stays pageable).

### Scoring a replay one pass early

Found by EXP-0006's review. In a `LLAMA_MOE_CACHE_TRACE` file replay token 0 is
the prompt's last-layer call (output pruning leaves one row) and replay token t
is decode pass t. The first `coverage.py` took replay token t for timeline token
t + 1, so its windows, its online pin sets and its registration point were one
pass early. Here it moved the shares by at most 0.2 points, but a coverage
figure for a short window, or a pin set from a few passes, would move more.

**Guard:** align trace and timeline tokens by a known event (the registration
pass, the prompt's single-row call) before scoring windows.

### Inheriting a variable from the user's environment

Found by EXP-0007's pre-run review. `GGML_CUDA_REGISTER_HOST=1` is set in this
host's Windows user environment, and harnesses that only add variables passed
it to every process that did not set it. B3's "unpinned" cache-on rows pinned the default cap in address order in
every B3 run from the script's creation (2026-10-25; EXP-0002, EXP-0003,
EXP-0005, EXP-0006 and the merge check) to the first B3 of EXP-0007, and
the B2 llama-bench scripts (EXP-0002, EXP-0003, the merge check) most likely
did too; EXP-0007's planned unpinned cold arms would have been pinned. B3's
outputs were identical either way, so no pass/fail changed, but those rows were
not what their names said.

**How it presents:** nothing in the output; a configuration label that the
process never ran.

**Guard:** clear or set every variable the code reads in each harness, and
write the environment of every run to its record (`runs.sh` and `info.txt`).

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

### Prefetching every missed expert range

EXP-0007 (partial negative; mode 2 was adopted). `LLAMA_MOE_PREFETCH=1` issues
one `PrefetchVirtualMemory` call per layer over every missed expert range that
is not fully pinned. The call blocks until its reads finish and costs 0.12 to 0.26 us per 4 KiB page already in RAM, and warm most candidate pages are: under
address order it prefetched 112 MiB per pass for 4.6 to 4.7 ms, and the token
got 2.03 ms slower (47.19 against 45.16 ms), in both rounds. Under the profile
it gained 0.96 ms, 1.63 ms less than the probed mode 2. Early and cold it wins
(-8.5 to -12.9 ms per token over tokens 3-130).

**Revisit when:** no working-set probe exists (mode 2 falls back to it), or
off Windows, where `llama_prefetch` is `madvise(MADV_WILLNEED)` and does not
block (not measured).

### An online profile where a matching file profile exists

EXP-0006 (partial negative; the mode itself was adopted). Against a file profile
from the same kind of text, `LLAMA_MOE_PIN_ONLINE` gains nothing warm (within 1
ms, slower in both rounds in three of four arms) and costs about 3 s per
1024-token run up front at the default cap (6-7 s at 15000 MiB): the passes
before the registration are pageable, the register calls take 0.54-0.59 s at the
default cap, and one token stalls for 1.4-1.8 s at 15000 MiB. Out of domain it
was 6.2% faster than the file profile warm.

**Revisit when:** no matching profile exists, or as part of a hybrid (ROADMAP
R18) that starts from the file profile and re-ranks online.
