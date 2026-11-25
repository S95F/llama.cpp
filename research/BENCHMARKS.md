# BENCHMARKS

Standard benchmark definitions and the current reference numbers. Every claimed
improvement must name the benchmark ID it was measured with.

Hardware, toolchain and model: ENVIRONMENT.md. **Every shell that runs a binary
must prepend `C:\Users\steve\cuda126\bin` to `PATH`** or the binaries exit with
`0xC0000135`.

## Metric policy

Two metrics are **exact** for a fixed model, prompt and greedy decode, and they
carry every claim:

- **cache hit rate** (generation path, ubatch <= 8)
- **bytes uploaded over PCIe**

Throughput is **corroboration only**. It must be reported with its control rows
(configurations where the change is inert by construction) and with the free host
RAM at the time, because this machine's 31.11 GiB does not comfortably hold an
18.32 GiB expert working set. See KNOWLEDGE_BASE.md "Measurement discipline".

Derived quantities (uploaded bytes over token time) are effective rates, **not
link rates, and only loose lower bounds on the copy rate**: nothing overlaps, so
the time includes compute, page faults, staging and syncs (EXP-0004). Copy rates
come from B7.

## B1 - Deterministic cache telemetry

The primary benchmark. Exact, machine-load independent.

```sh
LLAMA_MOE_CACHE_SAMPLE=$sample ./build/bin/llama-completion \
    -m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib "$mib" \
    -c 2048 -n 256 -f moe-bench/prompt.txt --temp 0 --seed 1 \
    -t 8 --no-warmup -no-cnv -v < /dev/null 2>&1 | grep -E "ubatch <= 8" | tail -1
```

`LLAMA_MOE_CACHE_SAMPLE=0` is the plain LRU arm (the sampled victim off); unset
is the shipping branch arm (32 samples). Through base `a91eab67d` the LRU arm was
byte-for-byte upstream's policy, which is why the older tables call it
"upstream". Upstream has since moved to ClassLRU (#31233), which the branch does
not build (merge `10c477876` kept the branch's policy), so from `cddaa915a` on
the arm is labelled `lru`, not upstream. `16` forces 16 samples at every size,
so above the crossover, where the shipping policy does not sample, it differs
from both (EXP-0003's `s16` rows at 1024 and 2048 MiB). Sweep `mib` over
`256,384,528,586,645,1024,2048`. 255 tokens are generated after the first.

### Reference, base `cddaa915a` (merge `10c477876`, 2026-11-25)

Source: `research/results/baseline/rebase-10c477876/b1.csv`. **Use this table
from `10c477876` on; do not compare it with the older ones.** The router changed
at this merge: the same prompt now selects different experts (7530 of 10201
lookups differ as sets, the first at call 10, layer 8), so the generated text
and every count below differ from `a4699400e`'s.

| cache MiB | lru | shipping | shipping hits / misses |
|---:|---:|---:|---:|
| 256 | 0.00% / 145030.88 MiB | **16.92%** / 120607.58 MiB | 13461 / 66107 |
| 384 | 0.00% / 145030.88 MiB | **23.76%** / 110632.98 MiB | 18907 / 60661 |
| 528 | 1.10% / 143324.84 MiB | **29.67%** / 102034.90 MiB | 23609 / 55959 |
| 586 | 21.72% / 113537.43 MiB | **32.04%** / 98591.91 MiB | 25497 / 54071 |
| 645 | 31.23% / 99745.74 MiB | 31.23% / 99745.74 MiB | 24850 / 54718 |
| 1024 | 41.06% / 88328.32 MiB | 41.06% / 88328.32 MiB | 33511 / 48097 |
| 2048 | 60.25% / 59733.91 MiB | 60.25% / 59733.91 MiB | 49165 / 32443 |

645 to 2048 MiB are the control: the two arms are identical to the integer
counts. The shipping policy's gain below the crossover is the same size as on
the old base (+10.3 points at 586 MiB, +28.6 at 528).

**The router check is now `60.25% / 59733.91 MiB` (hits 49165, misses 32443)**
at 2048 MiB. The simulator reproduces it exactly from this base's trace
(`moe_sim.py slru-ramp`), and reproduces the old `61.38%` from EXP-0001's old
trace, so the port of the policy through the merge is faithful and the change
is in the routing.

### Reference, base `a4699400e` (exact on `2ca33f0d7` where re-run)

Source: EXP-0003, `research/results/experiments/EXP-0003/b1.csv`. The shipping
policy is the sampled distance-weighted victim with 32 samples.

| cache MiB | upstream | shipping | previous (16 samples) |
|---:|---:|---:|---:|
| 256 | 0.00% | **17.00%** / 120494.19 MiB | 16.36% |
| 384 | 0.00% | **24.03%** / 110250.28 MiB | 23.17% |
| 528 | 1.04% | **29.96%** / 101622.50 MiB | 29.30% |
| 586 | 20.93% | **31.91%** / 98790.82 MiB | 31.38% |
| 1024 | 41.36% | 41.36% / 87882.56 MiB | byte-identical |
| 2048 | 61.38% | 61.38% / 58033.62 MiB | byte-identical |

**The 2048 MiB row is the standard rebase check.** Nothing samples there, so it
is fully deterministic: if it reproduces the current reference (`60.25% /
59733.91 MiB` since `10c477876`; `61.38% / 58033.62 MiB` before) the traces are
still valid for the new base. It has reproduced exactly across `d854d98b`,
`a4699400e`, `5df437a57` (EXP-0004) and `2ca33f0d7`. On `2ca33f0d7` the whole
shipping column reproduced to the integer counts, and the upstream column at
528 and 2048 MiB, the two sizes run
(`research/results/baseline/rebase-2ca33f0d7/b1.csv`).

Caution on the A/B toggle: `LLAMA_MOE_CACHE_SAMPLE` clamps to `g.n_slots` while
the automatic path does not, so on a group with fewer slots than the requested
count the toggle is not exactly the shipping policy. EXP-0003 bounded the
difference at 0.02 points, and it is zero wherever the count is below the
group's slot count.

### Reference, base `d854d98b`

Source: `moe-bench/RESULTS.md` section 12.

| cache MiB | slots/token | upstream hit | upstream uploaded MiB | branch hit | branch uploaded MiB | traffic |
|---:|---:|---:|---:|---:|---:|---:|
| 384 | 0.66 | 0.00% | 145030.88 | 21.32% | 114174.28 | -21.3% |
| 528 | 0.90 | 1.04% | 143422.10 | 27.53% | 105147.00 | -26.7% |
| 586 | 1.00 | 20.93% | 114692.98 | 29.64% | 102097.78 | -11.0% |
| 645 | 1.10 | 29.96% | 101592.12 | 29.96% | 101592.12 | 0% (byte-identical) |
| 1024 | 1.75 | 41.36% | 87882.56 | 41.36% | 87882.56 | 0% (byte-identical) |
| 2048 | 3.49 | 61.38% | 58033.62 | 61.38% | 58033.62 | 0% (byte-identical) |

The last three rows are the **control**: the branch is inert above the crossover
by construction, and hardware confirms it to the hundredth of a MiB.

### Reference, base `78a8ae1` (historical, do not mix with the above)

Source: RESULTS.md section 5.

| cache MiB | upstream hit | branch hit |
|---:|---:|---:|
| 384 | 0.00% | 20.94% |
| 528 | 1.10% | 27.37% |
| 586 | 21.73% | 29.50% |

## B2 - Generation throughput sweep

```sh
LLAMA_MOE_CACHE_SAMPLE=$sample ./build/bin/llama-bench \
    -m "$MODEL" -ngl 99 -ncmoe 40 -mcmoe 384,528,586,645,1024,2048 \
    -n 64 -p 0 -r 4 -t 8 -o md
```

**Must be order-balanced**: run the LRU arm (`LLAMA_MOE_CACHE_SAMPLE=0`,
"upstream" in the tables below) first then the branch arm, then the reverse, and take the mean of the two passes. An unbalanced sweep showed the
branch gaining 4 to 12% at sizes where its hit rate is lower by construction.
Single process only. Record free RAM.

### Reference, base `d854d98b`, 16.1 GiB free

| cache MiB | upstream tok/s | branch tok/s | change |
|---:|---:|---:|---:|
| 384 | 5.71 | 7.54 | +32.2% |
| 528 | 7.24 | 10.65 | +47.1% |
| 586 | 10.07 | 12.67 | +25.8% |
| 645 | 12.15 | 12.24 | +0.7% (control) |
| 1024 | 15.02 | 15.57 | +3.7% (control) |
| 2048 | 22.10 | 21.24 | -3.9% (control) |

**Noise floor of that session: +3.7% to -3.9%**, read off the three control rows,
which B1 proves are byte-identical. The sub-crossover gains sit far outside it
but carry at least that uncertainty.

**Calibration caveat, from EXP-0004.** A warm 2048 MiB token in a traced
1024-token CLI run takes 52 to 56 ms, 10 to 36% more than the B2 rows at 2048 MiB
imply (41.2 to 47.1 ms); at 586 MiB the two agree (74.0 to 77.7 against 75 to
79). Not yet reconciled (OPEN_QUESTIONS.md Q12), so do not mix B2 and B7 timings.

### Reference, base `78a8ae1`, about 18 GiB free

| cache MiB | upstream tok/s | branch tok/s | change |
|---:|---:|---:|---:|
| 384 | 7.35 | 8.60 | +17.0% |
| 528 | 8.54 | 11.35 | +32.9% |
| 586 | 11.34 | 13.28 | +17.1% |
| 645 | 13.63 | 13.68 | +0.4% (control) |
| 1024 | 17.20 | 16.98 | -1.3% (control) |
| 2048 | 24.16 | 24.25 | +0.4% (control) |

Noise floor at 18 GiB free: **+-1.3%**. Absolute throughput is 10 to 15% higher
than the 16.1 GiB session at every size. **The larger ratios in the 16.1 GiB
table are not an improvement** - host memory pressure penalises whichever arm
reads more bytes, which is the upstream arm, inflating the ratio.

### Across the merge `10c477876`, 2048 MiB only, 2026-11-25

The merged build against the pre-merge `980791788` (built in a worktree at
`C:\nsys\premerge_980791788`), same protocol as `2ca33f0d7` below, ABBA then
BAAB, `GGML_CUDA_REGISTER_HOST=1` set explicitly in both arms, 20567 to 20857
MiB available, `VIractal.exe` not running and the GPU idle (0%, about 18 W) at
every launch (`baseline/rebase-10c477876/bench.md`; per-rep samples in
`bench.jsonl`, reduced by `summarize.py`). Both binaries report build commit
`980791788`: the merged one was compiled from the merge's working tree before
it was committed, so its build info names the parent; no source is newer than
it.

| | merged tok/s | pre-merge tok/s | change |
|---|---:|---:|---:|
| block 1 (ABBA) | 27.04 | 28.45 | -5.0% |
| block 2 (BAAB) | 28.34 | 28.40 | -0.2% |
| all eight runs | 27.69 | 28.42 | -2.6% |

**No decode change detected across the merge.** The -2.6% is run 1 alone
(25.59, the first process after the trace collection), whose reps 1 and 3
(20.01, 25.33) are the lowest of all eight runs. Runs 2 to 8 all lie between
28.06 and 28.64; without run 1 (a choice made after seeing the data, which
breaks the balance) merged is 28.39 against 28.42, -0.1%. The same first-run
effect appeared at `2ca33f0d7`, in the other direction. The rep pattern
(rep 1 slowest, rep 4 fastest) holds again in every run.

Absolute throughput is about twice the `2ca33f0d7` check's (28 against 14 to
16 tok/s per run) at 0.4 to 2.3 GiB more available RAM, with pinning set
explicitly here and inherited there. Not investigated; do not mix the two.

### Across the merge `2ca33f0d7`, 2048 MiB only, 2026-11-01

The merged build against the pre-merge `a88e64081`, one size, ABBA then BAAB,
18.5 to 19.7 GiB (18976 to 20127 MiB) available, `VIractal.exe` not running
(the only process the script checks) and the GPU idle (0 to 2%, about 25 W) at
every launch (`baseline/rebase-2ca33f0d7/bench_r3.md`; per-rep samples in
`bench_r3.jsonl`, reduced by `summarize_r3.py` into `bench_r3_summary.txt`):

| | merged tok/s | pre-merge tok/s | change |
|---|---:|---:|---:|
| block 1 (ABBA) | 14.99 | 14.19 | +5.6% |
| block 2 (BAAB) | 14.48 | 14.45 | +0.2% |
| all eight runs | 14.73 | 14.32 | +2.9% |

**No decode change detected across the merge.** The +2.9% is run 1 alone
(15.86, the first process after `bench_r2.sh`). All its reps are high, most of
all reps 2 and 3 (16.69 and 15.70 against at most 14.46 in runs 2 to 8), and
its rep 1 is 10.05 against 7.33 to 9.58; why is untested. Dropping run 1, a
choice made after seeing the data, leaves three merged runs against four
pre-merge ones, no longer order-balanced. Reps 2 to 4 then differ by -0.2,
+0.4 and +0.1%, inside their run-to-run ranges of 0.9 to 2.0%; rep 1 (7.33 to
9.58) is too noisy to compare. A change under about 1% is not excluded.

`bench_r2.sh` (pp512 then tg64 in each process, ABBA, `bench_abba_tg_r2.md`)
gave tg64 +5.7% and pp512 +6.2% (218.75 against 205.89 tok/s). Both are inside
the run-to-run spread of one build (pre-merge tg64 24.72 and 26.63, 7.7%
apart; merged pp512 206.44 and 231.05, 11.9%): no conclusion either way.

**The reps are not exchangeable.** In all eight runs, both builds, rep 1 is the
slowest and rep 4 the fastest: rep 1 7.33 to 10.05 tok/s, reps 2 and 3 14.17
to 16.69, rep 4 20.55 to 21.02. llama-bench never seeds `std::rand`, so rep k
decodes the same tokens in every process, and its generation warm-up is one
token. A `-r 4` mean therefore mixes reps up to 2.8x apart, for a reason not
yet known, and it depends on what ran earlier in the process: with pp512 first
(`bench_r2.sh`; this also moves the rand stream, so the tokens differ too)
tg64 measured 24.72 to 27.26 tok/s. The model stays loaded across `-mcmoe`
values (a context parameter), so a sweep run as the command above measures
2048 MiB last, after five sizes in the same process. If the references were run
that way, that may be why a fresh process now measures 14.1 to 15.9 tok/s per
run against their 21.24 to 24.25 (level 4, not tested). Cause:
OPEN_QUESTIONS.md Q20.

## B3 - Correctness

```sh
./moe-bench/check-correctness.sh "$MODEL" [baseline_build/bin]
```

Greedily decodes 256 tokens (`NGEN`) of the same prompt under each
configuration and diffs against `--moe-cache-mib 0`. Configurations:
`cache2048-lru`, `cache2048-auto`, `cache2048-s16`, `cache528-lru`,
`cache528-auto`, `cache528-dist0`, `cache256-auto`, `cache1024-pinned`;
`cache2048-pinprof` when `PIN_PROFILE=<trace>` is set (profile pinning at the
default cap); `cache2048-pinonline` (online pinning after 16 passes, default
cap) and `cache2048-pinonline14000` (after 64 passes, 14000 MiB), so the
registration lands inside the 256 tokens (15000 MiB until `10c477876`, see
below); `cache2048-prefetch` (pageable,
`LLAMA_MOE_PREFETCH=1`), and with `PIN_PROFILE` also
`cache2048-pinprof-prefetch` and `cache2048-pinprof-prefetch2` (modes 1 and 2
around pinned experts; EXP-0007); and `upstream-off` and `upstream-2048` when a
baseline build is given.

**The script exports `GGML_CUDA_REGISTER_HOST=0`** (since EXP-0007), so only the
rows that set it pin. `GGML_CUDA_REGISTER_HOST=1` is set in this host's user
environment (ENVIRONMENT.md), and before EXP-0007 every cache-on row that did not set it inherited it and
pinned the default cap in address order (`cache-off` inherited it too, but
builds no cache and so never pins).
The outputs were identical either way, so no earlier pass/fail changes, but
those rows were not the configurations their names say.

**Every configuration computes its experts on the GPU**
(`GGML_OP_OFFLOAD_MIN_BATCH=1`, exported by the script). Without it, the cache-off
reference and any cache group with no slots compute their experts on the CPU,
whose output diverges from the GPU's within about 65 to 110 tokens (EXP-0005).
The 64-token gate used before EXP-0005 passed only because it stopped just
short of that.

**Reference: all configurations byte-identical.** This is a pass/fail gate, not
a measurement. Any difference is a bug.

Last run on the merge `10c477876` (upstream `cddaa915a`): 14 of 14 OK,
`PIN_PROFILE` set, all 15 outputs byte-identical to cache-off
(`baseline/rebase-10c477876/b3_correctness_gpu256.txt`). They differ from
EXP-0007's pre-merge outputs from byte 535 on, all at the same place, because
the router changed (B1). The first attempt failed the online pinning row at
15000 MiB: not a different decode but a CUDA out of memory error after
registration, with about 15 GiB being the WDDM pinned budget. The pre-merge
build fails the same way at 15200 MiB, so the row was testing the budget
ceiling; it now pins 14000 MiB (`b3_note.txt`).

Before it, EXP-0007 (working tree on `b2586549c`, prefetch modes 1 and 2):
14 of 14 OK with the pinning variable controlled, `PIN_PROFILE` set, all 15
outputs byte-identical to each other and to the merge check's
(`EXP-0007/b3_correctness_gpu256.txt`; the first attempt, with the variable
inherited, also 14 of 14: `b3_correctness_gpu256_inherited.txt`). Before it, on
the merge `2ca33f0d7`: 11 of 11 OK, `PIN_PROFILE` set, and all 12
outputs byte-identical to EXP-0006's pre-merge ones
(`baseline/rebase-2ca33f0d7/b3_correctness_gpu256.txt`, `b3_across_merge.txt`).
Before it, EXP-0006 on `24c58282f` plus its changes: 11 of 11 OK against the
reference, `PIN_PROFILE` set (`EXP-0006/b3_correctness_gpu256.txt`). Before that,
EXP-0005 on `5df437a57`: 9 of 9; EXP-0002 on `d854d98b`: 8 of 8 at 64 tokens
against the CPU reference.

A baseline build of upstream master does **not** currently exist for `d854d98b`
or any later base, so the two `upstream-*` rows have not run on any of them,
the current `cddaa915a` included.

## B4 - Prompt processing

```sh
./build/bin/llama-bench -m "$MODEL" -ngl 99 -ncmoe 40 \
    -mcmoe 384,528,645,1024,2048,3072 -n 0 -p 512 -r 4 -t 8 -o md
```

Bypasses the cache above 32 tokens per ubatch, so expect no cache sensitivity.

Reference: a 54-token prompt pushed **5696.85 MiB** on `78a8ae1`, **5707.74 MiB**
on `d854d98b`, with zero cache reuse. The `-p 512` sweep itself has not been
tabulated in the research record.

## B5 - Host expert page pinning sweep

```sh
GGML_CUDA_REGISTER_HOST=1 LLAMA_MOE_PIN_MIB=$mib ./build/bin/llama-bench \
    -m "$MODEL" -ngl 99 -ncmoe 40 -mcmoe 1024 -n 64 -p 0 -r 4 -t 8 -o md
```

Run at a 1024 MiB cache, where the eviction change is inert. Reference table in
KNOWLEDGE_BASE.md Finding 4.

**This is the noisiest benchmark here (+-25 to 45%) and it is confounded with
page residency.** Do not re-run it for small effects; it was deliberately skipped
during the `d854d98b` rebase verification for that reason.

## B6 - Offline policy simulation, no GPU

```sh
LLAMA_MOE_CACHE_TRACE=trace.csv ./build/bin/llama-completion -m "$MODEL" \
    -ngl 99 -ncmoe 40 --moe-cache-mib 2048 -c 2048 -n 256 \
    -f moe-bench/prompt.txt --temp 0 --seed 1 -no-cnv -v

python moe-bench/moe_sim.py trace.csv \
    --layout moe-bench/layout-qwen35moe-35b-a3b-ud-q4kxl.json \
    --sizes 384,528,645,1024,2048 --policies slru-ramp,ramp-s16-age8,lfu,belady \
    --json research/results/experiments/<id>.json

python moe-bench/trace_stats.py trace.csv --layout moe-bench/layout-*.json
```

Policies implemented: `lru`, `slru50`, `slru-ramp`, `lfu`, `s3fifo`, `belady`,
`ramp-freq4/8/16`, `lru-freq8`, `ramp-rand`, `lru-rand`.

**Validation status: validated on three bases.**

- **`cddaa915a` (merge `10c477876`), all four traces, 2048 MiB: exact to the
  integer hit and fill counts** and to the hundredth of a MiB: prompt 49165 /
  32443, prompt_b 50126 / 31482, prompt_c 53260 / 28348, prompt_d 50141 /
  31467 (`research/results/baseline/rebase-10c477876/sim_validate.txt`). The
  router changed at this merge, so **these four traces replace EXP-0001's** for
  any simulation on this base (`trace_*.csv` there).

- `78a8ae1`, one trace, 2048 MiB: simulator 60.82% / 58867.3 MiB / 2044.1 MiB
  VRAM against llama.cpp 60.82% / 58867.26 / 2044.11.
- **`d854d98b`, all four traces, 2048 MiB (EXP-0001): exact to the integer hit
  and fill counts.** prompt 50092 / 31516, prompt_b 49846 / 31762, prompt_c
  53248 / 28360, prompt_d 48715 / 32893, identical in both implementations.
  Uploaded bytes and VRAM match to the printed precision.

2048 MiB is the size used for validation because the branch is inert there, so
the policy is fully deterministic and the agreement should be exact. **Below the
crossover exact agreement is not expected**: the C++ uses a small LCG where the
simulator uses Python's RNG, so the two draw different slot samples. EXP-0000
measured that residual on hardware at 0.7 to 0.8 points. At 256 to 586 MiB there
is a second difference: a cache group with no slots runs on the CPU, so hardware
decodes a different text than the 2048 MiB traces from about token 110
(EXP-0005, OPEN_QUESTIONS.md Q15).

**After every rebase, run the router check** (B1 at 2048 MiB, KNOWLEDGE_BASE.md
"Standing practice") **and re-validate if it fails.** The traces are
base-specific even though the simulator is not: upstream's own commits can
change the router's top-8 and therefore the generated text. The check decodes
the `prompt` trace only; the other three are assumed to move with it.

### Screening rule for policy comparisons

From EXP-0001. One policy's hit rate varies by **4.2 to 7.3 points between the
four traces**, which is larger than several of the differences worth testing.
Therefore:

- compare policies **paired per trace**, not by their four-trace means;
- require the **sign of the delta to agree on all four traces** before calling a
  difference real.

Full LFU at 256 and 384 MiB is the worked counter-example: its mean beats the
branch at 256 by 0.63 points while three of the four traces go the other way.

## B7 - Per-token CUDA timeline

From EXP-0004. Answers **where the time goes**, not how fast: compute, DMA,
read-back and GPU idle per generated token, with each idle interval attributed
to what the host thread was doing.

```sh
GGML_CUDA_REGISTER_HOST=$pin nsys profile --trace=cuda-sw,nvtx --sample=none \
    --cpuctxsw=none --cuda-memory-usage=false --cuda-graph-trace=graph:host-only \
    --gpu-metrics-devices=none --export=sqlite --output="$CAP/$tag" \
    ./build/bin/llama-completion -m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib "$mib" \
    -c 2048 -n 1024 -f moe-bench/prompt.txt --temp 0 --seed 1 -t 8 \
    --no-warmup -no-cnv -lv 4 < /dev/null
python research/results/experiments/EXP-0004/analyze_timeline.py "$CAP/$tag.sqlite" \
    --out timeline.json --log-out log.txt
```

`bench_timeline.sh` (128 tokens, six arms, ABA) and `long_runs.sh` (1024 tokens,
with Windows paging counters) in `research/results/experiments/EXP-0004/` are the
reference harnesses. nsys is at
`C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe`.

Rules learned in EXP-0004:

- **Quote a warm window** (tokens 387 to 1022 of a 1024-token run) for steady
  state. In EXP-0004's state (no eviction, 16.5 to 18 GiB available) the first
  128 tokens of a fresh process were dominated by page faults; after an eviction
  (EXP-0005) they are early, not cold.
- **Record the prefill time** as a coldness check. Cold, the 46-token prompt
  prefills in 13.4 to 13.8 s with pageable experts (about 6480 slow copies) and
  9.3 to 9.9 s with 42% pinned (about 3600), at 16.5 to 18 GiB available and no
  eviction; there, 3.7 to 7.3 s pageable meant an earlier run had left the model
  partly cached. After an eviction at 19.7 to 21.1 GiB available (EXP-0005),
  pageable prefill took 2.1 to 3.0 s and 42% pinned 2.5 to 3.0 s, so a fast
  prefill does not by itself mean an earlier run left the model partly cached.
  Record a cold reference for each pinned size.
- **Tracing costs +0.6 to +2.2% per token** in the 128-token runs (unmeasured
  in the warm regime), host-side; it lands in GPU idle.
- Tokens 1 and 2 are excluded: token 1 runs eagerly and token 2 captures the
  graphs, and both are mostly page-fault idle (320 and 605 of 449 and 693 ms at
  2048 MiB). Every 256th token re-captures and runs most graphs eagerly.
- `graph:host-only` gives graph spans, not the kernels inside them.

Added by EXP-0005 (`research/results/experiments/EXP-0005/runs.sh` is now the
reference harness):

- **Evict the model before every run** (`evict.py`, no admin rights) and record
  residency with `probe.py` (0 to 7 of 256 pages cached is cold). Record
  available RAM before each run, and hold it fixed if reps must agree within
  3% (ROADMAP R16); eviction alone did not.
- **Tokens 3-130 are early, not cold**, after an eviction: the load's prefetch
  re-reads most of the file. Their figures are indicative only.
- **Run arms in ABBA order and report the per-round pair differences**, not only
  the means; ABBA does not cancel a memory-state effect that decays with token
  index.
- **A watchdog** stops a run if available RAM stays below 300 MiB for 20 s. At
  about 15 GiB pinned it fired once.
- **Integrate typeperf rates over the real row intervals**; rows stall for up
  to 5.5 s during a load. The counters are system-wide, so other processes'
  page-ins after an eviction land in them.

Added by EXP-0006 (`research/results/experiments/EXP-0006/runs.sh` is now the
reference harness):

- **Hold the launch headroom fixed**: `balloon.py` locks memory until 19968 MiB
  is available and holds it to the end of the run, with up to three attempts
  (each after a new eviction); a run whose balloon missed the target by more
  than 192 MiB is flagged. With it and a discarded warm-up run, warm reps agreed
  within 0.8% and no run-position drift was detectable (a drift half EXP-0005's
  size is not excluded; KNOWLEDGE_BASE.md, Measurement discipline). The eviction
  stays.
- **One benchmark process at a time, checked**: the script stops if another
  `llama-completion.exe` is running, and the monitor follows the new PID only.
- **Record the stdout of every run** (`out_<tag>.txt`, its sha256 in
  `runs.json`) and check it is identical across arms of one prompt.
- **Check the warm-up against every guard**, the 300 MiB floor too: EXP-0006's
  warm-up went below it for about 5 s and the script checked only for a kill.
- **Report 3-1022 next to 387-1022** for anything that pins mid-run: the
  online profile's cost is in the early passes.
- In the Nsight sqlite export, `CUPTI_ACTIVITY_KIND_MEMCPY.srcKind` is 1 for
  pinned and 0 for pageable host memory (`ENUM_CUDA_MEM_KIND`).

### Reference, base `b2586549c` plus EXP-0007, warm (tokens 387 to 1022)

2048 MiB cache, profile pinning at the default cap unless marked, mean of two
reps (mirror order), model evicted and about 19968 MiB available at every launch (19880 to 20028,
balloon-held except `prof_r1`; `lo`: 18944). Every arm ran with the copy log and the selection trace on.

| arm | ms/token | compute | DMA | idle | slow-copy idle | no-API idle | pinned share |
|---|---:|---:|---:|---:|---:|---:|---:|
| profile, no prefetch | 41.77 | 12.92 | 20.07 | 8.65 | 3.51 | 0.72 | 77.7% |
| profile, `LLAMA_MOE_PREFETCH=1` | 40.81 | 12.95 | 20.12 | 7.62 | 0.00 | 2.96 | 77.7% |
| profile, `LLAMA_MOE_PREFETCH=2` | **39.18** | 12.88 | 20.10 | 6.07 | 0.02 | 1.59 | 77.7% |
| address order, no prefetch | 45.16 | 12.96 | 22.63 | 9.46 | 2.81 | 0.77 | 49.3% |
| address order, mode 1 | 47.19 | 12.91 | 22.71 | 11.46 | 0.02 | 5.46 | 49.3% |
| address order, mode 2 | 43.42 | 12.98 | 22.61 | 7.71 | 0.00 | 1.54 | 49.3% |
| profile, 1 GiB less headroom | 42.69 | 12.88 | 20.01 | 9.68 | 4.71 | 0.72 | 77.7% |

The no-prefetch arms reproduce EXP-0006's within 0.2 ms (41.73 and 45.32).
Unpinned cold, tokens 3-126 of 128: 78.43 ms per token, 65.52 with mode 1.
Tokens 3-1022 and 3-130 are in
`research/results/experiments/EXP-0007/analysis.md`.

### Reference, base `24c58282f` plus EXP-0006, warm (tokens 387 to 1022)

2048 MiB cache, mean of two reps (ABBA), model evicted and 19968 MiB available
held at every launch. Prompt a unless marked; the story is `prompt_ood.txt`.

| arm | ms/token | compute | DMA | idle | slow-copy idle | pinned share |
|---|---:|---:|---:|---:|---:|---:|
| address order, default cap | 45.32 | 12.99 | 22.19 | 10.02 | 2.59 | 49.3% |
| profile order, default cap | 41.73 | 12.95 | 20.04 | 8.62 | 3.48 | 77.7% |
| online, N = 64, default cap | 42.38 | 12.94 | 20.24 | 9.08 | 3.78 | 75.2% |
| online, N = 128, default cap | 41.57 | 12.93 | 19.83 | 8.70 | 3.61 | 79.8% |
| profile order, 15000 MiB | 37.11 | 12.89 | 18.20 | 5.89 | 1.57 | 97.7% |
| online, N = 64, 15000 MiB | 37.94 | 12.90 | 18.50 | 6.42 | 1.83 | 93.3% |
| online, N = 128, 15000 MiB | 38.01 | 12.91 | 18.26 | 6.71 | 2.17 | 95.8% |
| story, address order | 36.71 | 12.94 | 15.88 | 7.77 | 1.69 | 50.9% |
| story, profile order | 36.17 | 12.93 | 14.93 | 8.19 | 2.79 | 65.5% |
| story, online, N = 128 | 33.92 | 12.92 | 13.44 | 7.45 | 2.86 | 88.4% |

The three arms repeated from EXP-0005 ran 0.7 to 1.4 ms slower here, mostly in
slow-copy idle, at 0.2 to 1.7 GiB less launch headroom; do not mix the two
sessions' tables. Tokens 3-1022 and 3-130 are in
`research/results/experiments/EXP-0006/analysis.md`.

### Reference, base `5df437a57`, warm (tokens 387 to 1022), EXP-0005

2048 MiB cache, mean of two reps (ABBA), model evicted before every run,
19.7 to 21.1 GiB available before eviction:

| arm | ms/token | compute | DMA | idle | pinned share |
|---|---:|---:|---:|---:|---:|
| pageable | 52.95 | 13.02 | 26.86 | 12.95 | 0% |
| address order, default cap (7842 MiB) | 43.97 | 12.96 | 22.69 | 8.20 | 49.3% |
| profile order, default cap (7943 MiB) | 40.42 | 12.96 | 20.08 | 7.26 | 77.7% |
| address order, 15000 MiB (387-841) | 39.35 | 12.92 | 20.12 | 6.18 | 79.8% |
| profile order, 15000 MiB | 36.43 | 12.89 | 18.23 | 5.19 | 97.7% |
| `--load-mode none` | 37.93 | 12.89 | 19.71 | 5.21 | 81.5% |

Read-back is 0.1 ms in every arm. Full tables in
`research/results/experiments/EXP-0005/analysis.md`.

### Reference, base `5df437a57`, warm (tokens 387 to 1022), EXP-0004

| arm | ms/token | compute | DMA | read-back | idle |
|---|---:|---:|---:|---:|---:|
| 2048 MiB pageable | 56.2 | 12.8 | 27.1 | 0.1 | 16.1 |
| 2048 MiB pinned 42% | 46.6 | 12.8 | 21.8 | 0.1 | 11.9 |
| 586 MiB pageable | 77.7 | 12.8 | 43.5 | 0.1 | 21.3 |

One traced rep each, 17.5 to 18.2 GiB free. Cold 128-token figures are in
EXP-0004.md.

## Raw results

`research/results/baseline/` and `research/results/experiments/`. JSON or CSV,
one directory per experiment ID, never overwritten.

Committed so far:

- `research/results/experiments/EXP-0001/` - the four `d854d98b` traces
  (`trace_*.csv`, 10202 lines each), the cache counters (`telemetry_2048.txt`),
  per-trace simulator output (`sim_validate_*.json`, `sim_sweep_*.json`), the
  routing statistics (`trace_stats.txt`), and the two scripts that produced and
  reduced them (`collect.sh`, `aggregate.py`).
- `research/results/experiments/EXP-0002/` and `EXP-0003/` - simulator screens
  and B1 hardware counters.
- `research/results/experiments/EXP-0004/` - per-token timelines
  (`timeline_*.json`), the app logs, the decomposition, paging counters, the
  analysis scripts and the review record. The nsys captures themselves (30 files, 482 MB)
  stay outside git; `provenance.txt` has their sha256.
- `research/results/experiments/EXP-0005/` - per-run eviction and probe records,
  paging counters, timelines and logs for the 12 pinning runs, `runs.json` and
  `analysis.json`, the offline coverage (`coverage_*.json`), the budget and
  copy-edge tests, three B3 records and the review record. Captures in
  `C:\nsys\exp0005` stay outside git; `provenance.txt` has their sha256.
- `research/results/experiments/EXP-0006/` - per-run balloon, paging, process
  and low-memory records, timelines, logs and summaries for the 21 runs
  (warm-up included), `runs.json`, `analysis.json`, `captures.json` (copy kinds,
  registration costs), the two 1024-token traces and their coverage replays
  (`coverage_1024.json` as written before the runs, `coverage_1024_aligned.json`
  after the one-pass fix), the B3 record and the review records. Captures in
  `C:\nsys\exp0006` stay outside git; `provenance.txt` has their sha256.
- `research/results/experiments/EXP-0007/` - gate G (`prefetch_bench.py`,
  `gate.csv`, `gate.txt`, `gate_pre.txt`), per-run balloon, paging, process
  and low-memory records, timelines, logs and summaries for the 19 runs
  (warm-up included), `runs.json`, `analysis.json` and `analysis.md`, the fault
  attribution (`attrib.json`, from `attrib.py`), the harness (`runs.sh`) and the
  two B3 records with a note on the first. Captures, selection traces and copy
  logs (1.67 GB: Nsight captures 1.44 GB, copy logs 195 MB, selection traces
  28 MB) in `C:\nsys\exp0007` stay outside git; `provenance.txt` has
  their sha256. The pre- and post-run reviews are summarized in `EXP-0007.md`;
  their records stayed in the session's local workflow log.
- `research/results/baseline/rebase-2ca33f0d7/` - the check of the upstream
  merge `2ca33f0d7`: B1 (`b1.csv`), B3 and its diff against the pre-merge
  outputs, the buffer sizes across the merge, test-backend-ops MUL_MAT_ID and
  MUL_MAT_VEC_FUSION (`test_backend_ops_mmvf.txt`, run after the rest), and
  the throughput checks (`check.sh`, `bench_r2.sh`, `bench_r3.sh`;
  `bench_abba_tg.md` came from an unscripted tg64-only loop of the same form).
  `bench_abba.md` and `bench_abba_tg.md` ran while a game used the GPU and are
  kept but invalid (`NOTE_game.txt`). `bench_abba_tg_r2.md` (pp512 and tg64) and
  `bench_r3.md` (B2 protocol, per-rep samples in `bench_r3.jsonl`, reduced by
  `summarize_r3.py` into `bench_r3_summary.txt`) are the re-runs. Logs in
  `C:\nsys\rebase_2ca33f0d7` stay outside git. The record of the nine-agent
  review before the merge stayed in the session's local workflow log and is
  not committed.

- `research/results/baseline/rebase-10c477876/` - the check of the upstream
  merge `10c477876`: B1 at seven sizes in both arms (`b1.csv`), B3 (first
  attempt, note, rerun) and its diff against EXP-0007's outputs,
  test-backend-ops MUL_MAT_ID, the four re-collected selection traces with
  their outputs, counters and simulator validation (`traces.sh`), and the
  buffer sizes and decode timing across the merge (`bench.sh`, `bench.md`,
  `bench.jsonl`, `summarize.py`). Logs in `C:\nsys\rebase_10c477876` stay
  outside git.

**Gap: nothing from the pre-`research/` work was preserved.**
`moe-bench/.gitignore` excludes `out/` and `*.csv`, so the `78a8ae1` reference
tables above are traceable only to the prose of `moe-bench/RESULTS.md`, not to a
raw file. Fixed from EXP-0001 onward; the historical raw data is gone.
