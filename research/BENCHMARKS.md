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

`LLAMA_MOE_CACHE_SAMPLE=0` is the upstream LRU arm; unset or `16` is the branch
arm. Sweep `mib` over `384,528,586,645,1024,2048`. 255 tokens are generated after
the first.

### Reference, base `a4699400e` (current)

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
is fully deterministic: if it reproduces `61.38% / 58033.62 MiB` the traces are
still valid for the new base. It has reproduced exactly across `d854d98b` and
`a4699400e`.

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

**Must be order-balanced**: run the upstream arm first then the branch arm, then
the reverse, and take the mean of the two passes. An unbalanced sweep showed the
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

## B3 - Correctness

```sh
./moe-bench/check-correctness.sh "$MODEL" [baseline_build/bin]
```

Greedily decodes 256 tokens (`NGEN`) of the same prompt under each
configuration and diffs against `--moe-cache-mib 0`. Configurations:
`cache2048-lru`, `cache2048-auto`, `cache2048-s16`, `cache528-lru`,
`cache528-auto`, `cache528-dist0`, `cache256-auto`, `cache1024-pinned`;
`cache2048-pinprof` when `PIN_PROFILE=<trace>` is set (profile pinning at the
default cap); and `upstream-off` and `upstream-2048` when a baseline build is
given.

**Every configuration computes its experts on the GPU**
(`GGML_OP_OFFLOAD_MIN_BATCH=1`, exported by the script). Without it, the cache-off
reference and any cache group with no slots compute their experts on the CPU,
whose output diverges from the GPU's within about 65 to 110 tokens (EXP-0005).
The 64-token gate used before EXP-0005 passed only because it stopped just
short of that.

**Reference: all configurations byte-identical.** This is a pass/fail gate, not
a measurement. Any difference is a bug.

Last run EXP-0005 on `5df437a57` plus its changes: 9 of 9 OK against the
reference (10 identical outputs), `PIN_PROFILE` set (`b3_correctness_gpu256.txt`).
Before it, EXP-0002 on `d854d98b`: 8 of 8 OK at 64 tokens against the CPU
reference.

A baseline build of upstream master at `d854d98b` does **not** currently exist,
so the two `upstream-*` rows were not run on the current base.

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

**Validation status: validated on both bases.**

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

**Re-validate after every rebase.** The traces are base-specific even though the
simulator is not: upstream's own commits can change the router's top-8 and
therefore the generated text.

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

**Gap: nothing from the pre-`research/` work was preserved.**
`moe-bench/.gitignore` excludes `out/` and `*.csv`, so the `78a8ae1` reference
tables above are traceable only to the prose of `moe-bench/RESULTS.md`, not to a
raw file. Fixed from EXP-0001 onward; the historical raw data is gone.
