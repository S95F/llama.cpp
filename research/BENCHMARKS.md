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

Derived quantities (effective GB/s) are lower bounds, never bus measurements.

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

### Reference, base `d854d98b` (current)

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

Greedily decodes the same prompt under each configuration and diffs against
`--moe-cache-mib 0`. Configurations: `cache2048-lru`, `cache2048-auto`,
`cache2048-s16`, `cache528-lru`, `cache528-auto`, `cache1024-pinned`, plus
`upstream-off` and `upstream-2048` when a baseline build is given.

**Reference: all configurations byte-identical on both bases.** This is a
pass/fail gate, not a measurement. Any difference is a bug.

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
measured that residual on hardware at 0.7 to 0.8 points.

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

## Raw results

`research/results/baseline/` and `research/results/experiments/`. JSON or CSV,
one directory per experiment ID, never overwritten.

Committed so far:

- `research/results/experiments/EXP-0001/` - the four `d854d98b` traces
  (`trace_*.csv`, 10202 lines each), the cache counters (`telemetry_2048.txt`),
  per-trace simulator output (`sim_validate_*.json`, `sim_sweep_*.json`), the
  routing statistics (`trace_stats.txt`), and the two scripts that produced and
  reduced them (`collect.sh`, `aggregate.py`).

**Gap: nothing from the pre-`research/` work was preserved.**
`moe-bench/.gitignore` excludes `out/` and `*.csv`, so the `78a8ae1` reference
tables above are traceable only to the prose of `moe-bench/RESULTS.md`, not to a
raw file. Fixed from EXP-0001 onward; the historical raw data is gone.
