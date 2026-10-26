# STATE

Restart checkpoint. Read this first in every session.

| | |
|---|---|
| Status | **`PAUSED`** |
| Iteration | 3 in progress (EXP-0003) |
| Experiment ID | EXP-0003 |
| Phase | implementation done and built; **stopped at the game check, before benchmarking** |
| Pause reason | **Game detected: Stellaris** (`D:\SteamLibrary\steamapps\common\Stellaris\stellaris.exe`), a real game executable under `steamapps`, not an idle launcher. Free host RAM had fallen to 9.41 GiB |
| Branch | `moe-gpu-optimize` |
| HEAD | `83ffbea00` (the `no_alloc` merge fix) |
| Upstream base | **`a4699400e`** - changed this iteration, see below |
| Build | current with the working tree, including the uncommitted `n_sample = 32` |
| Tree | **one uncommitted change**, see "Uncommitted work" |

## Protocol

One user authorization (`go`) = exactly one iteration:

Research -> Plan -> Code -> check for running games -> Benchmark and test ->
Evaluate -> Document -> Commit -> STOP.

While paused: do not monitor processes, do not recheck, do not schedule anything,
do no background work, and **do not resume when the game closes**. Wait for the
user.

`go` while paused = resume this iteration from here and finish it, overriding the
game check. Do not restart research or implementation.

## Uncommitted work

`src/llama-moe-cache.cpp`, **one line**, the whole of EXP-0003's change:

```c
// around line 501, in the branch taken when the cache is below one token's experts
n_sample = 32;   // was 16
```

The build already contains it. If the working tree is ever cleaned, re-apply that
one constant - nothing else is pending.

## The base changed mid-iteration

The user merged upstream master during this iteration. **Everything measured
before now was on `d854d98b`; the base is now `a4699400e`.**

New commits include `a4699400e llama : move no_alloc out of llama_hparams
(#31141)`, which moved `no_alloc` off `llama_hparams` and threaded it through the
`llama_moe_cache` constructor. The merge updated the signature and callers but
missed the host-pinning line this branch added, so the file stopped compiling the
moment it was touched. Fixed in `83ffbea00` by using the constructor argument
that is already in scope.

**This has the same consequence EXP-0001 was created to handle:** a base change
can perturb the router's top-8, which changes the generated text and therefore
the traces. So before quoting any simulated number on this base:

- the four traces in `research/results/experiments/EXP-0001/` were collected on
  `d854d98b`;
- the simulator validation and the EXP-0003 knee screen below both rest on them;
- EXP-0002's hardware figures were measured on `d854d98b`.

There is a cheap check for whether the router moved, folded into the plan below.

## Work already done this iteration

The sample-count screen is **complete and its raw data is committed** (the
`research` commit `e403ab429` holds `knee_prompt*.json` plus the new simulator
policies). Simulated on the `d854d98b` traces, mean of four traces:

| MiB | s8 | s16 | s24 | s32 | s48 | s64 | s96 | LFU | Belady |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16.91 | 18.68 | 19.24 | 19.55 | 19.84 | 19.84 | 19.83 | 16.57 | 32.01 |
| 384 | 24.63 | 26.21 | 26.71 | 26.92 | 27.10 | 27.16 | 27.14 | 22.53 | 43.51 |
| 528 | 31.12 | 32.33 | 32.64 | 32.84 | 32.92 | 33.03 | 32.94 | 27.69 | 51.65 |
| 586 | 33.24 | 34.29 | 34.54 | 34.68 | 34.78 | 34.80 | 34.78 | 29.45 | 54.07 |

Marginal gain per doubling, averaged over sizes and traces: **s8->s16 +1.40,
s16->s32 +0.62, s32->s64 +0.21, s64->s96 -0.04.** The curve saturates around
s48 to s64.

`s32` minus `s16` is **+0.39 to +0.87, sign-consistent on all four traces** at
256 to 586 MiB, which is what EXP-0003 set out to confirm.

**Why 32 and not 64**, which measures slightly better (+0.83 vs +0.62 average):
32 captures three quarters of the available gain for half the sampling work, and
`AGENTS.md` prefers the simpler change that does most of the job. The saturation
data is recorded so the choice can be revisited; it is a one-constant change
either way.

Slots per cache group, computed this iteration and worth keeping:

| MiB | group 0 (36 layers) | group 1 (1 layer) | group 2 (3 layers) |
|---:|---:|---:|---:|
| 256 | 124 | 2 | 9 |
| 384 | 187 | 4 | 14 |
| 528 | 258 | 6 | 20 |
| 586 | 286 | 6 | 22 |
| 645 | 315 | 7 | 25 |
| 1024 | 502 | 12 | 40 |
| 2048 | 1005 | 26 | 82 |

Two things follow. Group 1 never reaches `n_expert_used = 8` below the crossover,
which is why it is uncached and why layer 1 is missing from EXP-0002's per-layer
table. And group 2 has only 9 to 22 slots there, so sampling 16 or 32 of them
**oversamples with repeats** - which raises coverage rather than wasting work
(16 draws from 9 slots cover about 83% of them against 63% for 9 draws), so the
default is deliberately **not** clamped to `n_slots`.

## Exact next action on `go`

Resume EXP-0003 at the benchmark step. Implementation and build are done.

1. **Re-check free host RAM and the game.** `go` overrides the game check, but
   note in the record whether Stellaris was still running, because it competes
   for RAM and the GPU and will distort any timing.
2. **Hardware B1** with `research/results/experiments/EXP-0002/bench.sh` as the
   model, pointing the arms at `LLAMA_MOE_CACHE_SAMPLE=16` against the new
   default of 32, over 256, 384, 528, 586, plus 1024 and 2048 as inertness
   controls. Both arms measured in the same session, so the paired comparison is
   valid on the new base whatever the router did.
3. **Fold in the base-change check, which is nearly free:** the 2048 MiB row
   should reproduce `61.38% / 58033.62 MiB` exactly if the router is unchanged
   from `d854d98b`. If it does not, say so - the committed traces then need
   re-collecting before any simulated figure is quoted for `a4699400e`, and the
   EXP-0003 screen above becomes provisional.
4. Expect the hardware gain to match the simulated +0.39 to +0.87 within the 0.7
   to 0.8 point RNG margin.
5. **Watch the eviction path cost.** 32 samples is twice the work per fill, with
   the draw cap rising from 64 to 128. EXP-0002 argued this is negligible against
   the roughly 360 microseconds each 1.812 MiB upload takes, but neither
   iteration has measured it.
6. **Correctness gate B3** - mandatory, it is a source change.
   `./moe-bench/check-correctness.sh <model>`.
7. Then document EXP-0003 (`research/experiments/EXP-0003.md` is **not yet
   written**), update KNOWLEDGE_BASE.md Finding 7, BENCHMARKS.md, ROADMAP.md
   (R6), EXPERIMENTS.md, and commit the `n_sample` change on its own.

Throughput: do not claim it. Free RAM was 9.41 GiB with the game running, far
below the ~18 GiB where the floor is +-1.3%. EXP-0002's control row moved 6.9% at
17 GiB free.

## Environment facts needed to run anything

**Running a binary** needs the CUDA 12.6 runtime first on `PATH`; the system CUDA
is 13.2 and binaries otherwise exit with `0xC0000135` and no output:

```sh
export PATH="/c/Users/steve/cuda126/bin:$PATH"
```

**Building** additionally needs the MSVC environment, or every file fails with
`C1083: Cannot open include file: 'stdbool.h'`:

```powershell
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$cm = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
cmd /c "call `"$vc`" >nul 2>&1 && set `"PATH=C:\Users\steve\cuda126\bin;%PATH%`" && `"$cm`" --build build --target llama-bench llama-completion"
```

Python is 3.13.1 at `/c/Python313/python`. Model:
`C:\code\Codey\models\qwen3.6-35b\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`. See
ENVIRONMENT.md.

## Recommended experiment after EXP-0003

- **Re-validate on `a4699400e`** if step 3 shows the router moved: re-collect the
  four traces and re-run the simulator validation, as EXP-0001 did. This is now a
  recurring cost of every base change and should be a standing step.
- **OPEN_QUESTIONS.md Q6** - is the host-to-device copy actually overlapped with
  compute? Still unmeasured, needs no new hardware, and would reprioritize the
  roadmap if transfers turn out to be serialized.
- **Q7** - layer-to-layer expert prediction, the only remaining policy candidate
  with an unexploited information source. Pure analysis on committed traces.
- **Q10** - whether the distance term should also run above the crossover; needs
  more than four traces.
