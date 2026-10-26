# STATE

Restart checkpoint. Read this first in every session.

| | |
|---|---|
| Status | `READY` |
| Iteration | 2 complete (EXP-0002). Iteration 3 not started |
| Experiment ID | none active; next is EXP-0003 |
| Phase | awaiting user authorization (`go`) |
| Pause reason | n/a - iteration 2 finished normally; no game was detected |
| Branch | `moe-gpu-optimize` |
| Last code commit | the EXP-0002 commit below (`src/llama-moe-cache.cpp`) |
| Upstream base | `d854d98ba` (= `upstream/master` as of the last successful fetch) |
| Build | current with the source, built via `vcvars64` - see below |
| Tree | clean after the iteration 2 commit |

## Protocol

One user authorization (`go`) = exactly one iteration, in this order:

Research -> Plan -> Code -> check for running games -> Benchmark and test ->
Evaluate -> Document -> Commit -> STOP.

The game check happens after implementation and before any resource-intensive
run. If a game is found: save state here, tell the user, stop everything, and do
not recheck or resume on our own. `go` while paused resumes and overrides the
check.

## Last completed step

**Iteration 2, EXP-0002, 2026-10-26: Adopt.** Scored the sampled eviction victim
by the known cyclic distance to its layer's next visit, instead of by use count
alone.

- Chosen over per-layer slot budgets **by measurement**: the Q2 per-layer
  breakdown showed the gap to Belady is near-uniform across layers (stdev 3.23)
  while per-layer hit rate is already explained by routing skew (r = +0.979), so
  the loss is a timing loss, not an allocation loss.
- Hardware: hit rate **+1.74 to +2.94 points** over the previous policy at 256 to
  586 MiB, PCIe traffic a further **2.3 to 3.4%** lower. Against upstream, 16.3 to
  31.4% hit rate where upstream gets 0.00 to 20.93%, and 16.3 to 28.5% less
  traffic.
- Simulator: sign-consistent on all four traces at all eight sizes (+1.06 to
  +2.73), and it predicted the hardware within 0.28 points.
- **Byte-identical above the crossover** on hardware at 645, 1024 and 2048 MiB.
- Correctness gate **8 of 8 byte-identical**, including two new cases.
- `LLAMA_MOE_CACHE_DIST=0` reproduces EXP-0001's hardware numbers **exactly**
  (384: 21.32% / 114174.28 MiB), so the refactor did not disturb the old policy.
- **Throughput inconclusive and no claim made**: the 1024 MiB control row moved
  -6.9% on code B1 proves is byte-identical. Per-run spread was +-16 to 27%.

New knowledge base entries: Findings 7 and 8. New question: Q10.

## Environment facts needed to run anything

**Running a binary** needs the CUDA 12.6 runtime first on `PATH`; the system CUDA
is 13.2 and the binaries exit with `0xC0000135` and no output otherwise:

```sh
export PATH="/c/Users/steve/cuda126/bin:$PATH"
```

**Building** additionally needs the MSVC environment, or every file fails with
`C1083: Cannot open include file: 'stdbool.h'`, which looks like a broken
checkout:

```powershell
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$cm = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
cmd /c "call `"$vc`" >nul 2>&1 && set `"PATH=C:\Users\steve\cuda126\bin;%PATH%`" && `"$cm`" --build build --target llama-bench llama-completion"
```

Python is 3.13.1 at `/c/Python313/python`. Model:
`C:\code\Codey\models\qwen3.6-35b\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`. See
ENVIRONMENT.md.

## Exact next action on `go`

Start **EXP-0003**, ROADMAP.md item **R6** - raise the sampled-slot count from 16
to 32 below the crossover. It is the cheapest remaining item and its gain is
**already screened**: +0.39 to +0.87 points of hit rate, sign-consistent on all
four traces at 256 to 645 MiB (`research/results/experiments/EXP-0002/screen_*.json`,
policy `ramp-s32-age8-dist`).

1. Change `n_sample = 16` to 32 in the sub-crossover branch of
   `src/llama-moe-cache.cpp` (around the "MoE cache is too small" warning).
   One constant; `LLAMA_MOE_CACHE_SAMPLE` already overrides it for A/B.
2. Rebuild with `vcvars64` as above.
3. Game check.
4. Hardware B1 at 256, 384, 528, 586 with `LLAMA_MOE_CACHE_SAMPLE=16` against the
   new default, plus 1024 as the inertness control. Confirm the hardware gain
   matches the simulated +0.39 to +0.87 within the 0.7 to 0.8 point RNG margin.
5. **Watch for a cost in the eviction path**: 32 samples is twice the work per
   fill. EXP-0002 argued the cost is negligible against the traffic but did not
   measure it. If throughput cannot resolve it (likely), say so rather than
   implying there is no cost.
6. Correctness gate B3 - mandatory, it is a source change.

Throughput: do not claim it unless free host RAM is at least ~18 GiB **and** the
control row comes in under about 2%. EXP-0002's control moved 6.9% at ~17 GiB
free, so expect it to be unresolvable again.

## Pending benchmark commands

None queued. `research/results/experiments/EXP-0002/bench.sh` is the B1 harness
and can be reused by pointing its arms at `LLAMA_MOE_CACHE_SAMPLE=16` vs default.

## Recommended experiment after EXP-0003

- **OPEN_QUESTIONS.md Q6** - is the host-to-device copy actually overlapped with
  compute? Still never measured, needs no new hardware, and if transfers are
  serialized the fix is overlap rather than fewer bytes, which would reprioritize
  the whole roadmap. The highest-value unanswered question.
- **Q7** - layer-to-layer expert prediction. After EXP-0002 this is the only
  remaining policy candidate with an unexploited information source; routing skew
  and the cyclic sweep are both now used. Pure analysis on committed traces.
- **Q10** - whether the distance term should also run above the crossover. Needs
  more than four traces; collecting ten to twenty is cheap (four took 4 minutes).
