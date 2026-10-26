# STATE

Restart checkpoint. Read this first in every session.

| | |
|---|---|
| Status | `READY` |
| Iteration | 3 complete (EXP-0003). Iteration 4 not started |
| Experiment ID | none active; next is EXP-0004 |
| Phase | awaiting user authorization (`go`) |
| Pause reason | n/a - iteration 3 paused at the game check and finished on `go` |
| Branch | `moe-gpu-optimize` |
| Last code commit | `dabb2b026` (`n_sample = 32`), with `83ffbea00` before it (the `no_alloc` merge fix) |
| Upstream base | **`a4699400e`** - moved from `d854d98b` this iteration |
| Build | current with the source, built via `vcvars64` |
| Tree | clean after the iteration 3 commit |

## Protocol

One user authorization (`go`) = exactly one iteration:

Research -> Plan -> Code -> check for running games -> Benchmark and test ->
Evaluate -> Document -> Commit -> STOP.

The game check happens after implementation and before any resource-intensive
run. If a game is found: save state here, tell the user, stop everything, do not
recheck, and do not resume when the game closes. `go` while paused resumes and
overrides the check.

## Last completed step

**Iteration 3, EXP-0003, 2026-10-26: Adopt.** Raised the sampled-slot count from
16 to 32 below the crossover - one constant.

- Hardware: hit rate **+0.53 to +0.86 points**, PCIe traffic **0.76 to 1.12%**
  lower, matching the simulated prediction within 0.18 points.
- **Byte-identical above the crossover**, verified against two earlier sessions'
  recorded figures.
- Correctness gate **8 of 8 byte-identical**, the first run on the new base.
- The screen mapped the whole curve: marginal gain per doubling +1.40, +0.62,
  +0.21, **-0.04**, saturating near 48 to 64. **32 leaves +0.21 on the table**
  deliberately, for half the sampling work.
- Found and fixed a build break from the user's upstream merge (`83ffbea00`):
  `a4699400e` moved `no_alloc` off `llama_hparams`, and the branch's pinning line
  still read `model.hparams.no_alloc`. A stale `.obj` had hidden it.
- **Paused mid-iteration**: the game check found Stellaris running. Resumed on
  `go`, by which time it had exited, so nothing was measured alongside it.

Cumulative against upstream, below the crossover, on hardware:

| MiB | upstream | EXP-0000 | EXP-0002 | **EXP-0003** |
|---:|---:|---:|---:|---:|
| 256 | 0.00% | 13.40% | 16.34% | **17.00%** |
| 384 | 0.00% | 21.32% | 23.16% | **24.03%** |
| 528 | 1.04% | 27.53% | 29.30% | **29.96%** |
| 586 | 20.93% | 29.64% | 31.38% | **31.91%** |

### The base moved, and the router did not

The user merged 33 upstream commits mid-iteration. The one-run rebase check (B1
at 2048 MiB) came out **exact** at both 1024 and 2048 MiB, so the router's top-8
is unchanged and **the four committed traces remain valid for `a4699400e`**. No
re-collection was needed, unlike the `78a8ae1` to `d854d98b` move that EXP-0001
had to handle.

**Make that check a standing step after every rebase**: B1 at 2048 MiB should
reproduce `61.38% / 58033.62 MiB`. It costs one run and tells you whether every
simulated figure is still quotable.

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

## Exact next action on `go`

Start **EXP-0004**: **OPEN_QUESTIONS.md Q6 - is the host-to-device copy actually
overlapped with compute?**

This is now the highest-value open question. Finding 4 concluded generation is
bandwidth-bound on the copy and treated the derived 3.4 to 5.5 GB/s as a lower
bound *because* compute overlaps the copies - but **that overlap has never been
measured.** If transfers turn out to be serialized with compute, the lever is
overlap rather than fewer bytes, and the whole roadmap reprioritizes. Three
iterations of policy work have been spent cutting bytes on an unverified
assumption.

Start with the cheap version, which needs no new instrumentation:

1. For each cache size, compare measured time per token against
   `max(compute_time, bytes / measured_bandwidth)` and see which bound is active.
   Compute time can be estimated from a cache size large enough that uploads are
   small, or from `--moe-cache-mib 0` against a fully resident configuration.
2. If that is ambiguous, add CUDA event timing around the upload in
   `llama_moe_cache::copy` behind an env flag, in the style of the existing
   `LLAMA_MOE_CACHE_LOG`, and compare the summed upload time against wall time
   per token.
3. No source change is needed for step 1, so **no correctness gate** until step 2
   adds one.

Note the hazard: this is a **timing** question on a host where throughput has
been unresolvable for two iterations (control rows moving 4 to 7%). Prefer
measures that are ratios within a single run over comparisons across runs, and
state plainly if the host cannot answer it.

Alternatives if Q6 looks unanswerable here: **Q7** (layer-to-layer prediction,
pure offline analysis on committed traces, the only remaining policy candidate
with an unexploited information source) or **Q10** (now has a +2.10 point
hardware datapoint at 1024 MiB and needs only more traces, which are cheap -
four took 4 minutes).

## Pending benchmark commands

None queued. `research/results/experiments/EXP-0003/bench.sh` is the current B1
harness; point its arms at whatever two configurations an experiment compares.
