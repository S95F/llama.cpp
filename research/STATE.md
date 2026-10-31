# STATE

Restart checkpoint. Read this first in every session.

| | |
|---|---|
| Status | `READY` |
| Iteration | 4 complete (EXP-0004). Iteration 5 not started |
| Experiment ID | none active; next is EXP-0005 |
| Phase | awaiting user authorization (`go`) |
| Pause reason | n/a |
| Branch | `moe-gpu-optimize` |
| Last code commit | `dabb2b026` (`n_sample = 32`); EXP-0004 changed no llama.cpp code |
| Upstream base | **`63b4b303b`**, through the user's merge `5df437a57` (2026-10-26); moved from `a4699400e`. B1 rebase check passed |
| Build | binaries from 2026-10-26 08:40, current for this CUDA build: the merge touched only the hexagon and OpenCL backends |
| Tree | clean after the iteration 4 commit |

## Protocol

One user authorization (`go`) = exactly one iteration:

Research -> Plan -> Code -> check for running games -> Benchmark and test ->
Evaluate -> Document -> Commit -> STOP.

The game check happens after implementation and before any resource-intensive
run. If a game is found: save state here, tell the user, stop everything, do not
recheck, and do not resume when the game closes. `go` while paused resumes and
overrides the check.

## Last completed step

**Iteration 4, EXP-0004, 2026-10-31: Adopt (measurement; no code change).**
Answered OPEN_QUESTIONS.md Q6 with a CUDA timeline of every generated token
(Nsight Systems, 15 traced runs, six cache/pinning arms, three 1024-token runs).

- **The expert uploads never overlap compute**: 0 ns in every token. One stream
  and two host waits per MoE layer make it so by construction.
- **Finding 4 was wrong in its premise.** Its 3.4 to 5.5 GB/s were bytes over
  whole-token time. The copies run at 12.8 GB/s pinned and 8.2 to 13 GB/s
  pageable (slower while the host stages the next copy). Rewritten.
- **A fresh 128-token run is mostly page faults**: 91.7 of 139.6 ms per token at
  2048 MiB is the GPU idle while the host faults expert pages in from disk at
  about 175 us per 64 KiB. They die away over about 400 tokens.
- **Warm, a 2048 MiB token is 56.2 ms = 12.8 compute + 27.1 DMA + 0.1 read-back
  + 16.1 host waits.** DMA is the largest part, so bytes still matter.
- **Bounds** (Finding 10, projections): overlap can hide at most 9.4 ms (17%);
  pinning every expert about 35 ms, but this host caps pinning near half of RAM,
  so about 38 to 39 at the cap; the per-layer host round trip is 2.6 to 3.4 ms.
- Pinning's gain is now split (Q3 in part): cold, 78 to 86% avoided faults; warm,
  mostly the removal of staging contention.
- Adversarially reviewed: one draft claim refuted and part of another, eight
  corrected in all, before anything was written down (`review.json`); then every
  figure in the written docs was checked against the raw results twice, and all
  72 findings applied (`doc_check.json`, `doc_recheck.json`).

**Hazard found: run order.** Pageable runs that followed a pinned run started
partly warm (prefill 3.7 to 7.3 s instead of 13.4 to 13.8 s). That confounds the
256 MiB arm and the long pageable runs, and it is why R10 comes first next.

The standing rebase check passed on `5df437a57`: B1 at 2048 MiB reproduced
`61.38% / 58033.62 MiB` exactly, so the committed traces stay valid.

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

**Tracing** uses Nsight Systems 2025.6.3, not on `PATH`; BENCHMARKS.md B7 has the
command and `research/results/experiments/EXP-0004/long_runs.sh` the harness.
Captures go to `C:\nsys\` (outside git).

Python is 3.13.1 at `/c/Python313/python`. Model:
`C:\code\Codey\models\qwen3.6-35b\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`. See
ENVIRONMENT.md.

## Exact next action on `go`

Start **EXP-0005: ROADMAP R11 - pin up to the WDDM cap, most-missed ranges
first**, on **R10's controlled baseline**. Finding 10 bounds it at about
-17 to -18 ms per token warm at 2048 MiB (56.2 -> about 38 to 39, a projection),
the largest bounded lever that this host allows.

1. **R10 first, inside the same iteration.** Fix the page-cache state at the
   start of every run and record the prefill time and prefill slow copies as a
   coldness check (cold pageable is 13.4 to 13.8 s and about 6480 slow copies;
   cold with 42% pinned 9.3 to 9.9 s and about 3600; take a cold reference for
   each pinned size). For warm state, run a
   discarded generation in the same process first, or alternate arms so each
   follows the same kind of run. A truly cold arm needs the user to flush the
   standby list with an admin tool; ask rather than skip it.
2. **No code first.** The default pins layers 0 to 15 plus layer 16's gate and
   down tensors (address order, first fit per tensor), about 49% of the warm
   upload bytes for 42% of the expert bytes, because the lowest layers are the
   least cacheable. At 15266 MiB address order leaves layer 32's up tensor and
   layers 33 to 39 pageable, about 18% of the bytes; leaving the least-missed
   layers pageable instead would leave about 13 to 14%. So compute first, from
   EXP-0001's traces through the simulator (CPU only), the miss coverage of a
   layer-level and an expert-level order at the same MiB.
3. **Arms**, B7 timelines, 1024 tokens, warm window 387 to 1022: pinning off,
   default (7842 MiB), `LLAMA_MOE_PIN_MIB=15000`, and `--load-mode none` as the
   R5 comparison (it may fail to allocate; record that). Write the
   frequency-order code only if step 2 says address order leaves something.
4. **Watch free RAM**: 15 GiB locked of 31 leaves about 1.5 to 3 GiB. Abort an
   arm if the desktop starts paging.
5. A code change, if any, needs the B3 correctness gate.

Hit rate and uploaded bytes do not change with pinning, so this experiment's
claims rest on the timeline decomposition, with the rep-to-rep spread from step 1
as its noise floor.

Alternatives if the host cannot pin 15000 MiB: **R12** (batched prefetch of
missed experts; small code change, gated by a CPU-only microbenchmark) or the two
cheap gates of **R13** (Q13 concurrency microbenchmark; Q7 offline recall).

## Pending benchmark commands

None queued. `research/results/experiments/EXP-0004/long_runs.sh` is the current
timeline harness and `research/results/experiments/EXP-0003/bench.sh` the B1
harness.
