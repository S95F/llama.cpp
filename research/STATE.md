# STATE

Restart checkpoint. Read this first in every session.

| | |
|---|---|
| Status | `READY` |
| Iteration | 5 complete (EXP-0005). Iteration 6 not started |
| Experiment ID | none active; next is EXP-0006 |
| Phase | awaiting user authorization (`go`) |
| Pause reason | n/a |
| Branch | `moe-gpu-optimize` |
| Last code commit | the iteration 5 commit: `LLAMA_MOE_PIN_PROFILE`, uploads split at pinned-range edges, B3 on the GPU at 256 tokens |
| Upstream base | `63b4b303b`, through the user's merge `5df437a57` (2026-10-26) |
| Build | `llama.dll` 2026-10-31 19:07, current for this CUDA build (`llama-completion.exe` 17:46 links it dynamically) |
| Tree | clean after the iteration 5 commit |

## Protocol

One user authorization (`go`) = exactly one iteration:

Research -> Plan -> Code -> check for running games -> Benchmark and test ->
Evaluate -> Document -> Commit -> STOP.

The game check happens after implementation and before any resource-intensive
run. If a game is found: save state here, tell the user, stop everything, do not
recheck, and do not resume when the game closes. `go` while paused resumes and
overrides the check.

## Last completed step

**Iteration 5, EXP-0005, 2026-10-31: Adopt** the opt-in
`LLAMA_MOE_PIN_PROFILE` and the upload split. R11, with R10 and R5 folded in.
Twelve 1024-token traced runs at 2048 MiB: six arms, two reps each, ABBA, the
model evicted before every run.

- **Pinning near the budget, most-selected experts first, takes the warm token
  from 52.95 to 36.43 ms (-31%).** Profile order beats address order at the same
  MiB: -2.69 ms (-6.8%) at 15000 MiB (matched tokens 387-841), -3.55 ms (-8.1%)
  at the default cap, the same sign in both rounds.
- **The share left pageable is predictable offline**: the simulator replay plus
  GGUF byte ranges match CUPTI to 0.1 points over the traces' tokens.
- **Gain split at 98% pinned**: 86% DMA rate and fast-copy idle, 13% faults.
- **Pinned budget about 15 GiB.** At that size the desktop nearly ran out of
  memory: address order left 243 to 814 MiB available, and the watchdog killed
  one run. Profile order left about 1.5 GiB more (1764 and 2336 MiB, median).
- **`--load-mode none` is about 80% pinned**, not 100%: a 4 GiB chunk silently
  falls back to pageable. It runs at 38.21 ms (matched window) and loads in 15 s.
- **Copy-edge rule**: a copy that starts in a registered range and runs past it
  fails. Expert-level pinning crashed until every upload was split at range
  edges. The timed runs used a per-cache split; the process-wide table and the
  scheduler fallback came from the review, for cases these runs never hit (a
  second context, a declined tensor).
- **R10 falsified in part**: eviction fixed the launch state (0 to 7 of 256
  pages cached), but reps drifted with run position and available RAM
  (inseparable in that design), which rose 1.5 GiB through the session. Tokens
  3-130 are early, not cold.
- **B3 was comparing CPU with GPU arithmetic.** With the cache off (or a group
  with no slots) decode computes the experts on the CPU, and the text diverges
  just after token 64. The gate now computes every expert on the GPU and runs
  256 tokens: 10 of 10 byte-identical, profile pinning included. The knowledge
  base's correctness invariant is rewritten.
- Adversarially reviewed (`review.json`): eight of nine claims corrected, two
  latent crash risks in the code fixed, confounds listed.

**Side effect for the user:** each eviction pushed other processes' memory to
the pagefile, so desktop apps (and an idle ComfyUI) may page back in slowly for
a while.

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

Start **EXP-0006: ROADMAP R17 - an online selection profile**, timed with
**R16's fixed RAM headroom**.

1. **Code (small).** While the cache runs with `GGML_CUDA_REGISTER_HOST=1` and a
   new opt-in (for example `LLAMA_MOE_PIN_PROFILE=online:<N>`), count selections
   per (layer, expert) over the first N generated tokens in `copy()`. Then, between
   two tokens, unregister nothing and register by count up to the cap, using
   `ranges_by_use`'s ranking on in-memory counts. Register under the range
   table's lock, which already orders it against uploads. Offline it leaves
   5.5 to 7.6% (N = 64) or 2.8 to 4.2% (N = 128) of miss bytes pageable at
   15000 MiB (`coverage_2048_runcaps.json`, `online`).
2. **B3** with the online mode (add a `cache2048-pinonline` arm).
3. **Harness**: copy `research/results/experiments/EXP-0005/runs.sh`, and add a
   balloon (a process holding a fixed allocation for the whole run; 2 to 4 GiB
   would push available RAM below the watchdog's 300 MiB floor at 15000 MiB
   pinned, where EXP-0005 left a median of 243 to 814 MiB (address order) and
   1764 to 2336 MiB (profile order) without one; run the balloon arms at a lower
   cap or keep the balloon inside that headroom) and per-process typeperf
   counters (`\Process(llama-completion)\IO Read Bytes/sec`, `Working Set`) plus
   `\Memory\Standby Cache*` and `\Memory\Modified Page List Bytes`.
4. **Arms**, tokens 387-1022, ABBA or randomized, three reps if time allows:
   address order 15000, file profile 15000 (EXP-0005's), online N = 64 and
   N = 128 at 15000. Optionally the default cap, where order matters most.
5. **Out-of-domain check (Q17)**: one arm pair on a code or non-English prompt,
   with the file profile against the online one.
6. Watch available RAM. At 15000 MiB pinned, address order went below 300 MiB.

If the user prefers no new code: R13's two cheap gates (Q13, a second-stream
concurrency microbenchmark under WDDM; Q7, next-layer recall from the traces)
decide whether overlap, bounded at about 9 ms of a 36 ms pinned token, is
reachable at all.

## Pending benchmark commands

None queued. `research/results/experiments/EXP-0005/runs.sh` is the current
timeline harness (eviction, probe, watchdog, typeperf, Nsight) and
`research/results/experiments/EXP-0003/bench.sh` the B1 harness. Run the
correctness gate with `PIN_PROFILE=C:/nsys/exp0005/profile_bcd.csv` to include
profile pinning (the profile is EXP-0001's `trace_prompt_b/c/d.csv`
concatenated).
