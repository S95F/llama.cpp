# STATE

Restart checkpoint. Read this first in every session.

| | |
|---|---|
| Status | `COMPLETE` - iteration 6 done; awaiting `go` for iteration 7 |
| Iteration | 6 (EXP-0006), authorized 2026-10-31, completed 2026-10-31 |
| Experiment ID | EXP-0006 (R17 online profile, R16 fixed headroom, Q17), **Adopt** |
| Phase | stopped after the commit. Game check before the runs was clear (2026-10-31, Battle.net and Steam launchers only) |
| Pause reason | n/a |
| Branch | `moe-gpu-optimize` |
| Last code commit | the iteration 6 commit on top of `24c58282f` (`git log -1`): `LLAMA_MOE_PIN_ONLINE` in `src/llama-moe-cache.cpp`, two B3 arms, `research/` |
| Upstream base | `63b4b303b`, through the user's merge `5df437a57` (2026-10-26) |
| Build | `llama.dll` 2026-10-31 20:41, current for this CUDA build (`llama-completion.exe` links it dynamically) |
| Tree | clean after the commit |

## Protocol

One user authorization (`go`) = exactly one iteration:

Research -> Plan -> Code -> check for running games -> Benchmark and test ->
Evaluate -> Document -> Commit -> STOP.

The game check happens after implementation and before any resource-intensive
run. If a game is found: save state here, tell the user, stop everything, do not
recheck, and do not resume when the game closes. `go` while paused resumes and
overrides the check.

## Last completed step

**Iteration 6, EXP-0006, 2026-10-31: Adopt** the opt-in
`LLAMA_MOE_PIN_ONLINE=<N>` (pin the experts the run's own first N decode passes
select most; no trace file) and the fixed-headroom harness. R17 and R16's
reproducibility half, plus Q17 on one prompt. 21 traced 1024-token runs at
2048 MiB (a warm-up and ten arms by two reps, ABBA), the model evicted and
19968 MiB available held by a balloon at every launch. All four hypotheses
hold, H1 and H2 narrowly.

- **Warm, the online profile is within 1 ms of a matching file profile**
  (+0.65, -0.16, +0.83, +0.90 ms over tokens 387-1022), slower in both rounds
  in three of four arms.
- **It pays up front**: the passes before the registration are pageable (75-92
  ms per token) and registering takes 0.54-0.59 s, so over tokens 3-1022 it
  beats address order by only about 1% and trails the file profile by about 3 s
  per run at the default cap (6-7 s at 15000 MiB, where one token stalls for
  1.4-1.8 s).
- **Out of domain it wins**: on a children's story the file profile leaves
  34.5% pageable (22.3% on prompt a) and the online profile is 6.2% faster than
  it warm, 0.64 s ahead end to end.
- **A profile's pageable remainder faults more**: every remaining slow copy is
  pageable, and under a profile 2.0 to 2.7% (default cap) to 12% (15000 MiB) of
  the pageable expert copies fault, against 0.9% under address order (level 4:
  the cold tail). Slow-copy idle stays at 1.6 to 3.8 ms per token. That, and
  passes before the registration slower than assumed, are why the replay's exact
  share (0.1-0.2 points) did not convert to milliseconds (off by up to 1.5 ms).
  New Q18; R12 raised.
- **Load-time registration is followed by 0.9 to 2.9 s of GPU idle** before the
  prompt, though the register calls at load can be fast (2-3 ms for address
  order). New Q19, R19.
- **Reps agree within 0.8%** and no run-position drift is detectable (slope
  +0.011, 95% CI -0.036 to +0.058, against EXP-0005's 0.13; half of that is not
  excluded). The repeated EXP-0005 arms ran 0.7 to 1.4 ms slower,
  mostly slow-copy idle, at lower launch headroom (untested; R16b).
- B3 11 of 11 byte-identical at 256 tokens, both online arms included.
- Adversarially reviewed before the runs (`prerun_review.json`) and after
  (`review.json`): a one-pass offset in `coverage.py` fixed (aligned file kept
  beside the pre-run one), conclusions corrected (break-even, GiB units,
  the fragments not the cause, H4's falsifier about 40% of EXP-0005's slope).
- Protocol gap: the warm-up went below the 300 MiB floor for about 5 s; the
  script checked only for a kill.

**Side effect for the user:** the evictions and the balloon pushed other
processes' memory to the pagefile, so desktop apps (and an idle ComfyUI) may
page back in slowly for a while.

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

Start **EXP-0007: ROADMAP R12 - one batched prefetch per layer of the missed
experts' pages**, aimed at the cold tail profile pinning leaves pageable
(OPEN_QUESTIONS Q18), with R16b's second balloon level folded in.

1. **Attribute the faults first (Q18, cheap).** Behind an env var, log per
   pageable expert upload its layer, expert, bytes and host-side copy time in
   `llama_moe_cache_set_async` (or extend `LLAMA_MOE_CACHE_TRACE`), then count
   how often each slow expert was selected before it faulted. Confirms or
   refutes the cold-tail explanation.
2. **Microbenchmark gate (CPU only).** After `evict.py`, read scattered 576 to
   840 KiB expert ranges of the GGUF by touch against one `llama_prefetch`
   (`PrefetchVirtualMemory`) call over the same ranges. If batched is not
   faster, stop and record R12 as rejected.
3. **Code (small).** In `copy()`, after the router read-back and before the
   fills, collect the missed experts' pageable host ranges (skip pinned ones)
   and issue one `llama_prefetch` per layer. Opt-in (for example
   `LLAMA_MOE_PREFETCH=1`). `llama_prefetch_rows` / `llama_prefetch` exist in
   `src/llama-impl.cpp` and `src/llama-mmap.cpp`.
4. **B3** with the prefetch on, pinned and pageable.
5. **Arms** with `research/results/experiments/EXP-0006/runs.sh` (fix its
   warm-up check to stop below the 300 MiB floor first): file profile at the
   default cap and address order, each with and without prefetch; optionally
   prof15k and a pageable cold 128-token pair, where R12's large gain would be.
   Two balloon levels (19968 MiB and about 2 GiB less) for R16b.
6. Report tokens 387-1022 and 3-1022, slow copies per token and slow-copy idle.

If the user prefers no new code: R19's microbenchmark (`cudaHostRegister` on
an evicted mapping, time the first submission, with and without a prefetch) or
R13's two gates (Q13 second-stream concurrency under WDDM; Q7 next-layer
recall from the traces).

## Pending benchmark commands

None queued. `research/results/experiments/EXP-0006/runs.sh` is the current
timeline harness (balloon, eviction, watchdog, typeperf, Nsight; captures in
`C:\nsys\exp0006`) and `research/results/experiments/EXP-0003/bench.sh` the B1
harness. Run the correctness gate with
`PIN_PROFILE=C:/nsys/exp0005/profile_bcd.csv` to include profile pinning (the
profile is EXP-0001's `trace_prompt_b/c/d.csv` concatenated); the two online
arms run without it.
