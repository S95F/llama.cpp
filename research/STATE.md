# STATE

Restart checkpoint. Read this first in every session.

| | |
|---|---|
| Status | `COMPLETE` - iteration 7 complete; iteration 8 not started |
| Iteration | 7 (EXP-0007) complete |
| Experiment ID | EXP-0007 (R12 batched prefetch, Q18 fault attribution, R16b one level), **Adopt** |
| Phase | idle; the next `go` starts iteration 8 |
| Branch | `moe-gpu-optimize` |
| Last code commit | the EXP-0007 commit (iteration 7), on `b2586549c` |
| Upstream base | `a91eab67d`, through merge `2ca33f0d7` (2026-10-31) |
| Build | `llama.dll` rebuilt at 2026-11-01 03:22 local time, before the timed runs, from the source as it stood then; the `LLAMA_MOE_PREFETCH` comment was reworded at 04:41, after the runs (comment only, no code change) |
| Tree | clean after the EXP-0007 commit; nothing pushed |

## Last completed iteration

**Iteration 7, EXP-0007, 2026-11-01: Adopt** the opt-in
`LLAMA_MOE_PREFETCH=2`: after the router read-back, one `QueryWorkingSetEx`
call per layer drops the missed experts' host ranges already in the working
set, and one `PrefetchVirtualMemory` call reads the rest before the copies.
`=1` prefetches every missed range that is not fully pinned. A copy log
(`LLAMA_MOE_COPY_LOG`) records every upload and prefetch. 19 traced runs at
2048 MiB (a warm-up and nine arms by two reps, mirror order), the model evicted and
about 19968 MiB available at launch, held by a balloon (19880 to 20028;
18944 for `prof-lo`).

- **Gate G (CPU only) passed**: cold, one prefetch call plus the copies took
  0.24 to 0.32 of the faulting copies' time; over resident pages it is pure
  cost (0.12 to 0.16 us per page; 0.20 to 0.26 from standby).
- **H2 holds: mode 2 under the profile is 2.59 ms (6.2%) faster warm** (-2.60,
  -2.57), 4.29 ms (9.7%) over tokens 3-1022; under address order -1.74 (-3.9%).
  Mode 1 gains 0.96 under the profile and loses 2.03 under address order (it
  prefetches 112 MiB of mostly resident pages per pass).
- **H3 holds: cold and unpinned, mode 1 is 12.91 ms (16.5%) faster** over
  tokens 3-126, fault wait (slow-copy plus no-API idle) cut to 0.42.
- **H1 falsified, and Q18 settled differently than assumed**: 99.9% of the
  slow pageable copies are an expert's first read since the load; re-reads
  essentially never fault. Whether a first read faults depends on the expert's
  place in the file: the last 2.1 GiB never stayed resident after the load,
  and the head is lost in file order as the run's own memory grows (not the
  balloon). The profile faults more per pageable copy because more of its
  remainder is not yet read (composition, 103% of the gap over tokens 3-1022).
  EXP-0006's "pages leave RAM between uses" was wrong. New Q21, R20, R21 (the GPU-uploaded
  tensors' pages stay mapped after the load).
- **H4 holds**: 1 GiB less headroom, prefill +0.52 and +0.64 s, slow copies
  per token +1.30 and +1.28 over tokens 3-1022, +0.93 ms per token warm.
- Outputs identical in every run; B3 14 of 14. `GGML_CUDA_REGISTER_HOST=1` was
  found in the user's environment and is now controlled in every harness
  (ENVIRONMENT.md); B3 rerun with it exported as 0.
- Reviewed adversarially before the runs (five findings fixed) and after
  (summarized in EXP-0007.md; the records stayed in the session's workflow
  log). The post-run review rewrote the fault mechanism (file position, not
  the balloon) and added caveats (load-phase RAM dips the floor check cannot
  see, H3's fast-copy idle); no verdict changed.

**Side effect for the user:** the evictions and the balloon pushed other
processes' memory to the pagefile again; desktop apps may page back in slowly
for a while.

## Previous step: the upstream merge check (2026-10-31 to 11-01)

Merge `2ca33f0d7` of upstream `a91eab67d` (120 commits) checked: B1 exact, B3
11 of 11 byte-identical across the merge, test-backend-ops MUL_MAT_ID and
MUL_MAT_VEC_FUSION all pass, decode no change detected (B2 protocol, ABBA).
Paused once for a game. Detail in
`research/results/baseline/rebase-2ca33f0d7/` and the previous STATE in git.

## Protocol

One user authorization (`go`) = exactly one iteration:

Research -> Plan -> Code -> check for running games -> Benchmark and test ->
Evaluate -> Document -> Commit -> STOP.

The game check happens after implementation and before any resource-intensive
run. If a game is found: save state here, tell the user, stop everything, do not
recheck, and do not resume when the game closes. `go` while paused resumes and
overrides the check.

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
cmd /v:on /c "call `"$vc`" >nul 2>&1 && set `"PATH=C:\Users\steve\cuda126\bin;!PATH!`" && `"$cm`" --build build --target llama-bench llama-completion"
```

**Tracing** uses Nsight Systems 2025.6.3, not on `PATH`; BENCHMARKS.md B7 has the
command and `research/results/experiments/EXP-0004/long_runs.sh` the harness.
Captures go to `C:\nsys\` (outside git).

Python is 3.13.1 at `/c/Python313/python`. Model:
`C:\code\Codey\models\qwen3.6-35b\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`. See
ENVIRONMENT.md.

## Exact next action on `go`

Choose iteration 8 from ROADMAP priority 1, re-checked against the results:

- **R12b (cheap, decides the default):** the probed prefetch where it has not
  run: unpinned cold (128 tokens, `pg-pq` against `pg-pf` and `pg`), unpinned
  warm (1024 tokens), profile pinning at 15000 MiB, and the story prompt
  (`prompt_ood.txt`) under the profile. EXP-0007's `runs.sh` with new arms;
  add a pageable mode-2 row to B3. If mode 2 never loses, make it the default
  whenever the cache serves pageable experts (Windows), and say so in the PR
  notes.
- **Q21 then R21 (cheap measurement first):** right after the load, query
  whether the pages of the non-expert tensors uploaded to the
  GPU (2.0 GiB, kept mapped after the load on every OS; on Windows nothing of
  the file is unmapped) are in the working set, and map
  what of the file is resident at load, prefill and every 128 passes. If the
  non-expert pages are resident, release them (R21) and rerun `prof` and
  `prof-pq`: the tail should stay resident and the head front move later.
- **R20 (code):** read the lost pageable experts off the decode thread. Target:
  mode 2's 0.87 ms per token of blocking reads warm, 3.1 ms over tokens
  3-130. Bounded by the RAM budget; after Q21.
- R14 (sync idle 2.1 ms per token) stays next after these.

If the user prefers no new code: R12b alone, or R19's microbenchmark, or Q20's
llama-bench rep checks (now with `GGML_CUDA_REGISTER_HOST` set explicitly).

## Pending benchmark commands

None queued. `research/results/experiments/EXP-0007/runs.sh` is the current
timeline harness (clears the cache variables, balloon at two levels, eviction,
watchdog, typeperf, Nsight, selection trace and copy log; captures in
`C:/nsys/exp0007`), `attrib.py` the fault attribution and `analyze.py` the
pre-registered analysis. Run the correctness gate with
`PIN_PROFILE=C:/nsys/exp0005/profile_bcd.csv` to include profile pinning and
the profile-prefetch rows.
