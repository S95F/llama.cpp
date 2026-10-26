# STATE

Restart checkpoint. Read this first in every session.

| | |
|---|---|
| Status | `READY` |
| Iteration | 1 complete (EXP-0001). Iteration 2 not started |
| Experiment ID | none active; next is EXP-0002 |
| Phase | awaiting user authorization (`go`) |
| Pause reason | n/a - iteration 1 finished normally; no game was detected |
| Branch | `moe-gpu-optimize` |
| Last code commit | `3db731656` (sampled least-used victim). Everything since touches only `moe-bench/` and `research/`, so the branch tip moves without changing behaviour |
| Upstream base | `d854d98ba` (= `upstream/master` as of the last successful fetch) |
| Build | `build/` at `d8df72697`, code-current - the commits after it touch no source |
| Tree | clean after the iteration 1 commit |

## Protocol

One user authorization (`go`) = exactly one iteration, in this order:

Research -> Plan -> Code -> check for running games -> Benchmark and test ->
Evaluate -> Document -> Commit -> STOP.

The game check happens after implementation and before any resource-intensive
run. If a game is found: save state here, tell the user, stop everything, and do
not recheck or resume on our own. `go` while paused resumes and overrides the
check.

## Last completed step

**Iteration 1, EXP-0001, 2026-10-26: Adopt.** Re-collected the four expert
selection traces on base `d854d98b` and re-validated `moe-bench/moe_sim.py`
against hardware.

- Simulator reproduces llama.cpp's **integer hit and fill counts** at 2048 MiB on
  all four traces (prompt: 50092 / 31516 in both). Stronger than EXP-0000, which
  validated one trace to 2 decimal places.
- Every simulated figure in KNOWLEDGE_BASE.md reproduces **within +0.45 points**;
  no policy ranking changed; the crossover did not move.
- New methodological result: one policy's hit rate varies **4.2 to 7.3 points
  between traces**, so policies must be compared paired per trace with the sign
  agreeing on all four. Recorded in BENCHMARKS.md as the screening rule.
- Finding 2's routing-skew range widened (k=8: 25.9 to 33.2% against 31 to 33%),
  entirely because of `prompt_d`, the multilingual trace. Mechanism intact.
- FAILED_APPROACHES.md corrected for full LFU and S3-FIFO; neither rejection
  overturned.
- **Raw results committed for the first time in this project**, in
  `research/results/experiments/EXP-0001/`.

No source changed. No throughput claim was made: only 12.34 GiB of host RAM was
free against an 18.32 GiB expert working set.

## Environment facts needed to run anything

The system CUDA is now **13.2**, but `build/` links against the **12.6** install
in `C:\Users\steve\cuda126`. The binaries exit with `0xC0000135`
(STATUS_DLL_NOT_FOUND) and print nothing unless that directory is first on
`PATH`:

```sh
export PATH="/c/Users/steve/cuda126/bin:$PATH"
```

`cmake` and `ninja` are not on `PATH` either; the build uses the VS 2019
BuildTools copies recorded in `build/CMakeCache.txt`. Python is 3.13.1 at
`/c/Python313/python`. See ENVIRONMENT.md.

Model: `C:\code\Codey\models\qwen3.6-35b\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`.

## Exact next action on `go`

Start **EXP-0002**, ROADMAP.md item **R2** - close more of the Belady gap below
the crossover. The branch currently closes 50 to 58% of it at 256 to 528 MiB but
only **27% at 586 MiB**, which suggests the remaining loss is structured rather
than spread.

Run it in this order:

1. **First, the Q2 per-layer breakdown.** Pure offline analysis on the traces in
   `research/results/experiments/EXP-0001/`; no GPU time, no game check needed to
   start it. Add a per-layer hit rate and reuse-distance report to
   `moe-bench/trace_stats.py`. This decides which R2 candidate to implement:
   concentrated loss means per-layer slot budgets, spread loss means a better
   victim score.
2. Implement the one candidate the breakdown points at, in
   `src/llama-moe-cache.cpp`, inside the existing `moe_cache_slru`. It must stay
   inert at and above the crossover.
3. Screen with the simulator over `256,384,528,586,645,1024,2048` on all four
   traces, **paired per trace, sign-consistent on all four** (BENCHMARKS.md).
4. Game check, then validate the winner on hardware with B1, and run the
   correctness gate B3 - the first code change since EXP-0000, so B3 is
   mandatory.

Do not claim throughput unless free host RAM is at least ~16 GiB; it was 12.34
GiB during EXP-0001. Check it first and say so either way.

## Pending benchmark commands

None queued. BENCHMARKS.md B1 and B3 are the ones EXP-0002 will need.

## Recommended experiment after EXP-0002

Whichever of these the R2 result does not settle:

- **OPEN_QUESTIONS.md Q6** - is the host-to-device copy actually overlapped with
  compute? Never measured, needs no new hardware, and if transfers turn out to be
  serialized the fix is overlap rather than fewer bytes, which would reprioritize
  the whole roadmap. The highest-value unanswered question.
- **Q7** - layer-to-layer expert prediction, now pure analysis on committed
  traces.
