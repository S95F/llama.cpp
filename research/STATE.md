# STATE

Restart checkpoint. Read this first in every session.

| | |
|---|---|
| Status | `READY` |
| Iteration | 0 (setup) complete; iteration 1 not started |
| Experiment ID | none active; next is EXP-0001 |
| Phase | awaiting user authorization (`go`) |
| Pause reason | n/a - setup is not an iteration, see Protocol below |
| Branch | `moe-gpu-optimize` |
| Last code commit | `3db731656` (sampled least-used victim). Later commits are `moe-bench/` and `research/` only, so the branch tip moves without changing behaviour |
| Upstream base | `d854d98ba` (= `upstream/master` as of the last successful fetch) |
| Build | `build/` at `d8df72697`, code-current - the commits after it touch no source |
| Tree | clean at session start and after the setup commit |

## Protocol

One user authorization (`go`) = exactly one iteration, in this order:

Research -> Plan -> Code -> check for running games -> Benchmark and test ->
Evaluate -> Document -> Commit -> STOP.

The game check happens after implementation and before any resource-intensive
run. If a game is found: save state here, tell the user, stop everything, and do
not recheck or resume on our own. `go` while paused resumes and overrides the
check.

## Last completed step

Session of 2026-10-25 to 10-26: set up `research/` and consolidated the prior findings
from `moe-bench/RESULTS.md` into it. No code touched, no benchmark run.

## Environment facts needed to run anything

The system CUDA is now **13.2**, but `build/` links against the **12.6** install
in `C:\Users\steve\cuda126`. The binaries exit with `0xC0000135`
(STATUS_DLL_NOT_FOUND) unless that directory is first on `PATH`:

```powershell
$env:PATH = 'C:\Users\steve\cuda126\bin;' + $env:PATH
```

`cmake` and `ninja` are not on `PATH` either; the build uses the VS 2019
BuildTools copies recorded in `build/CMakeCache.txt`. See ENVIRONMENT.md.

## Exact next action on `go`

Start **EXP-0001**. Candidate selection and the reasoning behind it are in
ROADMAP.md; the current top candidate is item R1 (re-collect the four traces on
the `d854d98b` base and re-validate the simulator against it), because every
simulated table in KNOWLEDGE_BASE.md still rests on traces collected on the old
`78a8ae1` base and upstream has since changed the router's top-8 selection. Until
that is redone, no simulated number can be quoted for the current base, which
blocks the policy work in R2 and R3.

Confirm before spending GPU time:
1. tree clean, `git log -1` = `ab24fafa7` or later;
2. `PATH` fixed as above and `llama-bench --version` exits 0;
3. free host RAM. At session start it was **14 GiB** against an 18.32 GiB expert
   working set, which is *less* than any prior session (18, then 16.1 GiB) and is
   enough to distort throughput. Hit rate and uploaded bytes stay exact.

## Pending benchmark commands

None queued. BENCHMARKS.md holds the standard definitions.

## Recommended experiment after EXP-0001

ROADMAP.md item R2: close the gap to Belady below the crossover, where the
branch gets 15.8 to 32.0% hit rate against Belady's 31.9 to 53.9%.
