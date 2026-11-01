# ENVIRONMENT

Measured on 2026-10-25. Re-verify at the start of every session: the CUDA
mismatch below was introduced by a toolkit upgrade between sessions and silently
breaks every binary.

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, 8188 MiB, compute capability 8.9 |
| Driver | 621.01, **WDDM** (CUDA driver API 13.5); `HwSchMode` not set in the registry, so GPU scheduling is at the OS default |
| Link | PCIe 4.0 x8, `current` = `max` for both gen and width; 15.75 GB/s raw, **pinned copies measured at 12.8 GB/s** (EXP-0004) |
| Copy engines | **1** (`asyncEngineCount`), `concurrentKernels = 1`, `pageableMemoryAccess = 0` (`device_query.json`, EXP-0004) |
| CPU | AMD Ryzen 7 9700X, 8 cores / 16 threads |
| RAM | 31.11 GiB total; 14 GiB free at the 2026-10-25 session start, 16.45 to 18.24 GiB during EXP-0004, 17.09 GiB available at the start of EXP-0005 and 19.67 to 21.14 GiB before its runs' evictions (rising through the session); EXP-0006 held 19966 to 20022 MiB available at every launch with a balloon, EXP-0007 19880 to 20028 (18957 to 18989 at its lower level), with 523 to 5852 MiB the lowest during a run |
| OS | Windows 11 Pro 10.0.26300 |

Queried with:

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
```

## Toolchain

| | |
|---|---|
| Compiler | MSVC 14.29.30133 (VS 2019 BuildTools) |
| Generator | Ninja, from the VS 2019 BuildTools CMake |
| CUDA used by `build/` | **12.6**, at `C:\Users\steve\cuda126` |
| CUDA on the system `PATH` | **13.2**, at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2` |
| Build type | Release, `GGML_CUDA=ON` |
| Profiler | Nsight Systems 2025.6.3, `C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe` (not on `PATH`) |

nsys logs `driver 13.5 not supported, using 13.2 libraries` as a diagnostic on
this driver. EXP-0004 checked that the traces are still complete (exact event
counts, every expert byte accounted for), so it is harmless here, but re-check
completeness after a driver change.

Neither `cmake` nor `ninja` is on `PATH`. The paths are in
`build/CMakeCache.txt`:

```
CMAKE_COMMAND      C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe
CMAKE_MAKE_PROGRAM C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe
CMAKE_CUDA_COMPILER C:/Users/steve/cuda126/bin/nvcc.exe
```

### The CUDA 12.6 / 13.2 trap

`build/bin/*.exe` are thin launchers; the code is in the matching `-impl.dll` and
in `ggml-cuda.dll`, which links the **12.6** runtime. With only 13.2 on `PATH`
every binary exits immediately with `-1073741515` = `0xC0000135`
(STATUS_DLL_NOT_FOUND) and prints nothing, which looks like a crash rather than a
missing dependency. Prepend the 12.6 bin directory in every shell that runs a
binary or a benchmark script:

```powershell
$env:PATH = 'C:\Users\steve\cuda126\bin;' + $env:PATH
```

Verified working afterwards: `llama-bench --version` exits 0 and reports
`version: 0.7.0-dev (build 11896, commit d8df72697)`, 1 CUDA device, 8187 MiB.

A clean rebuild against 13.2 is an option but has not been done, and it would
invalidate comparability with every number recorded so far.

### Building needs the MSVC environment, not just the paths

Found in EXP-0002. Invoking the VS 2019 `cmake.exe` from a plain shell is not
enough: without `INCLUDE` and `LIB` the compiler cannot find the standard
headers, and **every** translation unit fails with

```
fatal error C1083: Cannot open include file: 'stdbool.h': No such file or directory
fatal error C1083: Cannot open include file: 'algorithm': No such file or directory
```

which reads like a broken checkout rather than a missing environment. Source
`vcvars64.bat` first. From PowerShell, in one `cmd` invocation so the variables
survive:

```powershell
$vc = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$cm = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
cmd /v:on /c "call `"$vc`" >nul 2>&1 && set `"PATH=C:\Users\steve\cuda126\bin;!PATH!`" && `"$cm`" --build build --target llama-bench llama-completion"
```

`/v:on` and `!PATH!` matter. With `%PATH%`, cmd expands the variable when it
parses the line, before `vcvars64.bat` runs, so the MSVC directories drop out
of `PATH`. C++ files still compile (CMake calls `cl.exe` by full path), but
every `.cu` file fails with `nvcc fatal : Cannot find compiler 'cl.exe' in
PATH`. Found at the 2026-10-31 upstream merge, the first build since
2026-10-26 that recompiled CUDA files.

Running a built binary needs only the CUDA 12.6 `PATH` fix above, not vcvars.

## Model

| | |
|---|---|
| Path | `C:\code\Codey\models\qwen3.6-35b\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` |
| Size | 20.82 GiB |
| Architecture | Qwen3.6-35B-A3B, 40 MoE layers, 256 experts per layer, 8 used per token |
| Quantization | Unsloth `UD-Q4_K_XL` (mixed, so three cache groups - see ARCHITECTURE.md) |

Other models on disk, none MoE, available if a dense control is needed:
`Meta-Llama-3.1-8B-Instruct-Q4_K_M` (4.58 GiB),
`Llama-3.2-3B-Instruct-Q4_K_M` (1.88 GiB),
`Llama-3.2-1B-Instruct-Q4_K_M` (0.75 GiB).

## Repository

| | |
|---|---|
| Working dir | `C:\code\llama.cpp-fork` |
| Branch | `moe-gpu-optimize`, tracking `origin/moe-gpu-optimize` |
| Upstream base | `a91eab67d` through merge `2ca33f0d7`, 2026-10-31 (was `63b4b303b` via `5df437a57`, and before that `a4699400e`, `d854d98b` and `78a8ae1`) |
| `origin` | `https://github.com/S95F/llama.cpp.git` (the fork) |
| `upstream` | `https://github.com/ggml-org/llama.cpp.git` |

Sibling checkouts exist and are referenced by the older results:
`C:\code\llama.cpp-src` (held the `build-base` upstream baseline build),
`C:\code\llama.cpp-moe`.

## Known capability gaps

Recorded explicitly so no document implies these were tested.

- **No second GPU and no multi-device split.** Everything is single-device.
- **PCIe width and generation cannot be varied** - the link is fixed at 4.0 x8,
  and the firmware exposes no narrower setting that has been tried. The
  constrained-bandwidth axis is approximated by changing cache size, which
  changes bytes moved rather than link speed.
- **No bus-level counter.** Since EXP-0004 copy rates are measured per copy from
  CUPTI timestamps (B7), not derived from token time, but nothing measures the
  bus itself. `nvidia-smi dmon` pcie columns were not used.
- **Host RAM is smaller than the working set.** 31.11 GiB total against 18.32 GiB
  of expert weights plus a desktop session, so mmap page-cache misses contaminate
  any timing taken with little RAM free. EXP-0004 measured them: a decode fault
  reads 64 KiB synchronously in about 175 us, and a fresh 128-token pageable run
  at 586 to 2048 MiB spends 55 to 66% of its mean token on them
  (KNOWLEDGE_BASE.md, Finding 9).
- **The OS keeps part of the model cached between processes after a pinned run**
  (the next pageable run prefilled in 3.7 to 7.3 s instead of 13.4 to 13.8 s;
  after a pageable run the next one started cold), so the page-cache state a run
  starts in depends on what ran before. **Flushing it needs an admin tool** (the
  standby list), which these sessions do not run, but EXP-0005's `evict.py`
  empties it by memory pressure without one (0 to 7 of 256 probed pages cached;
  see the eviction bullet below). That fixes the launch state, not the state at
  prefill, which the load's prefetch refills.
- **Pinned memory is capped near half of RAM** by WDDM's budget: about 15 GiB
  (EXP-0005: `cudaMallocHost` refused at 15360 MiB after 14848; 15243 to 15266
  MiB granted at other times). The branch pins at most RAM/4 by default (7842
  MiB of whole tensors here, 7943 MiB in profile order); `LLAMA_MOE_PIN_MIB`
  raises it. **At about 15 GiB pinned the desktop runs close to empty**: address
  order at 15000 MiB left 243 to 814 MiB available, and one run was stopped by
  the watchdog at 205 MiB. Other processes holding pinned memory count against
  the same budget.
- **Eviction without admin rights trims every other process** (EXP-0005).
  `evict.py` commits and touches about 20 GiB, which pushes the desktop's working
  sets to the pagefile. They page back in during the next run (system-wide disk
  and paging counters include them), and available RAM drifted up 1.5 GiB over
  the session. Interactive apps may be sluggish for a while after a run.
- **A balloon holds the launch headroom** (EXP-0006, `balloon.py`): it commits
  and locks memory until available RAM reaches the target (19968 MiB), holds it
  until the run ends, and is then released. It needs no admin rights. Its
  pressure adds to the eviction's, so the desktop's processes page out further;
  after a session of 21 runs, desktop apps and an idle ComfyUI server may page
  back in slowly.
- **Other GPU and RAM users.** An idle ComfyUI server and other processes held
  about 1.1 GiB of VRAM through EXP-0004 and 0005 (7051 or 7075 MiB free in
  EXP-0004's logs, 7051 in every EXP-0005 log). Steam, Battle.net (launchers,
  idle) and Discord were running; no game was. CUPTI traces only this process,
  so "idle" cannot exclude another process's GPU use.
- **Fetching `upstream` may fail in a sandboxed session** (`git fetch upstream`
  exited 53, no network, on 2026-10-25). When it does, "no new upstream commits"
  means "none known locally", not "none exist". The base has since moved twice
  through merges the user performed outside these sessions (to `a4699400e`, then
  to `63b4b303b` via `5df437a57`), so check the base rather than assuming it.
  The 2026-10-31 fetch worked, and the merge to `a91eab67d` (`2ca33f0d7`, 120
  upstream commits) was done in-session at the user's request.
- **`GGML_CUDA_REGISTER_HOST=1` is set in the user's environment** (Windows User
  scope, `HKCU\Environment`, which was last written 2026-10-24 17:49 -06:00,
  before EXP-0001), and it is the only `LLAMA`/`GGML` variable set there. Every
  process that does not set it pins host experts: at the default cap, in
  address order, whenever the MoE cache is on. Found by EXP-0007's pre-run
  review. EXP-0004 to EXP-0007's timed harnesses set it explicitly for every arm.
  `check-correctness.sh` (B3) did not before EXP-0007 and now exports 0. The
  llama-bench scripts of EXP-0002 and EXP-0003 (B2) and of the merge check do
  not set it, so they most likely ran pinned. Their processes' environments
  were not recorded, so this is not certain. Set it explicitly, or unset it, in
  every harness.
- **Only one MoE model.** No second MoE architecture, expert count, or
  quantization has been tested.
