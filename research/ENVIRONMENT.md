# ENVIRONMENT

Measured on 2026-10-25. Re-verify at the start of every session: the CUDA
mismatch below was introduced by a toolkit upgrade between sessions and silently
breaks every binary.

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, 8188 MiB, compute capability 8.9 |
| Driver | 621.01 |
| Link | PCIe 4.0 x8, `current` = `max` for both gen and width, about 14 GB/s usable |
| CPU | AMD Ryzen 7 9700X, 8 cores / 16 threads |
| RAM | 31.11 GiB total; **14 GiB free at session start** |
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
cmd /c "call `"$vc`" >nul 2>&1 && set `"PATH=C:\Users\steve\cuda126\bin;%PATH%`" && `"$cm`" --build build --target llama-bench llama-completion"
```

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
- **No bus-level counter.** PCIe throughput is derived from uploaded bytes over
  time, never measured on the bus, so every GB/s figure here is a lower bound.
  `nvidia-smi dmon` pcie columns were not used.
- **Host RAM is smaller than the working set.** 31.11 GiB total against 18.32 GiB
  of expert weights plus a desktop session, so mmap page-cache misses contaminate
  any timing taken with little RAM free. This also makes the pinning result
  permanently confounded on this machine (KNOWLEDGE_BASE.md, Finding 4).
- **`upstream` could not be fetched in this session** (`git fetch upstream`
  exited 53, no network). `upstream/master` locally is `d854d98ba`, which equals
  the branch base, so "no new upstream commits" here means "none known locally",
  not "none exist".
- **Only one MoE model.** No second MoE architecture, expert count, or
  quantization has been tested.
