#!/usr/bin/env python3
"""EXP-0005: how much pinned host memory the driver gives this process.

Allocates pinned memory with cudaMallocHost in --chunk MiB pieces until the
driver refuses, prints the total, and frees everything. Uses the CUDA 12.6
runtime the binaries link (cudart64_12.dll on PATH).

    python pin_budget.py [--chunk 512]
"""
import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evict import avail_mib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--dll", default="C:/Users/steve/cuda126/bin/cudart64_12.dll")
    a = ap.parse_args()
    rt = ctypes.CDLL(a.dll)
    ptrs = []
    t0 = time.perf_counter()
    before = avail_mib()
    err = 0
    while True:
        p = ctypes.c_void_p()
        err = rt.cudaMallocHost(ctypes.byref(p), ctypes.c_size_t(a.chunk << 20))
        if err != 0:
            break
        ptrs.append(p)
    secs = time.perf_counter() - t0
    low = avail_mib()
    for p in ptrs:
        rt.cudaFreeHost(p)
    print(f"pin_budget: cudaMallocHost gave {len(ptrs) * a.chunk} MiB in {a.chunk} MiB chunks in {secs:.1f} s "
          f"before error {err}; available {before:.0f} -> {low:.0f} MiB")


if __name__ == "__main__":
    main()
