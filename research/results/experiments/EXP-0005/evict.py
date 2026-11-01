#!/usr/bin/env python3
"""EXP-0005: push the model out of the Windows page cache without admin rights.

Emptying the standby list needs SeProfileSingleProcessPrivilege, which these
sessions do not have. Committing and touching private memory until only
--margin MiB stay available makes the memory manager repurpose the standby
pages instead, oldest and lowest priority first; releasing it leaves them free
and zeroed. What survives is at most the margin, which probe.py measures.

    python evict.py [--margin 512]

Prints the available, free and standby memory before and after, in MiB.
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import subprocess
import sys
import time

MIB = 1024 * 1024
MEM_COMMIT, MEM_RESERVE, MEM_RELEASE, PAGE_READWRITE = 0x1000, 0x2000, 0x8000, 0x04

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def avail_mib():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / MIB


def lists_mib():
    """Free+zero and standby list sizes from the performance counters."""
    ps = ("(Get-Counter '\\Memory\\Free & Zero Page List Bytes','\\Memory\\Standby Cache Normal Priority Bytes',"
          "'\\Memory\\Standby Cache Reserve Bytes','\\Memory\\Standby Cache Core Bytes').CounterSamples | "
          "% { [math]::Round($_.CookedValue/1MB) }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout.split()
        free, normal, reserve, core = (int(x) for x in out[:4])
        return dict(free=free, standby_normal=normal, standby_reserve=reserve, standby_core=core)
    except Exception as e:  # counters are informative only
        return dict(error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=int, default=512, help="MiB to leave available")
    ap.add_argument("--chunk", type=int, default=256, help="MiB per allocation")
    a = ap.parse_args()

    before = dict(avail=round(avail_mib()), **lists_mib())
    t0 = time.perf_counter()
    blocks = []
    chunk = a.chunk * MIB
    try:
        while avail_mib() - a.chunk > a.margin:
            p = k32.VirtualAlloc(None, chunk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
            if not p:
                break
            ctypes.memset(p, 1, chunk)
            blocks.append(p)
    finally:
        low = avail_mib()
        for p in blocks:
            k32.VirtualFree(p, 0, MEM_RELEASE)
    secs = time.perf_counter() - t0
    time.sleep(1)
    after = dict(avail=round(avail_mib()), **lists_mib())
    print(f"evict: touched {len(blocks) * a.chunk} MiB in {secs:.1f} s, available fell to {low:.0f} MiB; "
          f"before {before} after {after}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
