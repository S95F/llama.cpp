#!/usr/bin/env python3
"""EXP-0006 (ROADMAP R16): hold the available RAM at a fixed value for a run.

Allocates and locks enough private memory that the available physical memory
falls to --target MiB, then holds it until --stop names an existing file (or
the process is killed). Locked pages cannot be trimmed to the pagefile, so
the allocation stays the same size however hard the run presses on memory.
If less than --target is available, it holds nothing and says so.

    python balloon.py --target 19968 --out balloon.txt --stop stop.flag

The first line of --out is written once the balloon is in place, so the
caller can wait for it.
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "EXP-0005"))
from evict import avail_mib  # noqa: E402

MIB = 1024 * 1024
MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE = 0x1000, 0x2000, 0x04

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
k32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k32.GetCurrentProcess.restype = wt.HANDLE
k32.SetProcessWorkingSetSizeEx.argtypes = [wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wt.DWORD]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, required=True, help="MiB to leave available")
    ap.add_argument("--chunk", type=int, default=64, help="MiB per allocation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop", required=True)
    a = ap.parse_args()

    before = avail_mib()
    want = int(max(0.0, before - a.target) // a.chunk) * a.chunk
    locked = True
    blocks = []
    if want > 0:
        # VirtualLock is limited by the minimum working set; raising it needs only the
        # "increase a process working set" right that every user has
        if not k32.SetProcessWorkingSetSizeEx(k32.GetCurrentProcess(), (want + 256) * MIB, (want + 1024) * MIB, 0):
            locked = False
        for _ in range(want // a.chunk):
            p = k32.VirtualAlloc(None, a.chunk * MIB, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
            if not p:
                break
            ctypes.memset(p, 1, a.chunk * MIB)
            if locked and not k32.VirtualLock(p, a.chunk * MIB):
                locked = False
            blocks.append(p)
    held = len(blocks) * a.chunk
    time.sleep(0.5)
    with open(a.out, "w") as f:
        f.write(f"balloon: available {before:.0f} MiB, target {a.target:.0f}, held {held} MiB, "
                f"locked {'yes' if locked or not held else 'no, err %d' % ctypes.get_last_error()}, "
                f"available now {avail_mib():.0f} MiB\n")

    # without the lock, touching every page keeps the balloon in the working set
    page = 4096
    while not os.path.exists(a.stop):
        if not locked:
            for p in blocks:
                for off in range(0, a.chunk * MIB, page):
                    ctypes.c_char.from_address(p + off).value
        time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
