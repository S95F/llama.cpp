#!/usr/bin/env python3
"""EXP-0006: EXP-0005's watchdog plus per-process memory counters.

Every --every seconds it records the available physical memory and, for the
benchmark process, its working set, private bytes, page fault count (hard and
soft) and file read bytes (ReadFile only; page-ins of a mapped file are not
counted). It kills the benchmark if available memory stays below --min MiB
for --for seconds, and writes the lowest value seen to --out.

    python monitor.py --csv proc.csv --out low.txt
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "EXP-0005"))
from evict import avail_mib  # noqa: E402

MIB = 1024 * 1024
PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ, TH32CS_SNAPPROCESS = 0x1000, 0x10, 0x2

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
k32.OpenProcess.restype = wt.HANDLE


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                  "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


def pids(image):
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    e = PROCESSENTRY32W()
    e.dwSize = ctypes.sizeof(e)
    out = []
    ok = k32.Process32FirstW(snap, ctypes.byref(e))
    while ok:
        if e.szExeFile.lower() == image.lower():
            out.append(e.th32ProcessID)
        ok = k32.Process32NextW(snap, ctypes.byref(e))
    k32.CloseHandle(snap)
    return out


def sample(h):
    m = PROCESS_MEMORY_COUNTERS_EX()
    m.cb = ctypes.sizeof(m)
    io = IO_COUNTERS()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(m), m.cb):
        return None
    k32.GetProcessIoCounters(h, ctypes.byref(io))
    return m.WorkingSetSize / MIB, m.PrivateUsage / MIB, m.PageFaultCount, io.ReadTransferCount / MIB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="llama-completion.exe")
    ap.add_argument("--min", type=float, default=300.0)
    ap.add_argument("--for", dest="hold", type=float, default=20.0)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # the benchmark is the first process of that name started after this one
    old = set(pids(a.image))
    pid = None
    for _ in range(120):
        pid = next((p for p in pids(a.image) if p not in old), None)
        if pid:
            break
        time.sleep(0.5)
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid) if pid else None
    low = avail_mib()
    below = 0.0
    killed = False
    exit_code = wt.DWORD()
    with open(a.csv, "w", newline="\n") as f:
        f.write("t_unix,avail_mib,ws_mib,private_mib,page_faults,read_mib\n")
        while h:
            v = avail_mib()
            low = min(low, v)
            s = sample(h)
            if s is None:
                break
            f.write(f"{time.time():.3f},{v:.0f},{s[0]:.1f},{s[1]:.1f},{s[2]},{s[3]:.1f}\n")
            f.flush()
            below = below + a.every if v < a.min else 0.0
            if below >= a.hold:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
                killed = True
                break
            # STILL_ACTIVE
            if k32.GetExitCodeProcess(h, ctypes.byref(exit_code)) and exit_code.value != 259:
                break
            time.sleep(a.every)
    with open(a.out, "w") as f:
        f.write(f"lowest available {low:.0f} MiB{', KILLED: below %.0f MiB for %.0f s' % (a.min, a.hold) if killed else ''}"
                f"{'' if pid else ', benchmark process not found'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
