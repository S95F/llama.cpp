#!/usr/bin/env python3
"""EXP-0007 gate (ROADMAP R12): copying a layer's missed expert ranges out of
the GGUF mapping, with and without one PrefetchVirtualMemory call over them
first. CPU only.

The file is mapped as llama_mmap maps it (CreateFileMapping PAGE_READONLY,
MapViewOfFile FILE_MAP_READ). A trial is one layer and k experts not used by
any other trial and not adjacent to one, so trials do not warm each other. Its
ranges are the experts' slices of ffn_gate_exps, ffn_up_exps and ffn_down_exps,
copied tensor by tensor as copy() uploads them.

    copy      memmove each range into a buffer (the pageable upload's staging copy)
    prefetch  one PrefetchVirtualMemory call over the page-aligned ranges, then the same copies

Each trial runs in three states, in this order:

    cold      after evict.py (run it first; the script does not)
    standby   after the process's working set is emptied: the pages are in RAM, not mapped
    resident  right after: the pages are in the working set

Trials of the two methods alternate, so both see the same disk and RAM state.
Writes one CSV row per trial and state, then a summary per k and state.

    python prefetch_bench.py <model.gguf> --out <csv> [--k 1 3 8] [--trials 20] [--seed 7]
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import csv
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "gguf-py"))

from gguf import GGUFReader  # noqa: E402

PAGE = 4096
GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL = 0x80000000, 1, 3, 0x80
PAGE_READONLY, FILE_MAP_READ = 0x02, 0x04

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wt.HANDLE
k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
k32.CreateFileMappingW.restype = wt.HANDLE
k32.CreateFileMappingW.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.DWORD, wt.LPCWSTR]
k32.MapViewOfFile.restype = ctypes.c_void_p
k32.MapViewOfFile.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_size_t]
k32.GetCurrentProcess.restype = wt.HANDLE
k32.SetProcessWorkingSetSize.argtypes = [wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t]


class RANGE(ctypes.Structure):
    _fields_ = [("VirtualAddress", ctypes.c_void_p), ("NumberOfBytes", ctypes.c_size_t)]


k32.PrefetchVirtualMemory.argtypes = [wt.HANDLE, ctypes.c_size_t, ctypes.POINTER(RANGE), wt.ULONG]
k32.PrefetchVirtualMemory.restype = wt.BOOL


def map_file(path):
    h = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if h in (None, wt.HANDLE(-1).value):
        sys.exit(f"CreateFileW failed: {ctypes.get_last_error()}")
    m = k32.CreateFileMappingW(h, None, PAGE_READONLY, 0, 0, None)
    if not m:
        sys.exit(f"CreateFileMappingW failed: {ctypes.get_last_error()}")
    base = k32.MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0)
    if not base:
        sys.exit(f"MapViewOfFile failed: {ctypes.get_last_error()}")
    return base


def prefetch_entries(ranges):
    """Page-aligned, sorted and merged, as llama_prefetch builds them."""
    pages = sorted((a // PAGE * PAGE, (a + n + PAGE - 1) // PAGE * PAGE) for a, n in ranges)
    merged = []
    for lo, hi in pages:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    arr = (RANGE * len(merged))()
    for i, (lo, hi) in enumerate(merged):
        arr[i].VirtualAddress = lo
        arr[i].NumberOfBytes = hi - lo
    return arr


def run_trial(method, ranges, entries, buf):
    """Returns (prefetch_us, copy_us)."""
    t0 = time.perf_counter_ns()
    if method == "prefetch":
        if not k32.PrefetchVirtualMemory(k32.GetCurrentProcess(), len(entries), entries, 0):
            sys.exit(f"PrefetchVirtualMemory failed: {ctypes.get_last_error()}")
    t1 = time.perf_counter_ns()
    for a, n in ranges:
        ctypes.memmove(buf, a, n)
    t2 = time.perf_counter_ns()
    return (t1 - t0) / 1000, (t2 - t1) / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 8])
    ap.add_argument("--trials", type=int, default=20, help="per method and k")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    # layer -> [(offset, expert bytes)] for gate, up, down
    layers = {}
    for t in GGUFReader(a.model).tensors:
        for kind in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            if t.name.endswith(f"{kind}.weight"):
                il = int(t.name.split(".")[1])
                n_exp = int(t.shape[2])
                layers.setdefault(il, {})[kind] = (int(t.data_offset), int(t.n_bytes) // n_exp, n_exp)
    layers = {il: [d[k] for k in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")] for il, d in layers.items() if len(d) == 3}
    n_exp = next(iter(layers.values()))[0][2]
    base = map_file(a.model)
    buf = ctypes.create_string_buffer(max(s for v in layers.values() for _, s, _ in v))

    rng = random.Random(a.seed)
    used = set()  # (layer, expert) taken or adjacent to a taken one

    def pick(k):
        while True:
            il = rng.choice(sorted(layers))
            free = [e for e in range(n_exp) if (il, e) not in used]
            if len(free) < 4 * k:
                continue
            es = []
            for e in rng.sample(free, len(free)):
                if (il, e) not in used and len(es) < k:
                    es.append(e)
                    used.update({(il, e - 1), (il, e), (il, e + 1)})
            if len(es) == k:
                return il, sorted(es)

    trials = []
    for k in a.k:
        for i in range(2 * a.trials):
            il, es = pick(k)
            ranges = [(base + off + e * size, size) for off, size, _ in layers[il] for e in es]
            trials.append(dict(k=k, method="copy" if i % 2 == 0 else "prefetch", layer=il, experts=es,
                               ranges=ranges, entries=prefetch_entries(ranges)))
    rng.shuffle(trials)

    rows = []
    for state in ("cold", "standby", "resident"):
        if state == "standby":
            k32.SetProcessWorkingSetSize(k32.GetCurrentProcess(), ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        for t in trials:
            pf, cp = run_trial(t["method"], t["ranges"], t["entries"], buf)
            rows.append(dict(state=state, k=t["k"], method=t["method"], layer=t["layer"],
                             experts=" ".join(map(str, t["experts"])), bytes=sum(n for _, n in t["ranges"]),
                             n_entries=len(t["entries"]), prefetch_us=round(pf, 1), copy_us=round(cp, 1),
                             total_us=round(pf + cp, 1)))

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'state':9} {'k':>2} {'MiB':>5} | {'copy med':>9} {'mean':>8} | {'pf+copy med':>11} {'mean':>8} {'pf call med':>11} | {'med ratio':>9}")
    for state in ("cold", "standby", "resident"):
        for k in a.k:
            c = [r["total_us"] for r in rows if r["state"] == state and r["k"] == k and r["method"] == "copy"]
            p = [r for r in rows if r["state"] == state and r["k"] == k and r["method"] == "prefetch"]
            pt = [r["total_us"] for r in p]
            mib = statistics.mean(r["bytes"] for r in p) / 2**20
            print(f"{state:9} {k:2d} {mib:5.2f} | {statistics.median(c):9.0f} {statistics.mean(c):8.0f} | "
                  f"{statistics.median(pt):11.0f} {statistics.mean(pt):8.0f} {statistics.median(r['prefetch_us'] for r in p):11.1f} | "
                  f"{statistics.median(pt) / statistics.median(c):9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
