#!/usr/bin/env python3
"""EXP-0005: estimate how much of the model's expert data is in the page cache.

Windows has no unprivileged mincore(). Instead this reads one 4 KiB page at a
fixed set of random offsets inside the expert tensors and times each read: a
page in the cache comes back in microseconds, a page on disk in about 100 us
or more. The fixed seed reads the same pages every time, so the probe itself
warms only those pages (and whatever the cache manager reads around them).

    python probe.py <model.gguf> [--n 256] [--threshold-us 40]
"""

import argparse
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "gguf-py"))

from gguf import GGUFReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--threshold-us", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()

    spans = [(int(t.data_offset), int(t.n_bytes)) for t in GGUFReader(a.model).tensors if "_exps" in t.name]
    total = sum(n for _, n in spans)
    rng = random.Random(a.seed)
    offs = []
    for _ in range(a.n):
        x = rng.randrange(total)
        for off, n in spans:
            if x < n:
                offs.append((off + x) // 4096 * 4096)
                break
            x -= n

    us = []
    with open(a.model, "rb", buffering=0) as f:
        for off in offs:
            t0 = time.perf_counter_ns()
            f.seek(off)
            f.read(4096)
            us.append((time.perf_counter_ns() - t0) / 1000)
    cached = sum(1 for u in us if u < a.threshold_us)
    print(f"probe: {cached}/{len(us)} expert pages cached ({100 * cached / len(us):.1f}%), "
          f"median {statistics.median(us):.0f} us, p90 {sorted(us)[int(0.9 * len(us))]:.0f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
