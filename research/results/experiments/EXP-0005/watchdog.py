#!/usr/bin/env python3
"""EXP-0005: stop a run before pinned memory starves the desktop.

Polls the available physical memory every --every seconds and kills the
benchmark process if it stays below --min MiB for --for seconds in a row.
Writes the lowest value seen to --out when the benchmark exits or is killed.

    python watchdog.py --image llama-completion.exe --out low.txt
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evict import avail_mib  # noqa: E402


def running(image):
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/NH"], capture_output=True, text=True).stdout
    return image.lower() in out.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="llama-completion.exe")
    ap.add_argument("--min", type=float, default=300.0)
    ap.add_argument("--for", dest="hold", type=float, default=20.0)
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # wait for the benchmark to start
    for _ in range(60):
        if running(a.image):
            break
        time.sleep(1)
    low = avail_mib()
    below = 0.0
    killed = False
    n = 0
    while True:
        v = avail_mib()
        low = min(low, v)
        below = below + a.every if v < a.min else 0.0
        if below >= a.hold:
            subprocess.run(["taskkill", "/F", "/IM", a.image], capture_output=True)
            killed = True
            break
        n += 1
        # tasklist is slower than the memory query, so check the process less often
        if n % 5 == 0 and not running(a.image):
            break
        time.sleep(a.every)
    with open(a.out, "w") as f:
        f.write(f"lowest available {low:.0f} MiB{', KILLED: below %.0f MiB for %.0f s' % (a.min, a.hold) if killed else ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
