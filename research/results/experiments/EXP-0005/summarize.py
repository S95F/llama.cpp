#!/usr/bin/env python3
"""EXP-0005: one row per traced run - the coldness check, the load, and the
per-token decomposition over fixed token windows.

    python summarize.py <tag>...            (reads timeline_<tag>.json, log_<tag>.txt,
                                             the capture <cap>/<tag>.sqlite)
    python summarize.py --table runs.json   (prints the comparison table)

Coldness check: the expert copies of the prefill (host-to-device copies of at
least 512 KiB before the first logits read-back), and how many of them ran
below 1.5 GB/s on the host side, which is what page faults look like
(BENCHMARKS.md B7).
"""

import argparse
import collections
import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = [(3, 130), (131, 386), (387, 1022), (3, 1022)]


def prefill(db, expert_min=512 * 1024, slow_gbps=1.5):
    con = sqlite3.connect(db)
    oper = {r[1]: r[0] for r in con.execute("SELECT id, name FROM ENUM_CUDA_MEMCPY_OPER")}
    htod = [v for k, v in oper.items() if "HTOD" in k.upper()]
    dtoh = [v for k, v in oper.items() if "DTOH" in k.upper()]
    rows = list(con.execute("SELECT start, end, bytes, copyKind, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"))
    sizes = collections.Counter(r[2] for r in rows if r[3] in dtoh and r[2] >= 64 * 1024)
    logits = sizes.most_common(1)[0][0]
    first = min(r[1] for r in rows if r[3] in dtoh and r[2] == logits)
    api = dict((c, (s, e)) for s, e, c in con.execute("SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    ex = [r for r in rows if r[3] in htod and r[2] >= expert_min and r[1] <= first]
    slow = 0
    for r in ex:
        a = api.get(r[4])
        if a is not None and r[2] / max(1, a[1] - a[0]) < slow_gbps:
            slow += 1
    span = (first - ex[0][0]) / 1e9 if ex else 0.0
    return dict(expert_copies=len(ex), slow=slow, mib=sum(r[2] for r in ex) / 2**20, span_s=round(span, 2))


def window(tokens, lo, hi):
    s = [t for t in tokens if lo <= t["token"] <= hi]
    n = len(s)
    if not n:
        return None
    f = lambda k: sum(t[k] for t in s) / n / 1e6
    g = lambda k: sum(t["idle_while_host_in_ns"][k] for t in s) / n / 1e6
    pin = sum(t["h2d_by_src_kind"].get("pinned", {}).get("bytes", 0) for t in s)
    tot = sum(t["h2d_bytes"] for t in s)
    return dict(n=n, ms=f("t_ns"), compute=f("compute_ns"), dma=f("h2d_ns"), d2h=f("d2h_ns"), idle=f("idle_ns"),
                idle_slow=g("expert_copy_slow"), idle_fast=g("expert_copy_fast"), idle_sync=g("cudaStreamSynchronize"),
                slow_per_token=sum(t["n_expert_copy_slow"] for t in s) / n,
                mib=sum(t["h2d_expert_bytes"] for t in s) / n / 2**20, pinned_share=pin / tot if tot else 0.0,
                overlap_ns=sum(t["overlap_compute_h2d_ns"] for t in s))


def log_fields(path):
    out = {}
    if not os.path.exists(path):
        return out
    txt = open(path, encoding="utf-8", errors="replace").read()
    # log lines start with minutes.seconds.ms.us since the process started
    def ts(needle):
        m = re.search(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) .*" + re.escape(needle), txt, re.M)
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1e3 + int(m.group(4)) / 1e6 if m else None
    t_cache, t_reg, t_ready = ts("MoE cache size"), ts("left pageable"), ts("sampler seed")
    if t_ready is not None:
        out["ready_s"] = round(t_ready, 2)
    if t_cache is not None and t_reg is not None:
        out["register_s"] = round(t_reg - t_cache, 3)
    pats = {
        "prompt_eval_ms": r"prompt eval time =\s+([\d.]+) ms",
        "eval_ms": r"  eval time =\s+([\d.]+) ms",
        "registered_mib": r"registered ([\d.]+) MiB of host experts",
        "pageable_mib": r"\(([\d.]+) MiB left pageable\)",
        "ranges": r"left pageable\) in (\d+) ranges",
        "hit_line": r"(ubatch <= 8:.*)",
        "profile": r"(pinning the experts most selected.*)",
    }
    for k, p in pats.items():
        m = re.search(p, txt)
        if m:
            v = m.group(1)
            try:
                out[k] = float(v) if k not in ("hit_line", "profile") else v.strip()
            except ValueError:
                out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--cap", default="C:/nsys/exp0005")
    ap.add_argument("--out", default=os.path.join(HERE, "runs.json"))
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()

    runs = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for tag in a.tags:
        r = dict(tag=tag)
        tl = os.path.join(HERE, f"timeline_{tag}.json")
        if os.path.exists(tl):
            toks = json.load(open(tl))["tokens"]
            r["windows"] = {f"{lo}-{hi}": window(toks, lo, hi) for lo, hi in WINDOWS}
        db = os.path.join(a.cap, f"{tag}.sqlite")
        if os.path.exists(db):
            r["prefill"] = prefill(db)
        r["log"] = log_fields(os.path.join(HERE, f"log_{tag}.txt"))
        for name in ("pre", "low"):
            p = os.path.join(HERE, f"{name}_{tag}.txt")
            if os.path.exists(p):
                r[name] = open(p).read().strip()
        runs[tag] = r
    with open(a.out, "w", newline="\n") as f:
        json.dump(runs, f, indent=1, sort_keys=True)
        f.write("\n")

    if a.table or not a.tags:
        print(f"{'run':24s} {'ready s':>7s} {'pf s':>6s} {'pf slow':>8s} | "
              + " | ".join(f"{w:>9s} ms  dma  slow" for w in ("3-130", "387-1022")) + " | pinned%")
        for tag, r in sorted(runs.items()):
            w = r.get("windows", {})
            cells = []
            for k in ("3-130", "387-1022"):
                x = w.get(k)
                cells.append(f"{x['ms']:12.1f} {x['dma']:4.1f} {x['idle_slow']:5.1f}" if x else f"{'-':>23s}")
            p = r.get("prefill", {})
            lg = r.get("log", {})
            pw = w.get("387-1022") or {}
            print(f"{tag:24s} {lg.get('ready_s', 0):7.1f} {lg.get('prompt_eval_ms', 0) / 1000:6.1f} "
                  f"{p.get('slow', 0):5d}/{p.get('expert_copies', 0):<5d}| " + " | ".join(cells)
                  + f" | {100 * pw.get('pinned_share', 0):5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
