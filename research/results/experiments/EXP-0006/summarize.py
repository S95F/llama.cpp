#!/usr/bin/env python3
"""EXP-0006: one row per traced run, as EXP-0005's summarize.py, plus the
online registration (when, how long, what the token that did it cost), the
balloon, and the process memory counters of monitor.py.

    python summarize.py <tag>...             (reads timeline_<tag>.json, log_<tag>.txt,
                                              proc_<tag>.csv, <cap>/<tag>.sqlite)
    python summarize.py --table              (prints the comparison table)

Token k of the timeline is the interval that ends with the k-th logits
read-back after the prompt's, so it is decode pass k. LLAMA_MOE_PIN_ONLINE=N
counts passes 1 to N and registers at the start of pass N + 1.

The run's stdout (prompt and generated text) goes to <cap>/out_<tag>.txt and
its hash to runs.json, so the arms of one prompt can be checked byte for byte.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "EXP-0005"))
from summarize import log_fields, prefill, window  # noqa: E402

WINDOWS = [(3, 130), (131, 386), (387, 1022), (3, 1022)]


def online_fields(path):
    out = {}
    if not os.path.exists(path):
        return out
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"pinned the experts most selected in (\d+) small batches in ([\d.]+) ms", txt)
    if m:
        out["online_n"] = int(m.group(1))
        out["online_ms"] = float(m.group(2))
    m = re.search(r"pinning nothing until (\d+) small batches", txt)
    if m:
        out["online_requested"] = int(m.group(1))
    return out


def refused_mib(path):
    """MiB the driver refused to register, summed over the registration lines."""
    if not os.path.exists(path):
        return None
    v = re.findall(r"in \d+ ranges, ([\d.]+) MiB refused", open(path, encoding="utf-8", errors="replace").read())
    return sum(float(x) for x in v) if v else None


def stdout_of(db, out):
    c = sqlite3.connect(db)
    n = dict(c.execute("select id, value from StringIds"))
    s = "".join(n[cid] for fid, cid in c.execute("select filenameId, contentId from ProcessStreams") if "stdout" in n.get(fid, ""))
    c.close()
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(s)
    return dict(chars=len(s), think_end=s.find("</think>"), sha256=hashlib.sha256(s.encode()).hexdigest()[:16])


def proc_fields(path, t0, t1):
    """Process memory over the whole run and its last 60% (the warm part)."""
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {}
    f = lambda r, k: float(r[k])
    late = [r for r in rows if f(r, "t_unix") >= t0 + 0.4 * (t1 - t0)] or rows
    return dict(n=len(rows),
                ws_max_mib=max(f(r, "ws_mib") for r in rows),
                ws_late_median_mib=statistics.median(f(r, "ws_mib") for r in late),
                private_max_mib=max(f(r, "private_mib") for r in rows),
                avail_min_mib=min(f(r, "avail_mib") for r in rows),
                avail_late_median_mib=statistics.median(f(r, "avail_mib") for r in late),
                page_faults=int(f(rows[-1], "page_faults") - f(rows[0], "page_faults")),
                read_mib=f(rows[-1], "read_mib"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--cap", default="C:/nsys/exp0006")
    ap.add_argument("--out", default=os.path.join(HERE, "runs.json"))
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()

    runs = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for tag in a.tags:
        r = dict(tag=tag)
        lg = os.path.join(HERE, f"log_{tag}.txt")
        r["log"] = log_fields(lg)
        r["online"] = online_fields(lg)
        r["log"]["refused_mib"] = refused_mib(lg)
        if r["online"]:
            # the "left pageable" line of an online arm comes at the registration, not at load
            r["log"].pop("register_s", None)
        tl = os.path.join(HERE, f"timeline_{tag}.json")
        if os.path.exists(tl):
            toks = json.load(open(tl))["tokens"]
            wins = list(WINDOWS)
            n = r["online"].get("online_n")
            if n:
                wins += [(3, n), (n + 2, 386)]
                reg = [t for t in toks if t["token"] == n + 1]
                if reg:
                    r["online"]["reg_token_ms"] = reg[0]["t_ns"] / 1e6
                    near = [t["t_ns"] for t in toks if n - 9 <= t["token"] <= n + 11 and t["token"] != n + 1]
                    r["online"]["neighbours_median_ms"] = statistics.median(near) / 1e6
            r["windows"] = {f"{lo}-{hi}": window(toks, lo, hi) for lo, hi in wins}
        db = os.path.join(a.cap, f"{tag}.sqlite")
        if os.path.exists(db):
            r["prefill"] = prefill(db)
            r["stdout"] = stdout_of(db, os.path.join(a.cap, f"out_{tag}.txt"))
        wall = os.path.join(HERE, f"wall_{tag}.txt")
        if os.path.exists(wall):
            w = dict(l.split() for l in open(wall) if l.strip())
            r["proc"] = proc_fields(os.path.join(HERE, f"proc_{tag}.csv"), float(w["start_unix"]), float(w.get("end_unix", 0)) or 1e18)
        for name in ("pre", "low", "balloon", "flag"):
            p = os.path.join(HERE, f"{name}_{tag}.txt")
            if os.path.exists(p):
                r[name] = open(p).read().strip()
        runs[tag] = r
    with open(a.out, "w", newline="\n") as f:
        json.dump(runs, f, indent=1, sort_keys=True)
        f.write("\n")

    if a.table or not a.tags:
        print(f"{'run':16s} {'pf s':>5s} | {'3-130':>6s} {'387-1022':>8s} {'3-1022':>7s} | pinned% | "
              f"{'reg ms':>7s} {'tok ms':>7s} | {'avail min':>9s} {'ws late':>8s}")
        for tag, r in sorted(runs.items()):
            w = r.get("windows", {})
            ms = lambda k: f"{w[k]['ms']:.2f}" if w.get(k) else "-"
            on = r.get("online", {})
            pr = r.get("proc", {})
            print(f"{tag:16s} {r.get('log', {}).get('prompt_eval_ms', 0) / 1000:5.1f} | {ms('3-130'):>6s} {ms('387-1022'):>8s} "
                  f"{ms('3-1022'):>7s} | {100 * (w.get('387-1022') or {}).get('pinned_share', 0):6.1f} | "
                  f"{on.get('online_ms', 0):7.1f} {on.get('reg_token_ms', 0):7.1f} | "
                  f"{pr.get('avail_min_mib', 0):9.0f} {pr.get('ws_late_median_mib', 0):8.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
