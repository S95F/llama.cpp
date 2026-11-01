#!/usr/bin/env python3
"""EXP-0007 (Q18): which pageable expert uploads fault, and the prefetch calls.

Joins a run's selection trace (LLAMA_MOE_CACHE_TRACE) and copy log
(LLAMA_MOE_COPY_LOG) on the slot-map copy number. A decode pass is one sweep of
the cached layers by a one-token batch (8 ids), counted from 1 at the first
cached layer, as the timeline counts tokens. An upload is one copy-log row with
ip >= 0: one expert tensor (or n consecutive experts) of one layer. It is
pageable if less than half of it is pinned (an expert slice is not page
aligned, so one next to a pinned expert shares a page with it), and slow if its
pageable bytes went below --slow-gbps (1.5 GB/s, the timeline's threshold for a
page fault) over the host call. Uploads of n > 1 experts (cache warm-up) are
counted but not classified.

The distance of an upload is the number of passes since its expert (layer and
expert) was last read by the host, in an earlier pass: an upload in the decode,
or the prefill's large-batch uploads (copy-log call 0, counted as pass 0). A hit
reads the device copy, not the host's. "first" means not read by the host since
the load (whose prefetch read the whole file, perhaps evicted since): no gap is
known, so it is a class of its own.

    python attrib.py <tag>... --cap C:/nsys/exp0007     (writes attrib.json)
"""

import argparse
import bisect
import csv
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = {"3-1022": (3, 1022), "387-1022": (387, 1022), "3-130": (3, 130)}
BINS = [("1-15", 1, 15), ("16-127", 16, 127), ("128+", 128, 10**9), ("first", None, None)]


def bin_of(d):
    if d is None:
        return "first"
    for name, lo, hi in BINS[:-1]:
        if lo <= d <= hi:
            return name
    raise ValueError(d)


def analyze(tag, cap, slow_gbps):
    trace = os.path.join(cap, f"trace_{tag}.csv")
    copies = os.path.join(cap, f"copies_{tag}.csv")
    if not (os.path.exists(trace) and os.path.exists(copies)):
        return None

    # call -> (pass, layer); selections per (layer, expert) by pass
    # the prompt's last layer runs on one token too (only its output row is kept), so a sweep starts at the
    # lowest cached layer, and the calls before the first sweep are pass 0
    rows = [r for r in csv.DictReader(open(trace)) if int(r["n_ids"]) <= 8]  # larger: a multi-token batch of the prompt
    first_layer = min(int(r["layer"]) for r in rows)
    call_pass = {}
    sel_passes = {}
    n_pass = 0
    for r in rows:
        il = int(r["layer"])
        if il == first_layer:
            n_pass += 1
        call_pass[int(r["call"])] = (n_pass, il)
        for e in map(int, r["ids"].split()):
            sel_passes.setdefault((il, e), []).append(n_pass)

    last_up = {}       # (layer, expert) -> pass of its last upload
    pending = {}       # uploads of the current pass, applied when the pass changes
    cur = None
    ups = []
    pf = []
    n_multi = n_mixed = n_prefill = 0
    for r in csv.DictReader(open(copies)):
        call = int(r["call"])
        if call == 0:
            # a large-batch upload of the prefill: a host read at pass 0, not classified
            for k in range(int(r["n"])):
                last_up[(int(r["layer"]), int(r["expert"]) + k)] = 0
            n_prefill += 1
            continue
        if call not in call_pass:
            continue
        p, il = call_pass[call]
        if p != cur:
            last_up.update(pending)
            pending = {}
            cur = p
        ip, e, n = int(r["ip"]), int(r["expert"]), int(r["n"])
        b, pin, us = int(r["bytes"]), int(r["pinned"]), int(r["us"])
        if ip < 0:
            # expert = candidate ranges, n = ranges prefetched, bytes = prefetched, pinned = candidate bytes
            pf.append(dict(p=p, layer=il, cand=e, kept=n, bytes=b, cand_bytes=pin, us=us))
            continue
        for k in range(n):
            pending[(il, e + k)] = p
        if n > 1:
            n_multi += 1
            continue
        # an expert slice is not page aligned, so a pageable expert next to a pinned one shares a page with it
        if 2 * pin >= b:
            n_mixed += 0 < pin < b
            ups.append(dict(p=p, pageable=False, slow=False, us=us, bytes=b))
            continue
        n_mixed += pin > 0
        prev = last_up.get((il, e))
        s = sel_passes.get((il, e), [])
        i = bisect.bisect_left(s, p)
        ups.append(dict(p=p, pageable=True, slow=(b - pin) / max(us, 1) < slow_gbps * 1000, us=us, bytes=b,
                        dist=None if prev is None else p - prev,
                        sel_dist=None if i == 0 else p - s[i - 1], n_sel_before=i))

    out = dict(passes=n_pass, multi_expert_uploads=n_multi, partly_pinned_uploads=n_mixed, prefill_uploads=n_prefill)
    for wname, (lo, hi) in WINDOWS.items():
        hi = min(hi, n_pass - 1)  # the timeline drops the last token
        n_p = hi - lo + 1
        u = [x for x in ups if lo <= x["p"] <= hi]
        if not u or n_p <= 0:
            continue
        pg = [x for x in u if x["pageable"]]
        slow = [x for x in pg if x["slow"]]
        fast = [x for x in pg if not x["slow"]]
        bins = {}
        for name, _, _ in BINS:
            xb = [x for x in pg if bin_of(x["dist"]) == name]
            bins[name] = dict(n=len(xb), slow=sum(x["slow"] for x in xb),
                              slow_share=sum(x["slow"] for x in xb) / len(xb) if xb else None)
        med = lambda xs, k: statistics.median([x[k] for x in xs if x[k] is not None]) if any(x[k] is not None for x in xs) else None
        pw = [x for x in pf if lo <= x["p"] <= hi]
        out[wname] = dict(
            passes=n_p,
            uploads=len(u), pageable=len(pg), slow=len(slow),
            slow_share=len(slow) / len(pg) if pg else None,
            pageable_per_pass=len(pg) / n_p, slow_per_pass=len(slow) / n_p,
            slow_us_per_pass=sum(x["us"] for x in slow) / n_p,
            pageable_us_per_pass=sum(x["us"] for x in pg) / n_p,
            pinned_us_per_pass=sum(x["us"] for x in u if not x["pageable"]) / n_p,
            first_share_slow=sum(x["dist"] is None for x in slow) / len(slow) if slow else None,
            first_share_fast=sum(x["dist"] is None for x in fast) / len(fast) if fast else None,
            dist_median_slow=med(slow, "dist"), dist_median_fast=med(fast, "dist"),
            sel_dist_median_slow=med(slow, "sel_dist"), sel_dist_median_fast=med(fast, "sel_dist"),
            n_sel_before_median_slow=med(slow, "n_sel_before"), n_sel_before_median_fast=med(fast, "n_sel_before"),
            bins=bins,
            prefetch_calls_per_pass=len(pw) / n_p,
            prefetch_us_per_pass=sum(x["us"] for x in pw) / n_p,
            prefetch_mib_per_pass=sum(x["bytes"] for x in pw) / n_p / 2**20,
            prefetch_cand_mib_per_pass=sum(x["cand_bytes"] for x in pw) / n_p / 2**20,
            prefetch_ranges_per_pass=sum(x["kept"] for x in pw) / n_p,
            prefetch_cand_ranges_per_pass=sum(x["cand"] for x in pw) / n_p,
            prefetch_us_median=statistics.median(x["us"] for x in pw) if pw else None,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--cap", default="C:/nsys/exp0007")
    ap.add_argument("--slow-gbps", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(HERE, "attrib.json"))
    a = ap.parse_args()

    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for tag in a.tags:
        r = analyze(tag, a.cap, a.slow_gbps)
        if r is None:
            print(f"{tag}: no trace or copy log in {a.cap}", file=sys.stderr)
            continue
        res[tag] = r
        w = r.get("387-1022") or r.get("3-130") or {}
        print(f"{tag}: {r['passes']} passes; pageable/pass {w.get('pageable_per_pass', 0):.2f}, "
              f"slow share {100 * (w.get('slow_share') or 0):.2f}%, prefetch {w.get('prefetch_us_per_pass', 0):.0f} us/pass")
    with open(a.out, "w", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
