#!/usr/bin/env python3
"""EXP-0001: average the per-trace policy sweeps and compare against the
reference tables that were measured on base 78a8ae1.

  python research/results/experiments/EXP-0001/aggregate.py

Reference values are transcribed from moe-bench/RESULTS.md sections 4 and 8,
which average the same four prompts on the old base.
"""

import collections
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
TRACES = ["prompt", "prompt_b", "prompt_c", "prompt_d"]
SIZES = [256, 384, 528, 586, 645, 1024, 2048]

# the branch only samples below one token's experts; at and above the crossover
# it is byte-for-byte the upstream policy
SUB_CROSSOVER = {256, 384, 528, 586}

# RESULTS.md section 4: upstream / branch default / frequency forced on
REF_S4 = {
    256:  (0.00, 15.77, 15.77),
    384:  (0.00, 23.90, 23.90),
    528:  (1.17, 29.98, 29.98),
    586:  (24.27, 31.97, 31.97),
    645:  (35.50, 35.50, 33.84),
    1024: (44.34, 44.34, 42.47),
    2048: (61.65, 61.65, 58.71),
}
# RESULTS.md section 8: Belady
REF_BELADY = {256: 31.86, 384: 43.35, 528: 51.53, 586: 53.93,
              1024: 64.74, 2048: 77.30}


def load():
    acc = collections.defaultdict(list)
    for t in TRACES:
        for r in json.load(open(os.path.join(OUT, "sim_sweep_%s.json" % t))):
            acc[(r["cache_mib"], r["policy"])].append(r)
    return acc


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def spread(rows, key):
    v = [r[key] for r in rows]
    return max(v) - min(v)


def main():
    acc = load()
    pols = ["slru-ramp", "ramp-s16-age8", "lfu", "s3fifo", "ramp-rand", "belady"]

    out = {"base": "d854d98b", "traces": TRACES, "rows": []}

    print("## Policy sweep, mean of four traces, base d854d98b")
    print()
    print("| MiB | " + " | ".join(pols) + " | branch default |")
    print("|---:|" + "---:|" * (len(pols) + 1))
    for mib in SIZES:
        cells = []
        for p in pols:
            rows = acc[(mib, p)]
            cells.append("%.2f" % mean(rows, "hit_rate"))
        branch = mean(acc[(mib, "ramp-s16-age8" if mib in SUB_CROSSOVER
                           else "slru-ramp")], "hit_rate")
        print("| %d | %s | **%.2f** |" % (mib, " | ".join(cells), branch))
        row = {"cache_mib": mib, "branch_default": branch}
        for p in pols:
            rows = acc[(mib, p)]
            row[p] = {"hit_rate": mean(rows, "hit_rate"),
                      "uploaded_mib": mean(rows, "uploaded_mib"),
                      "hit_rate_spread": spread(rows, "hit_rate")}
        out["rows"].append(row)

    print()
    print("## Against the 78a8ae1 reference (RESULTS.md section 4)")
    print()
    print("| MiB | upstream ref | upstream now | d | branch ref | branch now | d | forced ref | forced now | d |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mib in SIZES:
        up = mean(acc[(mib, "slru-ramp")], "hit_rate")
        forced = mean(acc[(mib, "ramp-s16-age8")], "hit_rate")
        branch = forced if mib in SUB_CROSSOVER else up
        r_up, r_br, r_fo = REF_S4[mib]
        print("| %d | %.2f | %.2f | %+.2f | %.2f | %.2f | %+.2f | %.2f | %.2f | %+.2f |"
              % (mib, r_up, up, up - r_up, r_br, branch, branch - r_br,
                 r_fo, forced, forced - r_fo))

    print()
    print("## Belady gap (RESULTS.md section 8)")
    print()
    print("| MiB | upstream | branch | Belady now | Belady ref | gap closed |")
    print("|---:|---:|---:|---:|---:|---:|")
    for mib in SIZES:
        up = mean(acc[(mib, "slru-ramp")], "hit_rate")
        forced = mean(acc[(mib, "ramp-s16-age8")], "hit_rate")
        branch = forced if mib in SUB_CROSSOVER else up
        bel = mean(acc[(mib, "belady")], "hit_rate")
        ref = REF_BELADY.get(mib)
        closed = (branch - up) / (bel - up) * 100 if bel > up else float("nan")
        print("| %d | %.2f | %.2f | %.2f | %s | %.0f%% |"
              % (mib, up, branch, bel,
                 "%.2f" % ref if ref else "-", closed))

    print()
    print("## Per-trace spread of the branch arm (max - min over four traces)")
    print()
    print("| MiB | hit% spread |")
    print("|---:|---:|")
    for mib in SIZES:
        p = "ramp-s16-age8" if mib in SUB_CROSSOVER else "slru-ramp"
        print("| %d | %.2f |" % (mib, spread(acc[(mib, p)], "hit_rate")))

    json.dump(out, open(os.path.join(OUT, "sweep_mean.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
