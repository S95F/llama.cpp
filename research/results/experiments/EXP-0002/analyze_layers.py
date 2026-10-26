#!/usr/bin/env python3
"""EXP-0002 / OPEN_QUESTIONS.md Q2: is the remaining Belady gap concentrated in
a few layers or spread evenly?

Per layer, over the four EXP-0001 traces:
  - routing skew (share of selections taken by the hottest k experts)
  - distinct experts touched
  - hit rate under the shipping policy and under Belady, at one cache size
  - the gap between them

Decides between ROADMAP R2 candidate 1 (per-layer slot budgets) and candidate 2
(a better victim score). Concentrated loss favours budgets, spread loss favours
the score.

  python research/results/experiments/EXP-0002/analyze_layers.py [cache_mib]
"""

import collections
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "..", "moe-bench"))
import moe_sim  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "..")
TRACES_DIR = os.path.join(ROOT, "research", "results", "experiments", "EXP-0001")
LAYOUT = os.path.join(ROOT, "moe-bench",
                      "layout-qwen35moe-35b-a3b-ud-q4kxl.json")
OUT = os.path.dirname(os.path.abspath(__file__))
TRACES = ["prompt", "prompt_b", "prompt_c", "prompt_d"]

SHIPPING = "ramp-s16-age8"   # what the branch runs below the crossover
UPSTREAM = "slru-ramp"


def skew(trace, n_expert=256):
    """Per layer: hottest-k shares and distinct experts touched."""
    by_layer = collections.defaultdict(collections.Counter)
    for il, ids, _ in trace:
        by_layer[il].update(set(ids))
    out = {}
    for il, cnt in by_layer.items():
        ranked = sorted(cnt.values(), reverse=True)
        tot = sum(ranked)
        out[il] = {
            "distinct": len(cnt),
            "top8": 100.0 * sum(ranked[:8]) / tot,
            "top16": 100.0 * sum(ranked[:16]) / tot,
            "top32": 100.0 * sum(ranked[:32]) / tot,
            "top64": 100.0 * sum(ranked[:64]) / tot,
        }
    return out


def rate(d):
    t = d["hits"] + d["fills"]
    return 100.0 * d["hits"] / t if t else 0.0


def main():
    mib = int(sys.argv[1]) if len(sys.argv) > 1 else 528
    layout = json.load(open(LAYOUT))

    # which group each layer belongs to, for reference
    groups = moe_sim.build_groups(layout)
    group_of = {}
    for gi, g in enumerate(groups):
        for il in g.layers:
            group_of[il] = gi

    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in TRACES:
        trace = moe_sim.load_trace(os.path.join(TRACES_DIR, "trace_%s.csv" % t))
        sk = skew(trace, layout["n_expert"])
        for pol in (UPSTREAM, SHIPPING, "belady"):
            r = moe_sim.simulate(trace, layout, mib, pol, 8, small_only=True)
            for il, d in r["per_layer"].items():
                acc[il][pol].append(rate(d))
        for il, d in sk.items():
            for k, v in d.items():
                acc[il][k].append(v)

    rows = []
    for il in sorted(acc):
        a = acc[il]
        if not a[SHIPPING]:
            continue
        ship = statistics.mean(a[SHIPPING])
        bel = statistics.mean(a["belady"])
        rows.append({
            "layer": il,
            "group": group_of.get(il, -1),
            "upstream": statistics.mean(a[UPSTREAM]),
            "shipping": ship,
            "belady": bel,
            "gap": bel - ship,
            "top8": statistics.mean(a["top8"]),
            "top32": statistics.mean(a["top32"]),
            "top64": statistics.mean(a["top64"]),
            "distinct": statistics.mean(a["distinct"]),
        })

    print("# Per-layer breakdown at %d MiB, mean of four traces" % mib)
    print()
    print("| layer | grp | upstream | shipping | Belady | gap | top8%% | top32%% | top64%% | distinct |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print("| %d | %d | %.2f | %.2f | %.2f | **%.2f** | %.1f | %.1f | %.1f | %.0f |"
              % (r["layer"], r["group"], r["upstream"], r["shipping"],
                 r["belady"], r["gap"], r["top8"], r["top32"], r["top64"],
                 r["distinct"]))

    gaps = [r["gap"] for r in rows]
    ships = [r["shipping"] for r in rows]
    t64 = [r["top64"] for r in rows]
    print()
    print("## Distribution")
    print()
    print("| quantity | min | median | max | stdev |")
    print("|---|---:|---:|---:|---:|")
    for lbl, v in (("gap to Belady", gaps), ("shipping hit rate", ships),
                   ("top64 share", t64)):
        print("| %s | %.2f | %.2f | %.2f | %.2f |"
              % (lbl, min(v), statistics.median(v), max(v), statistics.pstdev(v)))

    srt = sorted(rows, key=lambda r: -r["gap"])
    share = sum(r["gap"] for r in srt[:8]) / sum(gaps) * 100
    print()
    print("Worst 8 layers of %d hold **%.1f%%** of the total gap "
          "(uniform would be %.1f%%)." % (len(rows), share, 8.0 / len(rows) * 100))
    print()
    print("Worst 5 by gap: " + ", ".join(
        "L%d (%.2f, top64 %.1f%%)" % (r["layer"], r["gap"], r["top64"])
        for r in srt[:5]))
    print("Best 5 by gap : " + ", ".join(
        "L%d (%.2f, top64 %.1f%%)" % (r["layer"], r["gap"], r["top64"])
        for r in srt[-5:]))

    # does skew predict the gap?
    def corr(xs, ys):
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        return num / den if den else float("nan")

    print()
    print("## Does routing skew predict the per-layer gap or hit rate?")
    print()
    print("| pair | Pearson r |")
    print("|---|---:|")
    for lbl, xs, ys in (
            ("top64 share vs gap to Belady", t64, gaps),
            ("top64 share vs shipping hit rate", t64, ships),
            ("top8 share vs shipping hit rate", [r["top8"] for r in rows], ships),
            ("distinct experts vs shipping hit rate", [r["distinct"] for r in rows], ships),
    ):
        print("| %s | %+.3f |" % (lbl, corr(xs, ys)))

    json.dump({"cache_mib": mib, "rows": rows},
              open(os.path.join(OUT, "per_layer_%d.json" % mib), "w"), indent=1)


if __name__ == "__main__":
    main()
