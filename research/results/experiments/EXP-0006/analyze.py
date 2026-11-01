#!/usr/bin/env python3
"""EXP-0006: per-arm results from runs.json and the paging CSVs, in one file
(analysis.json) and one table (analysis.md).

    python analyze.py

Arms ran in ABBA order (r1 forward, r2 reversed) within each prompt block.
Pairs are reported per round as well as for the mean of the reps. Tokens
3-130 are early, not cold (EXP-0005), and for the online arms they also hold
the pageable passes before the registration. The offline predictions come from
coverage.py on the 1024-token traces: coverage_1024.json as written before the
timed runs (one pass early), coverage_1024_aligned.json after the fix.
"""

import csv
import datetime
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "EXP-0005"))
import analyze as e5  # noqa: E402  (EXP-0005's, for its paging integral)

e5.HERE = HERE  # read this experiment's paging CSVs

ARMS = ["addr", "prof", "on64", "on128", "prof15k", "on64x15k", "on128x15k", "ood-addr", "ood-prof", "ood-on128"]
WINDOWS = ["3-130", "131-386", "387-1022", "3-1022"]
KEYS = ["ms", "compute", "dma", "idle", "idle_slow", "idle_fast", "idle_sync", "slow_per_token", "mib", "pinned_share"]
# offline prediction (coverage_1024.json): (cap, trace, order)
PREDICTED = {"addr": ("7964", "a", "address"), "prof": ("7964", "a", "select"),
             "on64": ("7964", "a", "online64"), "on128": ("7964", "a", "online128"),
             "prof15k": ("15000", "a", "select"), "on64x15k": ("15000", "a", "online64"),
             "on128x15k": ("15000", "a", "online128"), "ood-addr": ("7964", "ood", "address"),
             "ood-prof": ("7964", "ood", "select"), "ood-on128": ("7964", "ood", "online128")}
# the default run order of runs.sh, for the rep gaps of H4
ORDER = ("on64x15k_w prof_r1 on64_r1 addr_r1 on128_r1 prof15k_r1 on64x15k_r1 on128x15k_r1 on128x15k_r2 on64x15k_r2 "
         "prof15k_r2 on128_r2 addr_r2 on64_r2 prof_r2 ood-addr_r1 ood-prof_r1 ood-on128_r1 ood-on128_r2 ood-prof_r2 ood-addr_r2").split()
PAIRS = [
    # H1: online against the file profile, same cap
    ("on64", "prof"), ("on128", "prof"), ("on64x15k", "prof15k"), ("on128x15k", "prof15k"),
    # H2: against address order, default cap
    ("on64", "addr"), ("on128", "addr"), ("prof", "addr"),
    # N
    ("on128", "on64"), ("on128x15k", "on64x15k"),
    # H3: out of domain
    ("ood-on128", "ood-prof"), ("ood-prof", "ood-addr"), ("ood-on128", "ood-addr"),
]


def paging(tag):
    out = e5.paging(tag)
    rows = list(csv.reader(open(os.path.join(HERE, f"paging_{tag}.csv"))))
    hdr, data = rows[0], [r for r in rows[1:] if len(rows[0]) == len(r)]

    def col(name):
        i = next((j for j, h in enumerate(hdr) if name in h), None)
        if i is None:
            return []
        v = []
        for r in data:
            try:
                v.append(float(r[i]) / 2**20)
            except ValueError:
                pass
        return v
    for k, name in (("standby_normal", "Standby Cache Normal"), ("standby_reserve", "Standby Cache Reserve"),
                    ("modified", "Modified Page List"), ("free_zero", "Free & Zero"), ("total_ws", "Process(_Total)")):
        v = col(name)
        if v:
            out[f"{k}_min_mib"] = min(v)
            out[f"{k}_median_last60_mib"] = statistics.median(v[int(len(v) * 0.4):])
    return out


def balloon_fields(s):
    # "balloon: available A MiB, target T, held H MiB, locked yes, available now N MiB"
    out = {}
    try:
        parts = s.split("balloon: ")[1]
        out["avail_before_mib"] = float(parts.split("available ")[1].split(" MiB")[0])
        out["held_mib"] = float(parts.split("held ")[1].split(" MiB")[0])
        out["avail_launch_mib"] = float(parts.split("available now ")[1].split(" MiB")[0])
        out["locked"] = "locked yes" in parts
    except (IndexError, ValueError):
        pass
    return out


def main():
    runs = json.load(open(os.path.join(HERE, "runs.json")))
    covs = {k: json.load(open(os.path.join(HERE, f))) for k, f in (("pre_run", "coverage_1024.json"), ("aligned", "coverage_1024_aligned.json"))
            if os.path.exists(os.path.join(HERE, f))}
    out = dict(arms={}, pairs={}, notes=[])
    for a in ARMS:
        reps = {}
        for r in (1, 2):
            tag = f"{a}_r{r}"
            if tag not in runs or "windows" not in runs[tag]:
                out["notes"].append(f"{tag}: missing")
                continue
            x = runs[tag]
            reps[tag] = dict(windows=x["windows"], prefill=x.get("prefill"), log=x["log"], online=x.get("online", {}),
                             proc=x.get("proc", {}), pre=x.get("pre"), low=x.get("low"), stdout=x.get("stdout", {}),
                             balloon=balloon_fields(x.get("balloon", "")), paging=paging(tag),
                             partial=x["windows"]["3-1022"]["n"] < 1020)
            pg, pr, b = reps[tag]["paging"], reps[tag]["proc"], reps[tag]["balloon"]
            if all(k in d for d, k in ((pg, "total_ws_median_last60_mib"), (pr, "ws_late_median_mib"), (b, "held_mib"))):
                # working set of every other process in the warm part (shared pages counted once per process)
                pg["other_ws_mib"] = pg["total_ws_median_last60_mib"] - pr["ws_late_median_mib"] - b["held_mib"]
            if x.get("flag"):
                out["notes"].append(f"{tag}: FLAG {x['flag']}")
            if (x["log"].get("refused_mib") or 0) > 0:
                out["notes"].append(f"{tag}: the driver refused {x['log']['refused_mib']:.2f} MiB")
            if reps[tag]["partial"]:
                out["notes"].append(f"{tag}: partial run ({x['windows']['3-1022']['n']} tokens in 3-1022)")
            if (pr.get("avail_late_median_mib") or 1e9) < 1024:
                out["notes"].append(f"{tag}: warm median available {pr['avail_late_median_mib']:.0f} MiB, below 1 GiB")
        if not reps:
            continue
        mean = {}
        wins = sorted({w for t in reps for w in reps[t]["windows"]}, key=lambda w: int(w.split("-")[0]))
        for w in wins:
            have = [t for t in reps if reps[t]["windows"].get(w)]
            if not have:
                continue
            mean[w] = {k: statistics.mean(reps[t]["windows"][w][k] for t in have) for k in KEYS}
            ms = [reps[t]["windows"][w]["ms"] for t in have]
            mean[w]["n_reps"] = len(have)
            mean[w]["rep_spread_pct"] = 100 * (max(ms) - min(ms)) / statistics.mean(ms)
            if len(ms) == 2:
                mean[w]["r1_minus_r2_ms"] = ms[0] - ms[1]
        arm = dict(reps=reps, mean=mean)
        on = [reps[t]["online"] for t in reps if reps[t]["online"].get("online_ms") is not None]
        if on:
            arm["online"] = {k: statistics.mean(o[k] for o in on) for k in ("online_ms", "reg_token_ms", "neighbours_median_ms") if all(k in o for o in on)}
        if "387-1022" in mean:
            arm["measured_pageable_share_387_1022"] = 1 - mean["387-1022"]["pinned_share"]
        if a in PREDICTED and covs:
            cap, run, kind = PREDICTED[a]
            arm["predicted_pageable_share"] = {k: {w: c["caps"][cap]["runs"][run][w][kind] for w in ("387-1022", "3-1022")
                                                   if w in c["caps"][cap]["runs"][run]} for k, c in covs.items()}
        out["arms"][a] = arm

    for p, q in PAIRS:
        if p not in out["arms"] or q not in out["arms"]:
            continue
        d = {}
        for w in WINDOWS:
            per_round = []
            for r in (1, 2):
                rp = out["arms"][p]["reps"].get(f"{p}_r{r}")
                rq = out["arms"][q]["reps"].get(f"{q}_r{r}")
                if rp and rq and rp["windows"].get(w) and rq["windows"].get(w):
                    per_round.append(rp["windows"][w]["ms"] - rq["windows"][w]["ms"])
            mp, mq = out["arms"][p]["mean"][w]["ms"], out["arms"][q]["mean"][w]["ms"]
            d[w] = dict(mean_ms=mp - mq, pct=100 * (mp / mq - 1), per_round_ms=per_round)
        out["pairs"][f"{p} - {q}"] = d

    # H3: does the file profile transfer worse out of domain than in it?
    arms = out["arms"]
    if all(k in arms for k in ("prof", "on128", "ood-prof", "ood-on128", "addr", "ood-addr")):
        w = "387-1022"

        def ms(a_, r):
            return (arms[a_]["reps"].get(f"{a_}_r{r}", {}).get("windows", {}).get(w) or {}).get("ms")

        def m(a_):
            return arms[a_]["mean"][w]["ms"]
        h3 = dict(share_gap_points=100 * (arms["ood-prof"]["measured_pageable_share_387_1022"]
                                          - arms["prof"]["measured_pageable_share_387_1022"]))
        dd = []
        for r in (1, 2):
            v = [ms(x, r) for x in ("ood-prof", "ood-on128", "prof", "on128")]
            dd.append(None if None in v else (v[0] - v[1]) - (v[2] - v[3]))
        h3["online_lead_ood_minus_in_domain_ms"] = dict(per_round=dd, mean=(m("ood-prof") - m("ood-on128")) - (m("prof") - m("on128")))
        h3["profile_gain_rel"] = dict(ood=(m("ood-addr") - m("ood-prof")) / m("ood-addr"), in_domain=(m("addr") - m("prof")) / m("addr"))
        out["h3"] = h3

    # H4: r1 - r2 against the gap between the two reps in the run order
    pts = []
    for a in ARMS:
        r1, r2 = f"{a}_r1", f"{a}_r2"
        x = arms.get(a, {}).get("reps", {})
        if r1 in x and r2 in x and r1 in ORDER and r2 in ORDER:
            pts.append((ORDER.index(r2) - ORDER.index(r1), x[r1]["windows"]["387-1022"]["ms"] - x[r2]["windows"]["387-1022"]["ms"], a))
    if len(pts) >= 2:
        mx, my = statistics.mean(p[0] for p in pts), statistics.mean(p[1] for p in pts)
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        out["h4"] = dict(points=pts, slope_ms_per_position=sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx if sxx else None,
                         r1_slower=sum(p[1] > 0 for p in pts), n=len(pts),
                         max_spread_pct=max(arms[a]["mean"]["387-1022"]["rep_spread_pct"] for _g, _d, a in pts))

    # every run of one prompt must print the same text
    for grp, ood in (("prompt_a", False), ("ood", True)):
        h = {t: x["stdout"].get("sha256") for a_ in arms if a_.startswith("ood-") == ood for t, x in arms[a_]["reps"].items()}
        out[f"stdout_{grp}"] = sorted(set(h.values()), key=str)
        if len(set(h.values())) > 1:
            out["notes"].append(f"{grp}: the runs print different text: {h}")

    with open(os.path.join(HERE, "analysis.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")

    A = [a for a in ARMS if a in out["arms"]]
    lines = ["# EXP-0006 analysis (generated by analyze.py)", ""]
    for w in ("387-1022", "3-1022", "3-130"):
        lines += [f"## Tokens {w}, mean of r1 and r2", "",
                  "| arm | ms/token | r1 | r2 | spread | compute | DMA | idle | slow-copy idle | fast-copy idle | sync idle | pinned share |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for a in A:
            m = out["arms"][a]["mean"].get(w)
            if not m:
                continue
            r = [out["arms"][a]["reps"].get(f"{a}_r{i}", {}).get("windows", {}).get(w) for i in (1, 2)]
            rs = [f"{x['ms']:.2f}" if x else "-" for x in r]
            lines.append(f"| {a} | {m['ms']:.2f} | {rs[0]} | {rs[1]} | {m['rep_spread_pct']:.1f}% | "
                         f"{m['compute']:.2f} | {m['dma']:.2f} | {m['idle']:.2f} | {m['idle_slow']:.2f} | {m['idle_fast']:.2f} | "
                         f"{m['idle_sync']:.2f} | {100 * m['pinned_share']:.1f}% |")
        lines.append("")
    lines += ["## Pairs (negative = first arm faster)", "", "| pair | window | mean ms | % | r1 | r2 |", "|---|---|---:|---:|---:|---:|"]
    for k, d in out["pairs"].items():
        for w in ("387-1022", "3-1022"):
            x = d[w]
            pr = [f"{v:+.2f}" for v in x["per_round_ms"]] + ["-"] * (2 - len(x["per_round_ms"]))
            lines.append(f"| {k} | {w} | {x['mean_ms']:+.2f} | {x['pct']:+.1f}% | {pr[0]} | {pr[1]} |")
    if "h3" in out:
        h = out["h3"]
        dd = h["online_lead_ood_minus_in_domain_ms"]
        lines += ["", "## H3 (tokens 387-1022)", "",
                  f"- pageable share, ood-prof minus prof: {h['share_gap_points']:+.1f} points (falsified below +5)",
                  "- (ood-prof - ood-on128) - (prof - on128): " + ", ".join("-" if v is None else f"{v:+.2f}" for v in dd["per_round"])
                  + f" ms per round, {dd['mean']:+.2f} ms on the means (falsified if <= 0 in either round)",
                  f"- relative gain of the file profile over address order: ood {100 * h['profile_gain_rel']['ood']:.1f}%, "
                  f"prompt a {100 * h['profile_gain_rel']['in_domain']:.1f}%"]
    if "h4" in out:
        h = out["h4"]
        lines += ["", "## H4 (tokens 387-1022)", "",
                  f"- largest rep spread {h['max_spread_pct']:.1f}% (falsified above 2%)",
                  f"- slope of r1 - r2 against the rep gap: {h['slope_ms_per_position']:+.3f} ms per position (falsified above +0.05)",
                  f"- r1 slower in {h['r1_slower']} of {h['n']} arms (falsified at 8 or more of 10)",
                  "- points (gap, r1 - r2 ms, arm): " + ", ".join(f"({g}, {d:+.2f}, {a})" for g, d, a in h["points"])]
    lines += ["", "## Online registration", "",
              "register ms is the time of the cudaHostRegister calls; on128x15k's registering token also waits about 1.2 s in the next stream synchronization (captures.md).", "",
              "| run | N | register ms | registering token ms | median of 20 tokens around it | pinned MiB | ranges | before N+1: ms/token | N+2-386: ms/token |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in A:
        for t, x in out["arms"][a]["reps"].items():
            o = x["online"]
            if not o.get("online_n"):
                continue
            n = o["online_n"]
            w1, w2 = x["windows"].get(f"3-{n}"), x["windows"].get(f"{n + 2}-386")
            lines.append(f"| {t} | {n} | {o['online_ms']:.1f} | {o.get('reg_token_ms', 0):.1f} | {o.get('neighbours_median_ms', 0):.1f} | "
                         f"{x['log'].get('registered_mib', 0):.0f} | {x['log'].get('ranges', 0):.0f} | "
                         f"{w1['ms'] if w1 else 0:.2f} | {w2['ms'] if w2 else 0:.2f} |")
    lines += ["", "## Pageable share of the upload bytes, predicted offline vs measured", "",
              "Predicted: coverage.py's share of the expert miss bytes, pre-run (one pass early) and aligned. "
              "Measured: 1 - pinned share of all host-to-device bytes (CUPTI), which differs from the expert-only share by at most 0.02 points.", "",
              "| arm | pre-run 387-1022 | aligned 387-1022 | measured 387-1022 | pre-run 3-1022 | aligned 3-1022 | measured 3-1022 |",
              "|---|---:|---:|---:|---:|---:|---:|"]

    def pct(v):
        return f"{100 * v:.1f}%" if v is not None else "-"
    for a in A:
        x = out["arms"][a]
        pr = x.get("predicted_pageable_share", {})
        p0, p1 = pr.get("pre_run", {}), pr.get("aligned", {})
        whole = x["mean"].get("3-1022")
        lines.append(f"| {a} | {pct(p0.get('387-1022'))} | {pct(p1.get('387-1022'))} | {pct(x.get('measured_pageable_share_387_1022'))} | "
                     f"{pct(p0.get('3-1022'))} | {pct(p1.get('3-1022'))} | {pct(1 - whole['pinned_share'] if whole else None)} |")
    lines += ["", "## Per run: memory, load, prefill", "",
              "| run | avail before balloon | balloon MiB | avail at launch | lowest avail (proc) | median avail, warm | working set, warm | other processes' working set, warm | page faults | pinned MiB | ranges | prefill slow copies | prompt eval s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for a in A:
        for t, x in out["arms"][a]["reps"].items():
            b, pr, lg, pf = x["balloon"], x["proc"], x["log"], x["prefill"] or {}

            def f(v, fmt="0f"):
                return format(v, "." + fmt) if isinstance(v, (int, float)) else "-"
            lines.append(f"| {t} | {f(b.get('avail_before_mib'))} | {f(b.get('held_mib'))} | {f(b.get('avail_launch_mib'))} | "
                         f"{f(pr.get('avail_min_mib'))} | {f(pr.get('avail_late_median_mib'))} | {f(pr.get('ws_late_median_mib'))} | "
                         f"{f(x['paging'].get('other_ws_mib'))} | "
                         f"{f(pr.get('page_faults'))} | {f(lg.get('registered_mib'))} | {f(lg.get('ranges'))} | "
                         f"{pf.get('slow', '-')}/{pf.get('expert_copies', '-')} | {f((lg.get('prompt_eval_ms') or 0) / 1000, '2f')} |")
    lines += ["", "Memory columns come from monitor.py (every second, process alive); warm is the last 60% of the run's wall time."]
    for n in out["notes"]:
        lines.append(f"- {n}")
    with open(os.path.join(HERE, "analysis.md"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
