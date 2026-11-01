#!/usr/bin/env python3
"""EXP-0005: the per-arm results from runs.json, the paging CSVs and the offline
coverage, in one file (analysis.json) and one table (analysis.md).

    python analyze.py

Arms ran in ABBA order (r1 forward, r2 reversed), so the mean of the two reps
cancels a linear drift across the session, not the r1/r2 memory-state effect
(review.json, critic). addr15000_r1 was stopped by the watchdog after 839
tokens; its windows are partial and flagged, and 387-841 is the window matched
to it. Tokens 3-130 are early, not cold: the load has already read most of the
file (review.json).
"""

import csv
import datetime
import json
import os
import statistics

from summarize import window

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["off", "addr7842", "prof7842", "addr15000", "prof15000", "none"]
WINDOWS = ["3-130", "131-386", "387-1022", "387-841", "3-1022"]
KEYS = ["ms", "compute", "dma", "idle", "idle_slow", "idle_fast", "idle_sync", "slow_per_token", "mib", "pinned_share"]
# offline prediction for prompt a (coverage_2048_runcaps.json): share of warm upload bytes left pageable
PREDICTED_PAGEABLE = {"addr7842": ("7964", "address"), "prof7842": ("7964", "select"),
                      "addr15000": ("15000", "address"), "prof15000": ("15000", "select")}


def paging(tag):
    """typeperf stalls for up to about 5 s while the model loads, so the rates are
    integrated over the real interval to each row (a cooked rate is the average
    over that interval), not summed as if every row were 1 s."""
    rows = list(csv.reader(open(os.path.join(HERE, f"paging_{tag}.csv"))))
    hdr, data = rows[0], [r for r in rows[1:] if len(rows[0]) == len(r)]
    ts = [datetime.datetime.strptime(r[0], "%m/%d/%Y %H:%M:%S.%f").timestamp() for r in data]

    def col(name):
        i = next(j for j, h in enumerate(hdr) if name in h)
        out = []
        for r in data:
            try:
                out.append(float(r[i]))
            except ValueError:
                out.append(None)
        return out

    def integral(v):
        return sum((v[k] or 0.0) * (ts[k] - ts[k - 1]) for k in range(1, len(ts)))
    av = [x for x in col("Available MBytes") if x is not None]
    k = int(len(av) * 0.4)
    gaps = [ts[k] - ts[k - 1] for k in range(1, len(ts))]
    return dict(seconds=ts[-1] - ts[0], max_gap_s=max(gaps), avail_min_mib=min(av),
                avail_median_mib=statistics.median(av), avail_median_last60_mib=statistics.median(av[k:]),
                page_reads=integral(col("Page Reads/sec")), disk_read_gib=integral(col("Disk Read Bytes/sec")) / 2**30)


def main():
    runs = json.load(open(os.path.join(HERE, "runs.json")))
    cov = json.load(open(os.path.join(HERE, "coverage_2048_runcaps.json")))
    out = dict(arms={}, notes=[])
    for a in ARMS:
        reps = {}
        for r in (1, 2):
            tag = f"{a}_r{r}"
            x = runs[tag]
            toks = json.load(open(os.path.join(HERE, f"timeline_{tag}.json")))["tokens"]
            x["windows"]["387-841"] = window(toks, 387, 841)
            reps[tag] = dict(windows=x["windows"], prefill=x.get("prefill"), log=x["log"], pre=x.get("pre"),
                             low=x.get("low"), paging=paging(tag),
                             partial=x["windows"]["3-1022"]["n"] < 1020)
        mean = {}
        for w in WINDOWS:
            mean[w] = {k: statistics.mean(reps[t]["windows"][w][k] for t in reps) for k in KEYS}
            ms = [reps[t]["windows"][w]["ms"] for t in reps]
            mean[w]["rep_spread_pct"] = 100 * (max(ms) - min(ms)) / statistics.mean(ms)
            mean[w]["r1_minus_r2_ms"] = ms[0] - ms[1]
        arm = dict(reps=reps, mean=mean)
        if a in PREDICTED_PAGEABLE:
            cap, kind = PREDICTED_PAGEABLE[a]
            c = cov["caps"][cap][kind]
            covered = c["covered"]["trace_prompt"] if kind == "address" else c["covered_held"]["trace_prompt"]
            arm["predicted_pageable_share"] = 1 - covered
            arm["measured_pageable_share_387_1022"] = 1 - mean["387-1022"]["pinned_share"]
        out["arms"][a] = arm

    base = out["arms"]["off"]["mean"]
    for a in ARMS:
        m = out["arms"][a]["mean"]
        for w in WINDOWS:
            m[w]["vs_off_pct"] = 100 * (m[w]["ms"] / base[w]["ms"] - 1)
    pairs = [("prof7842", "addr7842"), ("prof15000", "addr15000"), ("prof7842", "addr15000"), ("none", "addr15000")]
    out["pairs"] = {}
    for p, q in pairs:
        d = {}
        for w in WINDOWS:
            per_round = [out["arms"][p]["reps"][f"{p}_r{r}"]["windows"][w]["ms"] - out["arms"][q]["reps"][f"{q}_r{r}"]["windows"][w]["ms"]
                         for r in (1, 2)]
            d[w] = dict(mean_ms=out["arms"][p]["mean"][w]["ms"] - out["arms"][q]["mean"][w]["ms"],
                        pct=100 * (out["arms"][p]["mean"][w]["ms"] / out["arms"][q]["mean"][w]["ms"] - 1),
                        per_round_ms=per_round)
        out["pairs"][f"{p} - {q}"] = d
    out["notes"].append("addr15000_r1 was stopped by watchdog.py (available below 300 MiB for 20 s) after 839 decode "
                        "tokens; its 387-1022 window holds 455 tokens and its log has no perf or hit-rate lines.")

    with open(os.path.join(HERE, "analysis.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")

    lines = ["# EXP-0005 analysis (generated by analyze.py)", ""]
    for w in ("387-1022", "387-841", "3-130", "3-1022"):
        lines += [f"## Tokens {w}{' (early, not cold)' if w == '3-130' else ''}, mean of r1 and r2", "",
                  "| arm | ms/token | vs off | r1 | r2 | spread | compute | DMA | idle | slow-copy idle | fast-copy idle | sync idle | pinned share |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for a in ARMS:
            m = out["arms"][a]["mean"][w]
            r = [out["arms"][a]["reps"][f"{a}_r{i}"]["windows"][w]["ms"] for i in (1, 2)]
            flag = "*" if out["arms"][a]["reps"][f"{a}_r1"]["partial"] and w in ("387-1022", "3-1022") else ""
            lines.append(f"| {a} | {m['ms']:.2f} | {m['vs_off_pct']:+.1f}% | {r[0]:.2f}{flag} | {r[1]:.2f} | {m['rep_spread_pct']:.1f}% | "
                         f"{m['compute']:.2f} | {m['dma']:.2f} | {m['idle']:.2f} | {m['idle_slow']:.2f} | {m['idle_fast']:.2f} | "
                         f"{m['idle_sync']:.2f} | {100 * m['pinned_share']:.1f}% |")
        lines.append("")
    lines += ["\\* addr15000_r1 stopped after 839 tokens (455 in 387-1022).", "",
              "## Order pairs (negative = first arm faster)", "",
              "| pair | window | mean ms | % | r1 | r2 |", "|---|---|---:|---:|---:|---:|"]
    for k, d in out["pairs"].items():
        for w in ("387-1022", "387-841", "3-130"):
            x = d[w]
            lines.append(f"| {k} | {w} | {x['mean_ms']:+.2f} | {x['pct']:+.1f}% | {x['per_round_ms'][0]:+.2f} | {x['per_round_ms'][1]:+.2f} |")
    lines += ["", "## Pageable share of warm upload bytes, predicted offline vs measured (tokens 387-1022)", "",
              "| arm | predicted | measured |", "|---|---:|---:|"]
    for a in ARMS:
        if "predicted_pageable_share" in out["arms"][a]:
            lines.append(f"| {a} | {100 * out['arms'][a]['predicted_pageable_share']:.1f}% | "
                         f"{100 * out['arms'][a]['measured_pageable_share_387_1022']:.1f}% |")
    lines += ["", "## Per run: load, prefill, memory", "",
              "| run | ready s | register s | ranges | pinned MiB | prompt eval s | prefill slow copies | eval s | probe | lowest avail MiB | median avail, last 60% of rows MiB | page reads | disk read GiB |",
              "|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|"]
    for a in ARMS:
        for i in (1, 2):
            t = f"{a}_r{i}"
            x = out["arms"][a]["reps"][t]
            lg, pf, pg = x["log"], x["prefill"] or {}, x["paging"]
            probe = (x["pre"] or "").split("probe: ")[-1].split(" expert")[0]

            def f(v, fmt):
                return format(v, fmt) if isinstance(v, (int, float)) else "-"
            lines.append(f"| {t} | {f(lg.get('ready_s'), '.2f')} | {f(lg.get('register_s'), '.3f')} | {f(lg.get('ranges'), '.0f')} | "
                         f"{f(lg.get('registered_mib'), '.0f')} | {f((lg.get('prompt_eval_ms') or 0) / 1000 if lg.get('prompt_eval_ms') else None, '.2f')} | "
                         f"{pf.get('slow', '-')}/{pf.get('expert_copies', '-')} | {f((lg.get('eval_ms') or 0) / 1000 if lg.get('eval_ms') else None, '.1f')} | "
                         f"{probe} | {pg['avail_min_mib']:.0f} | {pg['avail_median_last60_mib']:.0f} | {pg['page_reads']:.0f} | {pg['disk_read_gib']:.1f} |")
    lines += ["", "Lowest avail includes the load, during which typeperf stalls for about 5 s. The last 60% of rows include",
              "8-16 rows after the process exited. review.json has both aligned to the token windows instead."]
    with open(os.path.join(HERE, "analysis.md"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
