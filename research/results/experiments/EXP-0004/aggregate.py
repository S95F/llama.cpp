#!/usr/bin/env python3
"""EXP-0004: one table across arms from the per-capture timeline JSONs.

    python aggregate.py research/results/experiments/EXP-0004

Writes decomposition.csv (one row per traced capture) and prints a markdown
table of the per-arm means with the spread between the two traced repetitions.
"""

import csv
import glob
import json
import os
import re
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "research/results/experiments/EXP-0004"


def eval_ms(path):
    """ms per token from llama's own 'eval time' line, or None."""
    if not os.path.exists(path):
        return None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"\beval time =\s+([\d.]+) ms /\s+(\d+) runs", line)
        if m and "prompt" not in line:
            return float(m.group(1)) / int(m.group(2))
    return None


def cache_line(path):
    if not os.path.exists(path):
        return None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"ubatch <= 8: hits = (\d+), misses = (\d+), hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB", line)
        if m:
            return float(m.group(3)), float(m.group(4))
    return None, None


def pinned_line(path):
    if not os.path.exists(path):
        return None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"registered ([\d.]+) MiB of host experts as pinned memory \(([\d.]+) MiB left pageable\)", line)
        if m:
            return float(m.group(1))
    return 0.0


rows = []
for f in sorted(glob.glob(os.path.join(d, "timeline_*_t*.json"))):
    tag = os.path.basename(f)[len("timeline_"):-len(".json")]
    arm, rep = tag.rsplit("_", 1)
    s = json.load(open(f))["summary"]
    m = s["ms_per_token"]
    idle = s["idle_ms_per_token_while_host_in"]
    hit, up = cache_line(os.path.join(d, f"log_{tag}.txt"))
    rates = s["expert_copy_rates"]
    pg_fast = rates.get("pageable/fast", {})
    pn = rates.get("pinned", {})
    rows.append({
        "arm": arm,
        "rep": rep,
        "mib": int(arm.split("-")[0]),
        "mode": arm.split("-")[1],
        "tokens": s["n_tokens"],
        "ms_token": s["ms_per_token_mean"],
        "app_eval_ms_traced": eval_ms(os.path.join(d, f"log_{tag}.txt")),
        "app_eval_ms_untraced": eval_ms(os.path.join(d, f"log_{arm}_u.txt")),
        "compute_ms": m["compute_ns"],
        "h2d_ms": m["h2d_ns"],
        "d2h_ms": m["d2h_ns"],
        "idle_ms": m["idle_ns"],
        "overlap_ms": m["overlap_compute_h2d_ns"],
        "idle_slow_copy_ms": idle.get("expert_copy_slow", 0.0),
        "idle_fast_copy_ms": idle.get("expert_copy_fast", 0.0),
        "idle_sync_ms": idle.get("cudaStreamSynchronize", 0.0),
        "idle_no_api_ms": idle.get("no_api", 0.0),
        "idle_other_ms": m["idle_ns"] - sum(idle.get(k, 0.0) for k in ("expert_copy_slow", "expert_copy_fast", "cudaStreamSynchronize", "no_api")),
        "h2d_mib_token": s["h2d_expert_mib_per_token"],
        "h2d_pinned_mib_token": s["h2d_by_src_kind_mib_per_token"].get("pinned", 0.0),
        "gpu_gbps_while_copying": s["h2d_gbps_while_copying"],
        "slow_copies_token": s["expert_copies_slow_per_token"],
        "host_slow_ms": s["host_expert_copy_ms_per_token"]["slow"],
        "host_fast_ms": s["host_expert_copy_ms_per_token"]["fast"],
        "pageable_fast_gpu_gbps": pg_fast.get("aggregate_gpu_gbps"),
        "pageable_fast_host_gbps": pg_fast.get("aggregate_host_gbps"),
        "pinned_gpu_gbps": pn.get("aggregate_gpu_gbps"),
        "pinned_host_gbps": pn.get("aggregate_host_gbps"),
        "graph_launch_token": s["per_token"]["n_graph_launch"],
        "kernel_token": s["per_token"]["n_kernel"],
        "sync_calls_token": s["host_api_calls_per_token"].get("cudaStreamSynchronize"),
        "cache_hit_rate": hit,
        "cache_uploaded_mib": up,
        "trace_uploaded_mib": s.get("h2d_expert_mib_all_decode"),
        "pinned_mib": pinned_line(os.path.join(d, f"log_{tag}.txt")),
    })

if not rows:
    sys.exit("no timeline_*_t*.json")

with open(os.path.join(d, "decomposition.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, p=1):
    return "-" if x is None else f"{x:.{p}f}"


arms = []
for r in rows:
    if r["arm"] not in arms:
        arms.append(r["arm"])

print("| arm | ms/token traced | untraced | compute | DMA | idle | of which slow copies | fast-copy staging | sync | other | overlap | MiB/token | slow copies/token |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for a in arms:
    rs = [r for r in rows if r["arm"] == a]
    g = lambda k: mean([r[k] for r in rs])
    spread = max(r["ms_token"] for r in rs) - min(r["ms_token"] for r in rs)
    print(f"| {a} | {fmt(g('ms_token'))} (+-{spread/2:.1f}) | {fmt(g('app_eval_ms_untraced'))} | {fmt(g('compute_ms'))} | {fmt(g('h2d_ms'))} | {fmt(g('idle_ms'))} | {fmt(g('idle_slow_copy_ms'))} | {fmt(g('idle_fast_copy_ms'))} | {fmt(g('idle_sync_ms'))} | {fmt(g('idle_no_api_ms') + g('idle_other_ms'))} | {fmt(g('overlap_ms'), 3)} | {fmt(g('h2d_mib_token'))} | {fmt(g('slow_copies_token'))} |")

print()
print("| arm | cache hit rate | cache MiB | trace MiB | GPU GB/s copying | pageable fast host GB/s | pinned GPU GB/s | pinned host GB/s | pinned MiB |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for a in arms:
    rs = [r for r in rows if r["arm"] == a]
    g = lambda k: mean([r[k] for r in rs])
    hits = sorted({r["cache_hit_rate"] for r in rs if r["cache_hit_rate"] is not None})
    print(f"| {a} | {'/'.join(fmt(h, 2) for h in hits)} | {fmt(g('cache_uploaded_mib'), 2)} | {fmt(g('trace_uploaded_mib'), 2)} | {fmt(g('gpu_gbps_while_copying'), 2)} | {fmt(g('pageable_fast_host_gbps'), 2)} | {fmt(g('pinned_gpu_gbps'), 2)} | {fmt(g('pinned_host_gbps'), 2)} | {fmt(g('pinned_mib'), 0)} |")
