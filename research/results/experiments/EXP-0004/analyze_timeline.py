#!/usr/bin/env python3
"""EXP-0004: per-token decomposition of a generation timeline.

Reads the sqlite export of one Nsight Systems capture of llama-completion and
splits every generated token's wall time into GPU compute, host-to-device copy,
device-to-host copy and GPU idle, plus the overlap between compute and copy.

    python analyze_timeline.py capture.sqlite --out timeline.json [--skip 2]

Token boundaries are the logits read-backs: one device-to-host copy of
n_vocab floats per decode step. Token k is the interval from the end of
read-back k-1 to the end of read-back k. The first read-back belongs to the
prompt, so the first interval is decode token 1. The first --skip tokens and
the last one are dropped.

All GPU intervals (kernels, CUDA graph launches, memcpy, memset) come from the
same GPU clock, so the overlap figures compare like with like. Host API
intervals are on the CPU clock, converted by nsys; the idle attribution that
uses them is approximate.
"""

import argparse
import collections
import json
import sqlite3
import statistics
import sys


def table_exists(con, name):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def enum_ids(con, table, needle):
    ids = [r[0] for r in con.execute(f"SELECT id, name FROM {table}") if needle in r[1].upper()]
    if not ids:
        sys.exit(f"no {needle} in {table}: {list(con.execute(f'SELECT * FROM {table}'))}")
    return ids


def union(iv):
    """Merge [start, end) intervals; input need not be sorted."""
    out = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def clip(iv, lo, hi):
    return [(max(s, lo), min(e, hi)) for s, e in iv if e > lo and s < hi]


def length(u):
    return sum(e - s for s, e in u)


def intersect(a, b):
    """Total length of the intersection of two merged interval lists."""
    i = j = 0
    tot = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def complement(u, lo, hi):
    out = []
    t = lo
    for s, e in u:
        if s > t:
            out.append((t, s))
        t = max(t, e)
    if t < hi:
        out.append((t, hi))
    return out


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", type=int, default=2, help="decode tokens to drop at the start")
    ap.add_argument("--logits-bytes", type=int, default=0, help="size of the logits read-back, 0 = detect")
    ap.add_argument("--expert-min-bytes", type=int, default=512 * 1024, help="host-to-device copies at least this big count as expert uploads; the smallest expert tensor is 576 KiB")
    ap.add_argument("--slow-gbps", type=float, default=1.5, help="expert copies whose host call runs below this rate count as slow (page faults)")
    ap.add_argument("--log-out", help="write the captured stderr of the application here")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)

    htod = enum_ids(con, "ENUM_CUDA_MEMCPY_OPER", "HTOD")
    dtoh = enum_ids(con, "ENUM_CUDA_MEMCPY_OPER", "DTOH")
    kind_name = {r[0]: r[1].replace("CUDA_MEMOPR_MEMORY_KIND_", "").lower() for r in con.execute("SELECT id, name FROM ENUM_CUDA_MEM_KIND")}

    names = dict(con.execute("SELECT id, value FROM StringIds"))

    compute = []
    n_graph_rows = n_kernel_rows = 0
    if table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        rows = list(con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"))
        n_kernel_rows = len(rows)
        compute += [(s, e, "k") for s, e in rows]
    if table_exists(con, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        rows = list(con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"))
        n_graph_rows = len(rows)
        compute += [(s, e, "g") for s, e in rows]
    compute.sort()

    memcpy = list(con.execute("SELECT start, end, bytes, copyKind, srcKind, dstKind, streamId, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"))
    memset = []
    if table_exists(con, "CUPTI_ACTIVITY_KIND_MEMSET"):
        memset = list(con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET"))

    api = []
    if table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        for s, e, nid, tid, corr in con.execute("SELECT start, end, nameId, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"):
            api.append((s, e, names.get(nid, "?").split("_v")[0], tid, corr))
    api_by_corr = {a[4]: a for a in api}

    # the application's own stderr, captured by nsys into the report
    app_log = ""
    if table_exists(con, "ProcessStreams"):
        for fid, cid in con.execute("SELECT filenameId, contentId FROM ProcessStreams"):
            if "stderr" in names.get(fid, ""):
                app_log += names.get(cid, "")
    if args.log_out:
        open(args.log_out, "w", encoding="utf-8").write(app_log)
    app = {}
    for line in app_log.splitlines():
        for key in ("prompt eval time", "eval time", "ubatch <= 8:", "large batches:", "registered", "graphs reused"):
            if key in line and key not in app:
                app[key] = line.split(": ", 1)[-1].strip()
                break

    # the logits read-back: the most common large device-to-host size
    d2h_sizes = collections.Counter(m[2] for m in memcpy if m[3] in dtoh and m[2] >= 64 * 1024)
    if args.logits_bytes:
        logits_bytes = args.logits_bytes
    elif d2h_sizes:
        logits_bytes = d2h_sizes.most_common(1)[0][0]
    else:
        sys.exit("no large device-to-host copy found, cannot delimit tokens")
    ends = [m[1] for m in memcpy if m[3] in dtoh and m[2] == logits_bytes]
    if len(ends) < args.skip + 3:
        sys.exit(f"only {len(ends)} logits read-backs of {logits_bytes} B")

    # the thread that issues the read-backs runs the decode loop
    main_tids = collections.Counter(api_by_corr[m[7]][3] for m in memcpy if m[3] in dtoh and m[2] == logits_bytes and m[7] in api_by_corr)
    main_tid = main_tids.most_common(1)[0][0] if main_tids else None

    tokens = []
    for k in range(1 + args.skip, len(ends) - 1):
        lo, hi = ends[k - 1], ends[k]
        t = hi - lo

        c_iv = clip([(s, e) for s, e, _ in compute], lo, hi)
        u_comp = union(c_iv)
        n_graph = sum(1 for s, e, w in compute if w == "g" and lo <= s < hi)
        n_kern = sum(1 for s, e, w in compute if w == "k" and lo <= s < hi)

        mc = [m for m in memcpy if m[1] > lo and m[0] < hi]
        h = [m for m in mc if m[3] in htod]
        d = [m for m in mc if m[3] in dtoh]
        u_h2d = union(clip([(m[0], m[1]) for m in h], lo, hi))
        u_d2h = union(clip([(m[0], m[1]) for m in d], lo, hi))
        u_set = union(clip(memset, lo, hi))
        u_all = union(u_comp + u_h2d + u_d2h + u_set)

        by_kind = collections.defaultdict(lambda: [0, 0, 0])  # bytes, count, union ns
        for m in h:
            by_kind[kind_name.get(m[4], str(m[4]))][0] += m[2]
            by_kind[kind_name.get(m[4], str(m[4]))][1] += 1
        for kname in by_kind:
            u = union(clip([(m[0], m[1]) for m in h if kind_name.get(m[4], str(m[4])) == kname], lo, hi))
            by_kind[kname][2] = length(u)

        expert = [m for m in h if m[2] >= args.expert_min_bytes]
        small = [m for m in h if m[2] < args.expert_min_bytes]

        # host side, on the decode thread
        a = [x for x in api if x[3] == main_tid and x[1] > lo and x[0] < hi] if main_tid is not None else []
        host = collections.defaultdict(int)
        host_n = collections.Counter()
        for s, e, n, _, _ in a:
            host[n] += min(e, hi) - max(s, lo)
            host_n[n] += 1

        # expert copies split by how fast the host call ran: the slow mode is page faults on the mmap
        host_iv = {"slow": [], "fast": []}
        for m in expert:
            ap_ = api_by_corr.get(m[7])
            if ap_ is None:
                continue
            cls = "slow" if m[2] / max(1, ap_[1] - ap_[0]) < args.slow_gbps else "fast"
            host_iv[cls].append((max(ap_[0], lo), min(ap_[1], hi)))

        # what the host was doing while the GPU was idle
        idle = complement(u_all, lo, hi)
        idle_attr = {}
        for n in ("cudaStreamSynchronize", "cudaMemcpyAsync", "cudaGraphLaunch", "cudaLaunchKernel"):
            idle_attr[n] = intersect(idle, union([(max(s, lo), min(e, hi)) for s, e, nm, _, _ in a if nm == n]))
        idle_attr["no_api"] = intersect(idle, complement(union([(max(s, lo), min(e, hi)) for s, e, _, _, _ in a]), lo, hi))
        for cls in ("slow", "fast"):
            idle_attr[f"expert_copy_{cls}"] = intersect(idle, union(host_iv[cls]))

        tokens.append({
            "token": k,
            "t_ns": t,
            "compute_ns": length(u_comp),
            "h2d_ns": length(u_h2d),
            "d2h_ns": length(u_d2h),
            "memset_ns": length(u_set),
            "busy_ns": length(u_all),
            "idle_ns": t - length(u_all),
            "overlap_compute_h2d_ns": intersect(u_comp, u_h2d),
            "overlap_compute_d2h_ns": intersect(u_comp, u_d2h),
            "h2d_bytes": sum(m[2] for m in h),
            "h2d_expert_bytes": sum(m[2] for m in expert),
            "n_h2d": len(h),
            "n_h2d_expert": len(expert),
            "n_h2d_small": len(small),
            "n_d2h": len(d),
            "n_graph_launch": n_graph,
            "n_kernel": n_kern,
            "h2d_by_src_kind": {kname: {"bytes": v[0], "n": v[1], "ns": v[2]} for kname, v in by_kind.items()},
            "host_api_ns": dict(host),
            "host_api_n": dict(host_n),
            "n_expert_copy_slow": len(host_iv["slow"]),
            "host_expert_copy_ns": {cls: length(union(iv)) for cls, iv in host_iv.items()},
            "idle_while_host_in_ns": idle_attr,
        })

    if not tokens:
        sys.exit("no tokens after skipping")

    # per-copy rates for expert uploads, GPU side and host side, by source kind and host-rate class
    def add_rate(r, m, a):
        r["n"] += 1
        r["bytes"] += m[2]
        r["gpu_ns"] += m[1] - m[0]
        r["gpu_gbps"].append(m[2] / max(1, m[1] - m[0]))
        if a is not None:
            r["host_ns"] += a[1] - a[0]
            r["host_gbps"].append(m[2] / max(1, a[1] - a[0]))

    rates = collections.defaultdict(lambda: {"gpu_gbps": [], "host_gbps": [], "bytes": 0, "gpu_ns": 0, "host_ns": 0, "n": 0})
    lo_all, hi_all = ends[args.skip], ends[-2]
    for m in memcpy:
        if m[3] not in htod or m[2] < args.expert_min_bytes or not (lo_all <= m[0] < hi_all):
            continue
        a = api_by_corr.get(m[7])
        kname = kind_name.get(m[4], str(m[4]))
        add_rate(rates[kname], m, a)
        if a is not None:
            add_rate(rates[kname + ("/slow" if m[2] / max(1, a[1] - a[0]) < args.slow_gbps else "/fast")], m, a)

    def dist(xs):
        return {"p10": pct(xs, 0.1), "median": pct(xs, 0.5), "p90": pct(xs, 0.9)} if xs else None

    copy_rates = {}
    for kname, r in rates.items():
        copy_rates[kname] = {
            "n": r["n"],
            "bytes": r["bytes"],
            "aggregate_gpu_gbps": r["bytes"] / r["gpu_ns"] if r["gpu_ns"] else None,
            "aggregate_host_gbps": r["bytes"] / r["host_ns"] if r["host_ns"] else None,
            "per_copy_gpu_gbps": dist(r["gpu_gbps"]),
            "per_copy_host_gbps": dist(r["host_gbps"]),
        }

    # every decode step, for checking the trace against the cache's own upload counter
    all_decode_h2d = sum(m[2] for m in memcpy if m[3] in htod and m[2] >= args.expert_min_bytes and ends[0] <= m[0] < ends[-1])

    tot = lambda key: sum(x[key] for x in tokens)
    T = tot("t_ns")
    summary = {
        "n_tokens": len(tokens),
        "logits_bytes": logits_bytes,
        "kernel_rows": n_kernel_rows,
        "graph_rows": n_graph_rows,
        "ms_per_token_mean": T / len(tokens) / 1e6,
        "ms_per_token_median": statistics.median(x["t_ns"] for x in tokens) / 1e6,
        "share": {k: tot(k) / T for k in ("compute_ns", "h2d_ns", "d2h_ns", "memset_ns", "idle_ns", "overlap_compute_h2d_ns")},
        "ms_per_token": {k: tot(k) / len(tokens) / 1e6 for k in ("compute_ns", "h2d_ns", "d2h_ns", "memset_ns", "busy_ns", "idle_ns", "overlap_compute_h2d_ns", "overlap_compute_d2h_ns")},
        "overlap_over_min_compute_h2d": tot("overlap_compute_h2d_ns") / max(1, min(tot("compute_ns"), tot("h2d_ns"))),
        "h2d_mib_per_token": tot("h2d_bytes") / len(tokens) / 2**20,
        "h2d_expert_mib_per_token": tot("h2d_expert_bytes") / len(tokens) / 2**20,
        "h2d_expert_mib_total": tot("h2d_expert_bytes") / 2**20,
        "h2d_expert_mib_all_decode": all_decode_h2d / 2**20,
        "n_decode_steps": len(ends) - 1,
        "h2d_gbps_while_copying": tot("h2d_bytes") / max(1, tot("h2d_ns")),
        "h2d_gbps_over_token_time": tot("h2d_bytes") / T,
        "per_token": {k: statistics.mean(x[k] for x in tokens) for k in ("n_h2d", "n_h2d_expert", "n_h2d_small", "n_d2h", "n_graph_launch", "n_kernel")},
        "host_api_ms_per_token": {n: sum(x["host_api_ns"].get(n, 0) for x in tokens) / len(tokens) / 1e6 for n in sorted({n for x in tokens for n in x["host_api_ns"]})},
        "host_api_calls_per_token": {n: sum(x["host_api_n"].get(n, 0) for x in tokens) / len(tokens) for n in sorted({n for x in tokens for n in x["host_api_n"]})},
        "idle_ms_per_token_while_host_in": {n: sum(x["idle_while_host_in_ns"][n] for x in tokens) / len(tokens) / 1e6 for n in tokens[0]["idle_while_host_in_ns"]},
        "h2d_by_src_kind_mib_per_token": {kn: sum(x["h2d_by_src_kind"].get(kn, {}).get("bytes", 0) for x in tokens) / len(tokens) / 2**20 for kn in sorted({kn for x in tokens for kn in x["h2d_by_src_kind"]})},
        "expert_copy_rates": copy_rates,
        "slow_gbps": args.slow_gbps,
        "expert_copies_slow_per_token": statistics.mean(x["n_expert_copy_slow"] for x in tokens),
        "host_expert_copy_ms_per_token": {cls: sum(x["host_expert_copy_ns"][cls] for x in tokens) / len(tokens) / 1e6 for cls in ("slow", "fast")},
        "app": app,
    }

    json.dump({"db": args.db, "summary": summary, "tokens": tokens}, open(args.out, "w"), indent=1)

    s = summary
    print(f"{args.db}: {s['n_tokens']} tokens, {s['ms_per_token_mean']:.2f} ms/token (median {s['ms_per_token_median']:.2f})")
    m = s["ms_per_token"]
    print(f"  per token: compute {m['compute_ns']:.2f}  h2d {m['h2d_ns']:.2f}  d2h {m['d2h_ns']:.2f}  idle {m['idle_ns']:.2f}  compute/h2d overlap {m['overlap_compute_h2d_ns']:.3f} ms")
    print(f"  shares: " + "  ".join(f"{k[:-3]} {v*100:.1f}%" for k, v in s["share"].items()))
    print(f"  overlap / min(compute, h2d) = {s['overlap_over_min_compute_h2d']*100:.2f}%")
    print(f"  h2d {s['h2d_mib_per_token']:.1f} MiB/token ({s['h2d_expert_mib_per_token']:.1f} expert), {s['h2d_gbps_while_copying']:.2f} GB/s while copying, {s['h2d_gbps_over_token_time']:.2f} GB/s over token time")
    print(f"  by source kind MiB/token: {s['h2d_by_src_kind_mib_per_token']}")
    print(f"  calls/token: {s['per_token']}")
    print(f"  host API ms/token: " + ", ".join(f"{k} {v:.2f}" for k, v in s["host_api_ms_per_token"].items() if v >= 0.05))
    print(f"  GPU idle ms/token while host in: " + ", ".join(f"{k} {v:.2f}" for k, v in s["idle_ms_per_token_while_host_in"].items()))
    print(f"  host ms/token in expert copies: slow {s['host_expert_copy_ms_per_token']['slow']:.2f}, fast {s['host_expert_copy_ms_per_token']['fast']:.2f} ({s['expert_copies_slow_per_token']:.1f} slow copies/token)")
    for k, v in s["app"].items():
        print(f"  app {k}: {v}")
    for kn, r in sorted(copy_rates.items()):
        print(f"  expert copies from {kn}: n={r['n']}, aggregate GPU {r['aggregate_gpu_gbps']:.2f} GB/s, host {r['aggregate_host_gbps'] or 0:.2f} GB/s, per-copy GPU median {r['per_copy_gpu_gbps']['median']:.2f}")


if __name__ == "__main__":
    main()
