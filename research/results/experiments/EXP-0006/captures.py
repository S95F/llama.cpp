#!/usr/bin/env python3
"""EXP-0006: figures read straight from the Nsight Systems captures, for the
claims the review recomputed (captures.json, captures.md).

    python captures.py [--cap C:/nsys/exp0006]

Per run:
  - end to end: session start to the end of decode pass 1022's logits read-back
    (model load, prefill and decode; the load varies by up to about 0.45 s);
  - registration: cudaHostRegister calls (count, total, longest), and the
    longest stream synchronization between them and the next logits read-back,
    with the GPU time inside it (a deferred stall shows as a long idle sync);
  - copies in tokens 387-1022 (by logits read-back boundaries): host-to-device
    copies by memory kind, their mean size and busy rate (bytes over the union
    of their intervals), fragments under 4 and 8 KiB;
  - from the timelines: slow (page-fault) expert copies per token and the split
    of the warm window into 387-642 and 643-1022;
  - from monitor.py: the lowest available RAM before token 3 (load, prefill)
    and from token 3 on (decode);
  - for the online arms, where their 3-1022 time against the file profile goes:
    passes 3 to N, the registering pass N + 1, N + 2 to 386, and 387-1022.
"""

import argparse
import collections
import csv
import json
import os
import sqlite3
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["addr", "prof", "on64", "on128", "prof15k", "on64x15k", "on128x15k", "ood-addr", "ood-prof", "ood-on128"]
TAGS = ["on64x15k_w"] + [f"{a}_r{r}" for a in ARMS for r in (1, 2)]
SPLIT = [(387, 642), (643, 1022)]


def union_ns(iv):
    tot, cs, ce = 0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    return tot + (ce - cs if ce is not None else 0)


def gpu_ns(c, s, e):
    """GPU kernel and copy time inside [s, e]."""
    k = c.execute("select sum(min(end, ?) - max(start, ?)) from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?", (e, s, s, e)).fetchone()[0]
    m = c.execute("select sum(min(end, ?) - max(start, ?)) from CUPTI_ACTIVITY_KIND_MEMCPY where end > ? and start < ?", (e, s, s, e)).fetchone()[0]
    return (k or 0) + (m or 0)


def capture(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    names = dict(c.execute("select id, value from StringIds"))
    ids = lambda prefix: [k for k, v in names.items() if v.startswith(prefix + "_v") or v == prefix]
    mem = list(c.execute("select start, end, bytes, copyKind, srcKind, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY order by start"))
    # the logits read-back is the most common large device-to-host copy, one per pass
    d2h = collections.Counter(m[2] for m in mem if m[3] == 2 and m[2] >= 65536)
    rb = d2h.most_common(1)[0][0]
    ends = [m[1] for m in mem if m[3] == 2 and m[2] == rb]
    out = dict(n_readbacks=len(ends))
    # ends[0] is the prompt's read-back, ends[k] decode pass k
    if len(ends) > 1022:
        out["end_to_end_s"] = ends[1022] / 1e9
    api = lambda prefix: list(c.execute(f"select start, end from CUPTI_ACTIVITY_KIND_RUNTIME where nameId in ({','.join(map(str, ids(prefix))) or '-1'}) order by start"))
    reg = api("cudaHostRegister")
    if reg:
        last = reg[-1][1]
        nxt = min((e for e in ends if e > last), default=None)
        syncs = [(e - s, s, e) for s, e in api("cudaStreamSynchronize") if s >= last and (nxt is None or e <= nxt)]
        d, s, e = max(syncs) if syncs else (0, 0, 0)
        out["register"] = dict(n=len(reg), total_ms=sum(e - s for s, e in reg) / 1e6, longest_ms=max(e - s for s, e in reg) / 1e6,
                               at_s=reg[0][0] / 1e9, before_prompt=last < ends[0],
                               sync_after_ms=d / 1e6, sync_after_gpu_ms=gpu_ns(c, s, e) / 1e6 if d else 0.0)
    if len(ends) > 1022:
        lo, hi = ends[386], ends[1022]
        h2d = [m for m in mem if m[3] == 1 and lo <= m[0] < hi]
        n_tok = 1022 - 387 + 1
        cp = dict(n_per_token=len(h2d) / n_tok, share_under_8k=sum(m[2] < 8192 for m in h2d) / len(h2d))
        for kind, nm in ((0, "pageable"), (1, "pinned")):
            k = [m for m in h2d if m[4] == kind]
            b = sum(m[2] for m in k)
            cp[nm] = dict(n_per_token=len(k) / n_tok, mean_kib=b / len(k) / 1024 if k else None,
                          busy_gbps=b / union_ns([(m[0], m[1]) for m in k]) if k else None,
                          under_4k_per_token=sum(m[2] < 4096 for m in k) / n_tok)
        # expert copies (>= 512 KiB) whose host call ran below 1.5 GB/s, as analyze_timeline.py classifies page faults
        call = {r[0]: r[1] - r[2] for r in c.execute("select correlationId, end, start from CUPTI_ACTIVITY_KIND_RUNTIME")}
        for kind, nm in ((0, "pageable"), (1, "pinned")):
            ex = [m for m in h2d if m[4] == kind and m[2] >= 512 * 1024 and m[5] in call]
            cp[nm]["expert_per_token"] = len(ex) / n_tok
            cp[nm]["expert_slow_per_token"] = sum(m[2] / max(1, call[m[5]]) < 1.5 for m in ex) / n_tok
        out["copies_387_1022"] = cp
    t0 = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()
    out["session_start_unix"] = t0[0] / 1e9 if t0 else None
    out["token3_s"] = ends[2] / 1e9 if len(ends) > 2 else None
    c.close()
    return out


def timeline(tag):
    p = os.path.join(HERE, f"timeline_{tag}.json")
    if not os.path.exists(p):
        return {}
    toks = {t["token"]: t for t in json.load(open(p))["tokens"]}
    out = {}
    for lo, hi in [(387, 1022)] + SPLIT:
        ks = [k for k in range(lo, hi + 1) if k in toks]
        out[f"{lo}-{hi}"] = dict(ms=statistics.mean(toks[k]["t_ns"] for k in ks) / 1e6,
                                 slow_copies_per_token=statistics.mean(toks[k]["n_expert_copy_slow"] for k in ks))
    return out


def memory(tag, start_unix, token3_s):
    p = os.path.join(HERE, f"proc_{tag}.csv")
    if not os.path.exists(p) or start_unix is None or token3_s is None:
        return {}
    rows = [(float(r["t_unix"]), float(r["avail_mib"])) for r in csv.DictReader(open(p))]
    t3 = start_unix + token3_s
    load = [v for t, v in rows if t < t3]
    dec = [v for t, v in rows if t >= t3]
    return dict(load_min_avail_mib=min(load) if load else None, decode_min_avail_mib=min(dec) if dec else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default="C:/nsys/exp0006")
    a = ap.parse_args()

    runs = {}
    for tag in TAGS:
        db = os.path.join(a.cap, f"{tag}.sqlite")
        if not os.path.exists(db):
            continue
        r = capture(db)
        r["timeline"] = timeline(tag)
        r["memory"] = memory(tag, r.get("session_start_unix"), r.get("token3_s"))
        runs[tag] = r

    def mean_of(arm, f):
        v = [f(runs[t]) for t in (f"{arm}_r1", f"{arm}_r2") if t in runs]
        v = [x for x in v if x is not None]
        return (statistics.mean(v), v) if v else (None, v)

    pairs = {}
    for p, q in (("on64", "addr"), ("on128", "addr"), ("on64", "prof"), ("on128", "prof"), ("on64x15k", "prof15k"),
                 ("on128x15k", "prof15k"), ("ood-on128", "ood-prof"), ("ood-on128", "ood-addr"), ("ood-prof", "ood-addr"), ("prof", "addr")):
        d = [runs[f"{p}_r{r}"]["end_to_end_s"] - runs[f"{q}_r{r}"]["end_to_end_s"] for r in (1, 2)
             if f"{p}_r{r}" in runs and f"{q}_r{r}" in runs]
        d2 = {w: [runs[f"{p}_r{r}"]["timeline"][w]["ms"] - runs[f"{q}_r{r}"]["timeline"][w]["ms"] for r in (1, 2)] for w in [f"{lo}-{hi}" for lo, hi in SPLIT]}
        pairs[f"{p} - {q}"] = dict(end_to_end_s=d, end_to_end_mean_s=statistics.mean(d) if d else None, split_ms=d2)

    phases = {}
    for on, ref, n in (("on64", "prof", 64), ("on128", "prof", 128), ("on64x15k", "prof15k", 64), ("on128x15k", "prof15k", 128),
                       ("ood-on128", "ood-prof", 128)):
        for r in (1, 2):
            x, y = (json.load(open(os.path.join(HERE, f"timeline_{t}_r{r}.json")))["tokens"] for t in (on, ref))
            x, y = ({t["token"]: t["t_ns"] / 1e6 for t in z} for z in (x, y))
            d = lambda lo, hi: sum(x[k] - y[k] for k in range(lo, hi + 1) if k in x and k in y)
            tot = d(3, 1022)
            ph = dict(before=d(3, n), registering=d(n + 1, n + 1), after_to_386=d(n + 2, 386), warm=d(387, 1022))
            phases[f"{on} - {ref} r{r}"] = dict(total_s=tot / 1000, **{k + "_s": v / 1000 for k, v in ph.items()})

    out = dict(runs=runs, pairs=pairs, phases=phases)
    with open(os.path.join(HERE, "captures.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")

    L = ["# EXP-0006 captures (generated by captures.py)", "",
         "## End to end (session start to the end of decode pass 1022) and the split warm window, first arm minus second", "",
         "| pair | end to end s, mean | r1 | r2 | 387-642 ms, r1 / r2 | 643-1022 ms, r1 / r2 |", "|---|---:|---:|---:|---|---|"]
    for k, d in pairs.items():
        e = d["end_to_end_s"]
        sp = [" / ".join(f"{v:+.2f}" for v in d["split_ms"][f"{lo}-{hi}"]) for lo, hi in SPLIT]
        L.append(f"| {k} | {d['end_to_end_mean_s']:+.2f} | {e[0]:+.2f} | {e[1]:+.2f} | {sp[0]} | {sp[1]} |")
    L += ["", "## Registration and the longest stream synchronization after it (before the next logits read-back)", "",
          "| run | calls | calls ms | longest call ms | at s | sync after ms | GPU time in it ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for t, r in runs.items():
        g = r.get("register")
        if g:
            L.append(f"| {t} | {g['n']} | {g['total_ms']:.0f} | {g['longest_ms']:.0f} | {g['at_s']:.1f} | {g['sync_after_ms']:.0f} | {g['sync_after_gpu_ms']:.1f} |")
    L += ["", "## Copies in tokens 387-1022, per token (r1 and r2)", "",
          "| run | H2D copies | under 8 KiB | pinned | pinned mean KiB | pinned GB/s | pinned under 4 KiB | pageable | pageable mean KiB | pageable GB/s | slow copies | pageable expert copies | slow among them | slow pinned | ms 387-642 | ms 643-1022 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t, r in runs.items():
        cp, tl = r.get("copies_387_1022"), r["timeline"]
        if not cp or not tl:
            continue
        pn, pg = cp["pinned"], cp["pageable"]
        fm = lambda v, s=".1f": format(v, s) if v is not None else "-"
        L.append(f"| {t} | {cp['n_per_token']:.0f} | {100 * cp['share_under_8k']:.1f}% | {pn['n_per_token']:.0f} | {fm(pn['mean_kib'], '.0f')} | "
                 f"{fm(pn['busy_gbps'], '.2f')} | {pn['under_4k_per_token']:.1f} | {pg['n_per_token']:.0f} | {fm(pg['mean_kib'], '.0f')} | "
                 f"{fm(pg['busy_gbps'], '.2f')} | {tl['387-1022']['slow_copies_per_token']:.2f} | {pg['expert_per_token']:.1f} | "
                 f"{pg['expert_slow_per_token']:.2f} ({100 * pg['expert_slow_per_token'] / max(1e-9, pg['expert_per_token']):.1f}%) | "
                 f"{pn['expert_slow_per_token']:.2f} | {tl['387-642']['ms']:.2f} | {tl['643-1022']['ms']:.2f} |")
    L += ["", "## Where the online arms' 3-1022 time against the file profile goes (seconds, online minus profile)", "",
          "| pair | total | passes 3-N | pass N+1 | N+2-386 | 387-1022 |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in phases.items():
        L.append(f"| {k} | {v['total_s']:+.2f} | {v['before_s']:+.2f} | {v['registering_s']:+.2f} | "
                 f"{v['after_to_386_s']:+.2f} | {v['warm_s']:+.2f} |")
    L += ["", "## Lowest available RAM (monitor.py, MiB)", "", "| run | load and prefill | decode |", "|---|---:|---:|"]
    for t, r in runs.items():
        m = r["memory"]
        L.append(f"| {t} | {m.get('load_min_avail_mib', 0):.0f} | {m.get('decode_min_avail_mib', 0):.0f} |")
    with open(os.path.join(HERE, "captures.md"), "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
