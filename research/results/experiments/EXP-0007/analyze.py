#!/usr/bin/env python3
"""EXP-0007: the arm tables, the pairs and the four hypotheses, from runs.json
(summarize.py), attrib.json (attrib.py) and the timelines.

    python analyze.py        (writes analysis.json and analysis.md)

Windows are decode tokens (= passes). The pg arms generate 128 tokens, so their
3-130 window holds the tokens they have. "no-API idle" is GPU idle while the
decode thread is in no CUDA call, where a blocking prefetch call shows.
"""

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["prof", "prof-pf", "prof-pq", "addr", "addr-pf", "addr-pq", "prof-lo", "pg", "pg-pf"]
WINS = ["387-1022", "3-1022", "3-130"]
PAIRS = [("prof-pf", "prof"), ("prof-pq", "prof"), ("prof-pq", "prof-pf"),
         ("addr-pf", "addr"), ("addr-pq", "addr"), ("addr-pq", "addr-pf"),
         ("prof-lo", "prof"), ("pg-pf", "pg")]
# by the gap since the host last read the expert; "first" (not read since the load) has no known gap, so the
# monotone test covers the other three and the standardization keeps it as a stratum of its own
ORDER = ["1-15", "16-127", "128+", "first"]
MONO = ["1-15", "16-127", "128+"]
COLS = ["ms", "compute", "dma", "idle", "idle_slow", "idle_fast", "idle_sync", "idle_noapi", "slow_per_token", "pinned_share"]


def no_api_idle(tag, lo, hi):
    p = os.path.join(HERE, f"timeline_{tag}.json")
    if not os.path.exists(p):
        return None
    s = [t for t in json.load(open(p))["tokens"] if lo <= t["token"] <= hi]
    return sum(t["idle_while_host_in_ns"]["no_api"] for t in s) / len(s) / 1e6 if s else None


def rep(runs, tag):
    r = runs.get(tag)
    if r is None:
        return None
    out = dict(prompt_eval_s=r.get("log", {}).get("prompt_eval_ms", 0) / 1000, prefill=r.get("prefill"),
               avail_min_mib=r.get("proc", {}).get("avail_min_mib"), flag=r.get("flag"),
               sha=r.get("stdout", {}).get("sha256"), win={})
    for w in WINS:
        x = (r.get("windows") or {}).get(w)
        if not x:
            continue
        lo, hi = map(int, w.split("-"))
        d = {k: x.get(k) for k in COLS if k in x}
        d["n"] = x["n"]
        d["idle_noapi"] = no_api_idle(tag, lo, hi)
        out["win"][w] = d
    return out


def merged_bins(bins, min_n=50):
    """The MONO bins in order, a bin with fewer than min_n uploads merged into its neighbour (the previous one, or the next for the first)."""
    b = [[name, bins[name]["n"], bins[name]["slow"]] for name in MONO]
    while len(b) > 1:
        i = next((i for i, x in enumerate(b) if x[1] < min_n), None)
        if i is None:
            break
        lo, hi = (i - 1, i) if i > 0 else (0, 1)
        b[lo:hi + 1] = [[b[lo][0] + "+" + b[hi][0], b[lo][1] + b[hi][1], b[lo][2] + b[hi][2]]]
    return b


def pool_bins(att, tags, w):
    out = {}
    for t in tags:
        for name, v in att[t][w]["bins"].items():
            o = out.setdefault(name, dict(n=0, slow=0))
            o["n"] += v["n"]
            o["slow"] += v["slow"]
    return out


def h1(att, w="3-1022"):
    res = {}
    need = ["prof_r1", "prof_r2", "addr_r1", "addr_r2"]
    if not all(t in att and w in att[t] for t in need):
        return dict(decided=False, reason="missing attribution")
    pooled = {a: pool_bins(att, [f"{a}_r1", f"{a}_r2"], w) for a in ("prof", "addr")}
    mono = {}
    for a in ("prof", "addr"):
        b = merged_bins(pooled[a])
        shares = [s / n if n else None for _, n, s in b]
        mono[a] = dict(bins=[(name, n, s, (s / n if n else None)) for name, n, s in b],
                       monotone=all(shares[i] is not None and shares[i + 1] is not None and shares[i + 1] >= shares[i] for i in range(len(shares) - 1)))
    # standardization on bins merged so that both arms have at least 50 uploads in each
    groups = [[o] for o in ORDER]
    while True:
        bad = [i for i, g in enumerate(groups) if min(sum(pooled[a][x]["n"] for x in g) for a in ("prof", "addr")) < 50]
        if not bad or len(groups) == 1:
            break
        i = bad[0]
        j = i - 1 if i > 0 else i + 1
        lo, hi = min(i, j), max(i, j)
        groups[lo:hi + 1] = [groups[lo] + groups[hi]]
    tot = {a: (sum(v["n"] for v in pooled[a].values()), sum(v["slow"] for v in pooled[a].values())) for a in ("prof", "addr")}
    obs = {a: tot[a][1] / tot[a][0] for a in ("prof", "addr")}
    pred = 0.0
    for g in groups:
        n_p = sum(pooled["prof"][x]["n"] for x in g)
        n_a = sum(pooled["addr"][x]["n"] for x in g)
        s_a = sum(pooled["addr"][x]["slow"] for x in g)
        pred += n_p / tot["prof"][0] * (s_a / n_a)
    gap = obs["prof"] - obs["addr"]
    cover = (pred - obs["addr"]) / gap if gap else None
    res = dict(window=w, monotone=mono, groups=["+".join(g) for g in groups], observed=obs, predicted_prof=pred,
               gap=gap, covered=cover,
               falsified=(not mono["prof"]["monotone"]) or (not mono["addr"]["monotone"]) or cover is None or cover < 0.5)
    return res


def pair_rows(reps, a, b, w, key="ms"):
    rows = []
    for k in ("r1", "r2"):
        x, y = reps.get(f"{a}_{k}"), reps.get(f"{b}_{k}")
        if x and y and w in x["win"] and w in y["win"] and x["win"][w].get(key) is not None and y["win"][w].get(key) is not None:
            rows.append(x["win"][w][key] - y["win"][w][key])
    return rows


def main():
    runs = json.load(open(os.path.join(HERE, "runs.json")))
    att = json.load(open(os.path.join(HERE, "attrib.json"))) if os.path.exists(os.path.join(HERE, "attrib.json")) else {}
    reps = {t: rep(runs, t) for t in runs if not t.endswith("_w")}
    reps = {t: v for t, v in reps.items() if v}

    arms = {}
    for a in ARMS:
        rs = [reps.get(f"{a}_r1"), reps.get(f"{a}_r2")]
        if not all(rs):
            continue
        m = dict(prompt_eval_s=statistics.mean(r["prompt_eval_s"] for r in rs), win={})
        for w in WINS:
            if not all(w in r["win"] for r in rs):
                continue
            m["win"][w] = {k: statistics.mean(r["win"][w][k] for r in rs) for k in COLS + ["n"] if rs[0]["win"][w].get(k) is not None}
            m["win"][w]["r1"] = rs[0]["win"][w]["ms"]
            m["win"][w]["r2"] = rs[1]["win"][w]["ms"]
        for k in ("r1", "r2"):
            at = att.get(f"{a}_{k}", {})
            m[f"attrib_{k}"] = {w: {x: at[w].get(x) for x in ("pageable_per_pass", "slow_per_pass", "slow_share", "slow_us_per_pass",
                                                             "pageable_us_per_pass", "prefetch_us_per_pass", "prefetch_mib_per_pass",
                                                             "prefetch_cand_mib_per_pass", "prefetch_calls_per_pass")}
                                for w in WINS if w in at}
        arms[a] = m

    pairs = []
    for a, b in PAIRS:
        for w in WINS:
            d = pair_rows(reps, a, b, w)
            if len(d) == 2:
                pairs.append(dict(pair=f"{a} - {b}", window=w, mean=statistics.mean(d), r1=d[0], r2=d[1],
                                  pct=100 * statistics.mean(d) / arms[b]["win"][w]["ms"] if b in arms and w in arms[b]["win"] else None))

    def pr(a, b, w):
        return next((p for p in pairs if p["pair"] == f"{a} - {b}" and p["window"] == w), None)

    hyp = {"H1": h1(att) if att else dict(decided=False)}
    for name, a in (("H2", "prof-pq"), ("H2_pf", "prof-pf"), ("H2_addr_pq", "addr-pq"), ("H2_addr_pf", "addr-pf")):
        b = "addr" if a.startswith("addr") else "prof"
        p = pr(a, b, "387-1022")
        hyp[name] = dict(**(p or {"pair": f"{a} - {b}"}),
                         falsified=None if p is None else not (p["mean"] <= -0.5 and p["r1"] < 0 and p["r2"] < 0))
    p_ms, p_idle = pr("pg-pf", "pg", "3-130"), None
    if all(f"{x}_{k}" in reps for x in ("pg", "pg-pf") for k in ("r1", "r2")):
        p_idle = [reps[f"pg-pf_{k}"]["win"]["3-130"]["idle_slow"] / reps[f"pg_{k}"]["win"]["3-130"]["idle_slow"] for k in ("r1", "r2")]
    # the prefetch moves the fault wait out of the copies into its own call (no-API idle), so H3 counts both
    p_idle_only = None
    if p_idle is not None:
        p_idle_only = p_idle
        p_idle = [(reps[f"pg-pf_{k}"]["win"]["3-130"]["idle_slow"] + reps[f"pg-pf_{k}"]["win"]["3-130"]["idle_noapi"]) /
                  (reps[f"pg_{k}"]["win"]["3-130"]["idle_slow"] + reps[f"pg_{k}"]["win"]["3-130"]["idle_noapi"]) for k in ("r1", "r2")]
    hyp["H3"] = dict(ms=p_ms, slow_plus_noapi_idle_ratio=p_idle, slow_idle_only_ratio=p_idle_only,
                     falsified=None if p_ms is None or p_idle is None else not (p_ms["r1"] < 0 and p_ms["r2"] < 0 and max(p_idle) <= 0.5))
    if all(f"{x}_{k}" in reps for x in ("prof", "prof-lo") for k in ("r1", "r2")):
        pe = [reps[f"prof-lo_{k}"]["prompt_eval_s"] - reps[f"prof_{k}"]["prompt_eval_s"] for k in ("r1", "r2")]
        sl = pair_rows(reps, "prof-lo", "prof", "3-1022", "slow_per_token")
        hyp["H4"] = dict(prompt_eval_s_diff=pe, slow_per_token_diff=sl, falsified=not (min(pe) > 0 and len(sl) == 2 and min(sl) > 0))

    out = dict(arms=arms, pairs=pairs, hypotheses=hyp,
               reps={t: dict(prompt_eval_s=v["prompt_eval_s"], avail_min_mib=v["avail_min_mib"], flag=v["flag"], sha=v["sha"])
                     for t, v in sorted(reps.items())})
    with open(os.path.join(HERE, "analysis.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")

    L = ["# EXP-0007 analysis (generated by analyze.py)", ""]
    for w in WINS:
        L += [f"## Tokens {w}, mean of r1 and r2", "",
              "| arm | ms/token | r1 | r2 | compute | DMA | idle | slow-copy idle | fast-copy idle | sync idle | no-API idle | slow copies/token | pinned share |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for a in ARMS:
            x = arms.get(a, {}).get("win", {}).get(w)
            if not x:
                continue
            L.append(f"| {a} | {x['ms']:.2f} | {x['r1']:.2f} | {x['r2']:.2f} | {x['compute']:.2f} | {x['dma']:.2f} | {x['idle']:.2f} | "
                     f"{x['idle_slow']:.2f} | {x['idle_fast']:.2f} | {x['idle_sync']:.2f} | {x.get('idle_noapi', 0):.2f} | "
                     f"{x['slow_per_token']:.2f} | {100 * x['pinned_share']:.1f}% |")
        L.append("")
    L += ["## Pairs (negative = first arm faster)", "", "| pair | window | mean ms | % | r1 | r2 |", "|---|---|---:|---:|---:|---:|"]
    for p in pairs:
        L.append(f"| {p['pair']} | {p['window']} | {p['mean']:+.2f} | {p['pct']:+.1f}% | {p['r1']:+.2f} | {p['r2']:+.2f} |")
    L += ["", "## Copy log (tokens 387-1022; pg arms 3-130), per pass", "",
          "| run | pageable uploads | slow | slow share | slow us | pageable us | prefetch calls | prefetch us | prefetched MiB | candidate MiB |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in sorted(att):
        if t.endswith("_w"):
            continue
        x = att[t].get("387-1022") or att[t].get("3-130")
        if not x:
            continue
        L.append(f"| {t} | {x['pageable_per_pass']:.2f} | {x['slow_per_pass']:.3f} | {100 * (x['slow_share'] or 0):.2f}% | "
                 f"{x['slow_us_per_pass']:.0f} | {x['pageable_us_per_pass']:.0f} | {x['prefetch_calls_per_pass']:.1f} | "
                 f"{x['prefetch_us_per_pass']:.0f} | {x['prefetch_mib_per_pass']:.2f} | {x['prefetch_cand_mib_per_pass']:.2f} |")
    L += ["", "## Hypotheses", "", "```", json.dumps(hyp, indent=1, sort_keys=True), "```", ""]
    L += ["## Runs", "", "| run | prompt eval s | lowest available MiB | stdout sha256 | flag |", "|---|---:|---:|---|---|"]
    for t, v in sorted(out["reps"].items()):
        L.append(f"| {t} | {v['prompt_eval_s']:.2f} | {v['avail_min_mib'] or 0:.0f} | {v['sha']} | {v['flag'] or ''} |")
    with open(os.path.join(HERE, "analysis.md"), "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
