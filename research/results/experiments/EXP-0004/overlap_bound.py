# EXP-0004: upper bound on what overlapping the expert uploads with compute could hide.
#
#   python overlap_bound.py capture.sqlite [first_token] [last_token] [pinned_gbps]
#
# Uploads between the end of graph g[il] and the start of g[il+1] belong to MoE layer il.
# The bound simulates an ideal copy engine that may start layer il's upload once split
# S(il+1-L) has started (L = prediction lookahead in layers), with no host idle at all,
# and reports how much of the serial compute + upload time it hides. Written by an
# adversarial reviewer of EXP-0004 and re-run unchanged for the record.
import sqlite3, sys, collections, json, statistics

db = sys.argv[1]
lo_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 3
hi_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

graphs = list(con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE ORDER BY start"))
kernels = list(con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"))
mem = list(con.execute("SELECT start, end, bytes, copyKind, srcKind, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"))
d2h_sizes = collections.Counter(m[2] for m in mem if m[3] == 2 and m[2] >= 64 * 1024)
logits = d2h_sizes.most_common(1)[0][0]
ends = [m[1] for m in mem if m[3] == 2 and m[2] == logits]
print(db, "logits bytes", logits, "decode steps", len(ends) - 1)

import bisect
gstarts = [g[0] for g in graphs]
mstarts = [m[0] for m in mem]

rows = []
for k in range(max(1 + 2, lo_tok), min(len(ends) - 1, hi_tok)):
    lo, hi = ends[k - 1], ends[k]
    i0 = bisect.bisect_left(gstarts, lo)
    i1 = bisect.bisect_left(gstarts, hi)
    gs = graphs[i0:i1]
    if len(gs) != 41:
        rows.append(None)
        continue
    j0 = bisect.bisect_left(mstarts, lo)
    j1 = bisect.bisect_left(mstarts, hi)
    ms = [m for m in mem[j0:j1] if m[3] == 1 and m[2] >= 512 * 1024]
    # uploads for layer il lie between end of g[il] and start of g[il+1]
    per = []
    for il in range(40):
        a, b = gs[il][1], gs[il + 1][0]
        up = [m for m in ms if a <= m[0] < b]
        # union of GPU-side upload intervals
        iv = sorted((m[0], m[1]) for m in up)
        tot = 0; cur = None
        for s, e in iv:
            if cur is None or s > cur[1]:
                if cur: tot += cur[1] - cur[0]
                cur = [s, e]
            else:
                cur[1] = max(cur[1], e)
        if cur: tot += cur[1] - cur[0]
        per.append({
            "il": il,
            "u": tot,                       # upload ns (GPU side) for layer il
            "ubytes": sum(m[2] for m in up),
            "n_up": len(up),
            "c_prev": gs[il][1] - gs[il][0],  # span of split before the upload (S(il-1), or the layer-0 split)
            "c_own": gs[il + 1][1] - gs[il + 1][0],  # span of S(il)
            "gap": b - a,                   # end of previous split to start of S(il): DMA + idle
        })
    rows.append({"k": k, "t": hi - lo, "per": per, "g0": gs[0][1] - gs[0][0]})

R = [r for r in rows if r]
n = len(R)
print("tokens used", n, "skipped (graph count != 41)", len(rows) - n)
T = sum(r["t"] for r in R) / n / 1e6
C = sum(r["g0"] + sum(p["c_own"] for p in r["per"]) for r in R) / n / 1e6
U = sum(sum(p["u"] for p in r["per"]) for r in R) / n / 1e6
print(f"ms/token {T:.2f}  graph-span compute {C:.2f}  upload union {U:.2f}")
tok_bound = min(C, U)
la1 = sum(sum(min(p["u"], p["c_prev"]) for p in r["per"]) for r in R) / n / 1e6
same = sum(sum(min(p["u"], p["c_own"]) for p in r["per"]) for r in R) / n / 1e6
# one-layer lookahead, but split S(39) holds the LM head; also the first token split
print(f"token bound min(C,U) = {tok_bound:.2f} ms ; per-layer 1-ahead sum min(u_il, span S(il-1)) = {la1:.2f} ms ; same-split sum min(u_il, span S(il)) = {same:.2f}")
# per-layer stats
zero = sum(1 for r in R for p in r["per"] if p["n_up"] == 0) / n
print(f"layers with zero uploads per token: {zero:.2f} of 40")
cprev = [p["c_prev"] / 1e3 for r in R for p in r["per"]]
cown = [p["c_own"] / 1e3 for r in R for p in r["per"]]
u = [p["u"] / 1e3 for r in R for p in r["per"]]
def q(xs, f):
    xs = sorted(xs); return xs[int(f * (len(xs) - 1))]
print("split span us: p10 %.0f median %.0f p90 %.0f max %.0f" % (q(cown, .1), q(cown, .5), q(cown, .9), max(cown)))
print("upload per layer us: p10 %.0f median %.0f p90 %.0f max %.0f mean %.0f" % (q(u, .1), q(u, .5), q(u, .9), max(u), statistics.mean(u)))
g0 = statistics.mean(r["g0"] for r in R) / 1e3
s39 = statistics.mean(r["per"][39]["c_own"] for r in R) / 1e3
s_mid = statistics.mean(p["c_own"] for r in R for p in r["per"][:39]) / 1e3
print(f"first split (layer0 attn) {g0:.0f} us ; S(0..38) mean {s_mid:.0f} us ; S(39) incl LM head {s39:.0f} us")
# share of layers where u > c_prev
frac_u_gt_c = sum(1 for r in R for p in r["per"] if p["u"] > p["c_prev"]) / (n * 40)
print(f"fraction of (token,layer) with upload > preceding split span: {frac_u_gt_c:.3f}")
# per-layer mean upload by layer index
by_il = [statistics.mean(r["per"][il]["u"] for r in R) / 1e3 for il in range(40)]
print("mean upload us by layer:", " ".join(f"{x:.0f}" for x in by_il))
by_il_c = [statistics.mean(r["per"][il]["c_own"] for r in R) / 1e3 for il in range(40)]
print("mean span us of S(il):", " ".join(f"{x:.0f}" for x in by_il_c))
gap = sum(sum(p["gap"] for p in r["per"]) for r in R) / n / 1e6
print(f"sum of inter-split gaps ms/token {gap:.2f}")
# infinite lookahead within the token but causal: an upload for layer il can only overlap compute that precedes it in the token
# greedy: copy engine processes layer uploads in order, each can start at time 0 of token (perfect prediction) ; compute runs in order, layer il's S(il) waits for upload il.
def sim(r, lookahead):
    # lookahead L: upload for layer il may start once split index (il - L) has started (g index il+1-L) ; L=1 -> during S(il-1)
    t_gpu = r["g0"]  # time when compute of first split ends (starts at 0)
    split_start = [0.0] * 41
    split_end = [0.0] * 41
    split_end[0] = r["g0"]
    copy_free = 0.0
    for il in range(40):
        p = r["per"][il]
        # earliest time upload may begin: start of split g[il+1-L]
        gi = il + 1 - lookahead
        earliest = split_start[gi] if gi >= 0 else 0.0
        st = max(copy_free, earliest)
        copy_free = st + p["u"]
        split_start[il + 1] = max(split_end[il], copy_free)
        split_end[il + 1] = split_start[il + 1] + p["c_own"]
    return split_end[40]
for L in (1, 2, 3, 40):
    v = sum(sim(r, L) for r in R) / n / 1e6
    print(f"ideal pipelined compute+upload (no host idle), lookahead {L}: {v:.2f} ms ; serial C+U = {C+U:.2f} ; hidden {C+U-v:.2f}")

# projection: all uploads at the pinned GPU rate (12.8 GB/s), per-layer
RATE = float(sys.argv[4]) if len(sys.argv) > 4 else 12.8
for r in R:
    for p in r["per"]:
        p.setdefault("u_meas", p["u"]); p["u"] = p["ubytes"] / RATE  # bytes / (GB/s) = ns
Up = sum(sum(p["u"] for p in r["per"]) for r in R) / n / 1e6
la1p = sum(sum(min(p["u"], p["c_prev"]) for p in r["per"]) for r in R) / n / 1e6
print(f"at {RATE} GB/s: upload {Up:.2f} ms/token ; token bound min(C,U) {min(C,Up):.2f} ; 1-ahead per-layer bound {la1p:.2f}")
for L in (1, 40):
    v = sum(sim(r, L) for r in R) / n / 1e6
    print(f"  ideal pipelined lookahead {L}: {v:.2f} ms ; serial {C+Up:.2f} ; hidden {C+Up-v:.2f}")
