#!/usr/bin/env python3
"""EXP-0005 step 2: how much of the warm expert upload traffic each pinning
order covers at a given pinned size, computed offline (CPU only).

Per-(layer, expert) misses come from replaying the EXP-0001 selection traces
through the validated simulator at the cache size of the runs (upstream SLRU
with the protected ramp, which is what the branch runs above the crossover).
Tensor addresses and sizes come from the GGUF itself, so the address order is
the one pin_host_experts() uses: page-rounded tensor ranges, sorted by address,
first fit against the cap.

Orders compared at each cap:
  address  - the current code
  layer    - whole tensors, most missed bytes per byte first, first fit
  expert   - (layer, expert) units, all three tensors, most misses per byte first

Each profile-driven order is scored on the trace it was built from (oracle) and
leave-one-out on the other three traces (held out), since a profile from one
prompt must cover another.

  python -I coverage.py <model.gguf> <layout.json> <trace.csv>... --json out.json
"""

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "moe-bench"))
sys.path.insert(0, os.path.join(ROOT, "gguf-py"))

import moe_sim  # noqa: E402
from gguf import GGUFReader  # noqa: E402

PAGE = 4096
MIB = 1024 * 1024
TENSORS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def expert_tensors(model):
    """{(il, name): (file offset, nbytes, ne2)} for the expert tensors."""
    rd = GGUFReader(model)
    out = {}
    for t in rd.tensors:
        parts = t.name.split(".")
        if len(parts) == 4 and parts[0] == "blk" and parts[2] in TENSORS:
            ne2 = int(t.shape[2]) if len(t.shape) > 2 else 1
            out[(int(parts[1]), parts[2])] = (int(t.data_offset), int(t.n_bytes), ne2)
    return out


def replay(trace, layout, cache_mib, skip_tokens):
    """Per-(layer, expert) fills of the small-batch calls after the first
    skip_tokens tokens, plus the simulated totals to check against the trace."""
    groups = moe_sim.build_groups(layout)
    total_host = sum(g.host_bytes for g in groups)
    n_layer, n_expert = layout["n_layer"], layout["n_expert"]
    caches, layer_group = {}, {}
    for gi, g in enumerate(groups):
        n_slots = g.n_slots_for(int(cache_mib * MIB * g.host_bytes / total_host))
        for il in g.layers:
            layer_group[il] = gi
        net = len(g.layers) * 8
        caches[gi] = moe_sim.POLICIES["slru-ramp"](n_layer, n_expert, n_slots, net, None)

    fills = collections.Counter()
    events = []
    sim_fills = rec_fills = calls = 0
    tok = 0
    last_layer = None
    for il, ids, n_ids, rec_miss in trace:
        if n_ids > 8 * 8:
            continue
        # one token is one sweep of the 40 layers; count tokens by wrap-around
        if last_layer is not None and il <= last_layer:
            tok += 1
        last_layer = il
        c = caches[layer_group[il]]
        miss = [e for e in set(ids) if c.slot_map[il][e] < 0]
        res = c.plan(il, ids)
        n_fill = res[1] if res is not None else 0
        assert n_fill == len(miss), (il, ids, n_fill, miss)
        sim_fills += n_fill
        rec_fills += rec_miss
        calls += 1
        for e in set(ids):
            events.append((tok, il, e, e in miss))
        if tok >= skip_tokens:
            for e in miss:
                fills[(il, e)] += 1
    return fills, events, dict(calls=calls, tokens=tok + 1, sim_fills=sim_fills, rec_fills=rec_fills)


def load_trace(path):
    rows = []
    import csv
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                il = int(r["layer"])
                ids = [int(x) for x in r["ids"].split()] if r["ids"] else []
                n_ids, n_miss = int(r["n_ids"]), int(r["n_miss"])
            except (TypeError, ValueError, KeyError):
                continue
            if ids:
                rows.append((il, ids, n_ids, n_miss))
    return rows


def page_range(off, n):
    return off // PAGE * PAGE, (off + n + PAGE - 1) // PAGE * PAGE


def miss_bytes(fills, tens):
    """Missed bytes per (il, tensor) and per (il, expert)."""
    per_tensor = collections.Counter()
    per_expert = collections.Counter()
    for (il, e), n in fills.items():
        for name in TENSORS:
            off, nb, ne2 = tens[(il, name)]
            b = n * (nb // ne2)
            per_tensor[(il, name)] += b
            per_expert[(il, e)] += b
    return per_tensor, per_expert


def pin_tensors(tens, cap, key):
    """First fit of page-rounded tensor ranges in the order given by key."""
    items = []
    for k, (off, nb, ne2) in tens.items():
        a, b = page_range(off, nb)
        items.append((key(k, a), k, b - a))
    items.sort()
    pinned, used = set(), 0
    for _, k, n in items:
        if used + n <= cap:
            pinned.add(k)
            used += n
    return pinned, used


def covered_by_tensors(pinned, per_tensor):
    tot = sum(per_tensor.values())
    return sum(per_tensor[k] for k in pinned) / tot if tot else 0.0


def pin_experts(tens, cap, score):
    """Greedy (layer, expert) units by score, all three tensors of an expert.
    Page rounding is charged per unit (an upper bound on the real cost once
    adjacent units merge into runs)."""
    units = []
    for il in sorted({k[0] for k in tens}):
        ne2 = tens[(il, TENSORS[0])][2]
        for e in range(ne2):
            nb = 0
            for name in TENSORS:
                off, tb, n2 = tens[(il, name)]
                per = tb // n2
                a, b = page_range(off + e * per, per)
                nb += b - a
            units.append((-score.get((il, e), 0.0), il, e, nb))
    units.sort()
    pinned, used = set(), 0
    for s, il, e, nb in units:
        if used + nb <= cap:
            pinned.add((il, e))
            used += nb
    return pinned, used


def runs_of(pinned, tens):
    """Number of contiguous registration runs an expert-level pin set needs."""
    n = 0
    for il in sorted({k[0] for k in tens}):
        ne2 = tens[(il, TENSORS[0])][2]
        inrun = False
        for e in range(ne2):
            if (il, e) in pinned:
                if not inrun:
                    n += 1
                inrun = True
            else:
                inrun = False
    return n * len(TENSORS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("layout")
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--cache-mib", type=int, default=2048)
    ap.add_argument("--skip-tokens", type=int, default=16)
    ap.add_argument("--caps", default="7842,12000,15000,15266")
    ap.add_argument("--json")
    a = ap.parse_args()

    layout = json.load(open(a.layout))
    tens = expert_tensors(a.model)
    total_expert = sum(page_range(off, nb)[1] - page_range(off, nb)[0] for off, nb, _ in tens.values())

    profiles = {}
    events = {}
    checks = {}
    for p in a.traces:
        name = os.path.splitext(os.path.basename(p))[0]
        fills, ev, chk = replay(load_trace(p), layout, a.cache_mib, a.skip_tokens)
        profiles[name] = miss_bytes(fills, tens)
        events[name] = ev
        checks[name] = chk
        print(f"{name}: {chk['tokens']} tokens, {chk['calls']} calls, simulated fills {chk['sim_fills']}, "
              f"recorded {chk['rec_fills']}", file=sys.stderr)

    names = list(profiles)
    caps = [int(c) for c in a.caps.split(",")]
    out = dict(cache_mib=a.cache_mib, skip_tokens=a.skip_tokens, expert_mib=total_expert / MIB,
               checks=checks, caps={})

    # per-layer share of the missed bytes, all traces pooled
    pooled_layer = collections.Counter()
    for pt, _ in profiles.values():
        for (il, _n), b in pt.items():
            pooled_layer[il] += b
    tot = sum(pooled_layer.values())
    out["layer_share"] = {il: pooled_layer[il] / tot for il in sorted(pooled_layer)}

    for cap_mib in caps:
        cap = cap_mib * MIB
        row = {}
        addr, used = pin_tensors(tens, cap, lambda k, a_: a_)
        row["address"] = dict(pinned_mib=used / MIB,
                              covered={n: covered_by_tensors(addr, profiles[n][0]) for n in names},
                              pageable_layers=sorted({k[0] for k in tens if k not in addr}),
                              partial=sorted(f"{k[0]}:{k[1]}" for k in tens if k not in addr and
                                             any((k[0], t) in addr for t in TENSORS)))
        for kind in ("layer", "expert"):
            oracle, held = {}, {}
            runs = []
            for test in names:
                for mode, train in (("oracle", [test]), ("held", [n for n in names if n != test])):
                    pt = collections.Counter()
                    pe = collections.Counter()
                    for n in train:
                        pt.update(profiles[n][0])
                        pe.update(profiles[n][1])
                    if kind == "layer":
                        dens = {k: pt[k] / tens[k][1] for k in tens}
                        pin, used = pin_tensors(tens, cap, lambda k, a_: (-dens[k], a_))
                        cov = covered_by_tensors(pin, profiles[test][0])
                    else:
                        score = {}
                        for (il, e), b in pe.items():
                            per = sum(tens[(il, t)][1] // tens[(il, t)][2] for t in TENSORS)
                            score[(il, e)] = b / per
                        pin, used = pin_experts(tens, cap, score)
                        te = profiles[test][1]
                        tt = sum(te.values())
                        cov = sum(te[k] for k in pin) / tt
                        if mode == "held":
                            runs.append(runs_of(pin, tens))
                    (oracle if mode == "oracle" else held)[test] = cov
            row[kind] = dict(pinned_mib=used / MIB, covered_oracle=oracle, covered_held=held)
            if runs:
                row[kind]["registrations"] = max(runs)
        # selection counts instead of misses: what a LLAMA_MOE_CACHE_TRACE file
        # gives without a cache size, all tokens of the training traces
        held = {}
        for test in names:
            sel = collections.Counter()
            for n in names:
                if n != test:
                    sel.update((il, e) for _, il, e, _m in events[n])
            pin, used = pin_experts(tens, cap, sel)
            te = profiles[test][1]
            held[test] = sum(te[k] for k in pin) / sum(te.values())
        row["select"] = dict(pinned_mib=used / MIB, covered_held=held)

        # online: selections of the first N tokens of the same run, scored on
        # the misses of tokens >= 128 only (address order on the same window too)
        late = {}
        for test in names:
            m = collections.Counter()
            for t, il, e, miss in events[test]:
                if t >= 128 and miss:
                    m[(il, e)] += sum(tens[(il, x)][1] // tens[(il, x)][2] for x in TENSORS)
            late[test] = m
        on = {}
        for n_tok in (8, 32, 64, 128):
            cov = {}
            for test in names:
                sel = collections.Counter((il, e) for t, il, e, _m in events[test] if t < n_tok)
                pin, _ = pin_experts(tens, cap, sel)
                cov[test] = sum(late[test][k] for k in pin) / sum(late[test].values())
            on[n_tok] = cov
        late_addr = {}
        for test in names:
            per_t = collections.Counter()
            for (il, e), b in late[test].items():
                for x in TENSORS:
                    per_t[(il, x)] += b * (tens[(il, x)][1] // tens[(il, x)][2]) // sum(
                        tens[(il, y)][1] // tens[(il, y)][2] for y in TENSORS)
            late_addr[test] = covered_by_tensors(addr, per_t)
        row["online"] = dict(window="tokens >= 128", covered=on, address_same_window=late_addr)
        out["caps"][cap_mib] = row

        def fmt(d):
            v = list(d.values())
            return f"{100*min(v):5.1f} to {100*max(v):5.1f}%"
        print(f"cap {cap_mib:6d} MiB: pageable share of warm upload bytes", file=sys.stderr)
        print(f"   address {fmt({k: 1 - v for k, v in row['address']['covered'].items()})}"
              f"   pageable layers {row['address']['pageable_layers']} partial {row['address']['partial']}",
              file=sys.stderr)
        for kind in ("layer", "expert"):
            r = row[kind]
            print(f"   {kind:7s} oracle {fmt({k: 1 - v for k, v in r['covered_oracle'].items()})}"
                  f"   held out {fmt({k: 1 - v for k, v in r['covered_held'].items()})}"
                  + (f"   registrations {r['registrations']}" if 'registrations' in r else ""),
                  file=sys.stderr)
        print(f"   select  held out {fmt({k: 1 - v for k, v in row['select']['covered_held'].items()})}",
              file=sys.stderr)
        print(f"   online (tokens >= 128): address {fmt({k: 1 - v for k, v in row['online']['address_same_window'].items()})}"
              + "".join(f"  first {n}: {fmt({k: 1 - v for k, v in c.items()})}"
                        for n, c in row['online']['covered'].items()), file=sys.stderr)

    if a.json:
        with open(a.json, "w", newline="\n") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
