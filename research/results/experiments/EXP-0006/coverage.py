#!/usr/bin/env python3
"""EXP-0006: the pageable share of the expert upload bytes each pinning order
leaves, computed offline on the token windows the timed runs measure.

The run traces are 1024-token LLAMA_MOE_CACHE_TRACE files taken without
pinning (routing and the cache do not depend on it). They are replayed
through EXP-0005's simulator at the runs' cache size. Replay token 0 is the
prompt's last-layer call (the only one output pruning leaves small), and
replay token t is decode pass t, which is timeline token t. Orders, at each cap:
  address   whole tensors in address order (the default)
  select    experts by selection count in the profile traces (LLAMA_MOE_PIN_PROFILE)
  online N  experts by selection count in the small batches before decode pass
            N + 1 (LLAMA_MOE_PIN_ONLINE=N); those passes are pageable in full

The pre-run prediction (coverage_1024.json) came from a first version that
took replay token t for timeline token t + 1, one pass early.

  python coverage.py <model.gguf> <layout.json> --runs a=<trace> ood=<trace>
                        --profile <trace>... --json out.json
"""

import argparse
import collections
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("e5cov", os.path.join(HERE, "..", "EXP-0005", "coverage.py"))
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)

MIB = e5.MIB
WINDOWS = [(387, 1022), (3, 1022), (3, 130)]
ONLINE = (64, 128)


def selections(events, upto=None):
    """Selection counts per (layer, expert), of replay tokens <= upto if given."""
    return collections.Counter((il, e) for t, il, e, _m in events if upto is None or t <= upto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("layout")
    ap.add_argument("--runs", nargs="+", required=True, help="name=trace.csv")
    ap.add_argument("--profile", nargs="+", required=True)
    ap.add_argument("--cache-mib", type=int, default=2048)
    ap.add_argument("--caps", default="7964,15000")
    ap.add_argument("--json")
    a = ap.parse_args()

    layout = json.load(open(a.layout))
    tens = e5.expert_tensors(a.model)
    per_expert = {il: sum(tens[(il, x)][1] // tens[(il, x)][2] for x in e5.TENSORS) for il in {k[0] for k in tens}}

    prof = collections.Counter()
    for p in a.profile:
        _f, ev, _c = e5.replay(e5.load_trace(p), layout, a.cache_mib, 0)
        prof.update(selections(ev))

    runs, checks = {}, {}
    for spec_ in a.runs:
        name, path = spec_.split("=", 1)
        _f, ev, chk = e5.replay(e5.load_trace(path), layout, a.cache_mib, 0)
        runs[name] = ev
        checks[name] = chk
        print(f"{name}: {chk['tokens']} tokens, {chk['calls']} calls, simulated fills {chk['sim_fills']}, "
              f"recorded {chk['rec_fills']}", file=sys.stderr)

    out = dict(cache_mib=a.cache_mib, checks=checks, token="timeline token = replay token = decode pass", caps={})
    for cap_mib in (int(c) for c in a.caps.split(",")):
        cap = cap_mib * MIB
        addr, addr_used = e5.pin_tensors(tens, cap, lambda k, a_: a_)
        sel_pin, sel_used = e5.pin_experts(tens, cap, prof)
        row = dict(address_mib=addr_used / MIB, select_mib=sel_used / MIB, runs={})
        for name, ev in runs.items():
            on_pin = {n: e5.pin_experts(tens, cap, selections(ev, n))[0] for n in ONLINE}
            r = {}
            for lo, hi in WINDOWS:
                fills = collections.Counter((il, e) for t, il, e, miss in ev if miss and lo <= t <= hi)
                pt, pe = e5.miss_bytes(fills, tens)
                tot = sum(pe.values())
                if not tot:
                    continue
                w = dict(miss_mib_per_token=tot / MIB / (hi - lo + 1),
                         address=1 - e5.covered_by_tensors(addr, pt),
                         select=1 - sum(pe[k] for k in sel_pin) / tot)
                for n in ONLINE:
                    # passes 1 to n run before the registration, so all of their misses are pageable
                    pinned = sum(per_expert[il] for t, il, e, miss in ev
                                 if miss and lo <= t <= hi and t > n and (il, e) in on_pin[n])
                    w[f"online{n}"] = 1 - pinned / tot
                r[f"{lo}-{hi}"] = w
            row["runs"][name] = r
        out["caps"][str(cap_mib)] = row
        print(f"cap {cap_mib} MiB (address pins {addr_used / MIB:.0f}, select {sel_used / MIB:.0f}): pageable share of upload bytes",
              file=sys.stderr)
        for name, r in row["runs"].items():
            for wk, w in r.items():
                print(f"   {name:4s} {wk:9s} {w['miss_mib_per_token']:6.1f} MiB/token  address {100 * w['address']:5.1f}%"
                      f"  select {100 * w['select']:5.1f}%" + "".join(f"  online{n} {100 * w[f'online{n}']:5.1f}%" for n in ONLINE),
                      file=sys.stderr)

    if a.json:
        with open(a.json, "w", newline="\n") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
