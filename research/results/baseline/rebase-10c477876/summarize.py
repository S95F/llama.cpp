"""Summarize bench.jsonl: per-run reps, per-arm means, and paired ABBA/BAAB blocks.

    python research/results/baseline/rebase-10c477876/summarize.py
"""
import json
import statistics as st
from pathlib import Path

DIR = Path(__file__).parent
rows = [json.loads(l) for l in (DIR / "bench.jsonl").read_text().splitlines() if l.strip()]
rows.sort(key=lambda r: r["seq"])

print("seq arm       build      mean_ts  reps (tok/s)")
for r in rows:
    reps = " ".join(f"{x:6.2f}" for x in r["samples_ts"])
    print(f"{r['seq']:>3} {r['arm']:<9} {r['build_commit']:<10} {r['avg_ts']:7.2f}  {reps}")

arms = {}
for r in rows:
    arms.setdefault(r["arm"], []).append(r)

print()
for arm, rs in arms.items():
    means = [r["avg_ts"] for r in rs]
    reps = [x for r in rs for x in r["samples_ts"]]
    print(f"{arm:<9} runs {len(rs)}  mean of run means {st.mean(means):6.2f}"
          f"  run sd {st.stdev(means):5.2f}  reps {len(reps)} rep sd {st.stdev(reps):5.2f}")

# blocks of four in run order (ABBA, BAAB): merged minus pre-merge within each block
print()
diffs = []
for i in range(0, len(rows), 4):
    blk = rows[i:i + 4]
    a = st.mean(r["avg_ts"] for r in blk if r["arm"] == "merged")
    b = st.mean(r["avg_ts"] for r in blk if r["arm"] == "premerge")
    diffs.append((a - b) / b * 100)
    print(f"block {i // 4 + 1} ({''.join('A' if r['arm'] == 'merged' else 'B' for r in blk)}):"
          f" merged {a:6.2f}  pre-merge {b:6.2f}  change {diffs[-1]:+5.1f}%")

# per rep index, all runs (the reps are not exchangeable, BENCHMARKS.md B2)
print()
for k in range(len(rows[0]["samples_ts"])):
    a = st.mean(r["samples_ts"][k] for r in rows if r["arm"] == "merged")
    b = st.mean(r["samples_ts"][k] for r in rows if r["arm"] == "premerge")
    print(f"rep {k + 1}: merged {a:6.2f}  pre-merge {b:6.2f}  change {(a - b) / b * 100:+5.1f}%")

print()
a = st.mean(r["avg_ts"] for r in rows if r["arm"] == "merged")
b = st.mean(r["avg_ts"] for r in rows if r["arm"] == "premerge")
print(f"all: merged {a:6.2f}  pre-merge {b:6.2f}  change {(a - b) / b * 100:+5.1f}%"
      f"  (blocks {min(diffs):+.1f}% to {max(diffs):+.1f}%)")
