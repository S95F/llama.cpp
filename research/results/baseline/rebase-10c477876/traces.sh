#!/usr/bin/env bash
# The router changed at merge 10c477876 (B1 at 2048 MiB no longer reproduces), so the
# EXP-0001 traces are stale for this base. Re-collect the four selection traces at
# 2048 MiB and re-validate the simulator against the cache's own counters (B6).
#
#   ./research/results/baseline/rebase-10c477876/traces.sh [model.gguf]
#
# One process at a time: each maps the whole 20.8 GiB model.

set -uo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

MODEL=${1:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/baseline/rebase-10c477876
CAP=C:/nsys/rebase_10c477876
BIN=./build/bin
LAYOUT=moe-bench/layout-qwen35moe-35b-a3b-ud-q4kxl.json

mkdir -p "$OUT" "$CAP"
: > "$OUT/telemetry_2048.txt"
: > "$OUT/sim_validate.txt"

for p in prompt prompt_b prompt_c prompt_d; do
    echo "== $p =="
    GGML_CUDA_REGISTER_HOST=0 LLAMA_MOE_CACHE_TRACE="$OUT/trace_$p.csv" "$BIN/llama-completion" \
        -m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib 2048 \
        -c 2048 -n 256 -f "moe-bench/$p.txt" --temp 0 --seed 1 \
        -t 8 --no-warmup -no-cnv -v \
        > "$OUT/gen_$p.txt" 2> "$CAP/log_$p.txt" < /dev/null
    line=$(grep -E "ubatch <= 8" "$CAP/log_$p.txt" | tail -1)
    echo "$p $line" | tee -a "$OUT/telemetry_2048.txt"
    python moe-bench/moe_sim.py "$OUT/trace_$p.csv" --layout "$LAYOUT" --sizes 2048 \
        --policies slru-ramp --json "$OUT/sim_validate_$p.json" > /dev/null
    python - "$OUT/sim_validate_$p.json" "$p" "$line" <<'PY' | tee -a "$OUT/sim_validate.txt"
import json, re, sys
r = json.load(open(sys.argv[1]))[0]
m = re.search(r"hits = (\d+), misses = (\d+).*uploaded = ([0-9.]+) MiB", sys.argv[3])
hw = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
sim = (r["hits"], r["fills"], round(r["uploaded_mib"], 2))
print(f"{sys.argv[2]:9s} hw {hw[0]} / {hw[1]} / {hw[2]:.2f}  sim {sim[0]} / {sim[1]} / {sim[2]:.2f}  "
      f"{'EXACT' if hw[:2] == sim[:2] and abs(hw[2] - sim[2]) < 0.01 else 'MISMATCH'}")
PY
done
