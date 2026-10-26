#!/usr/bin/env bash
# EXP-0002: BENCHMARKS.md B2, order-balanced, for the distance-weighted victim.
#
#   ./research/results/experiments/EXP-0002/bench_tg.sh [model.gguf]
#
# 528 MiB is below the crossover, where the change is active.
# 1024 MiB is the control: B1 proved the arms byte-identical there, so that row
# measures nothing but noise and sets the floor for reading the 528 row.
#
# Order-balanced over two passes because the second arm inherits a warmer page
# cache; the mean of the two passes is what may be quoted.

set -euo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

MODEL=${1:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/experiments/EXP-0002
BIN=./build/bin
SIZES=528,1024

mkdir -p "$OUT"
: > "$OUT/b2_raw.md"

echo "free RAM before: $(wmic OS get FreePhysicalMemory /value 2>/dev/null | tr -d '\r' | grep -o '[0-9]*' | awk '{printf "%.2f GiB", $1/1048576}')" | tee "$OUT/b2_info.txt"
echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/b2_info.txt"

arm() {
    local name=$1; shift
    echo "### pass $PASS arm $name" | tee -a "$OUT/b2_raw.md"
    env "$@" "$BIN/llama-bench" -m "$MODEL" -ngl 99 -ncmoe 40 -mcmoe "$SIZES" \
        -n 64 -p 0 -r 4 -t 8 -o md 2>&1 | tee -a "$OUT/b2_raw.md"
}

PASS=1
arm dist0   LLAMA_MOE_CACHE_DIST=0
arm default LLAMA_MOE_CACHE_NONE=1

PASS=2
arm default LLAMA_MOE_CACHE_NONE=1
arm dist0   LLAMA_MOE_CACHE_DIST=0

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/b2_info.txt"
echo "done, results in $OUT/b2_raw.md"
