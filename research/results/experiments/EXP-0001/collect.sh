#!/usr/bin/env bash
# EXP-0001: re-collect the four expert selection traces on base d854d98b and
# capture the cache's own counters at 2048 MiB for the simulator validation.
#
#   ./research/results/experiments/EXP-0001/collect.sh [model.gguf]
#
# One process at a time: each maps the whole 20.8 GiB model.

set -euo pipefail

# build/ links the CUDA 12.6 runtime, which is not the one on the system PATH
export PATH="/c/Users/steve/cuda126/bin:$PATH"

MODEL=${1:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/experiments/EXP-0001
BIN=./build/bin
MIB=2048

mkdir -p "$OUT"
: > "$OUT/telemetry_2048.txt"

{
    echo "commit   : $(git rev-parse --short HEAD)"
    echo "base     : $(git rev-parse --short d854d98ba)"
    echo "model    : $MODEL"
    echo "cache    : $MIB MiB"
    echo "started  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$OUT/run_info.txt"

for p in prompt prompt_b prompt_c prompt_d; do
    echo "== $p =="
    free_before=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null || echo "n/a")
    LLAMA_MOE_CACHE_TRACE="$OUT/trace_$p.csv" "$BIN/llama-completion" \
        -m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib "$MIB" \
        -c 2048 -n 256 -f "moe-bench/$p.txt" --temp 0 --seed 1 \
        -t 8 --no-warmup -no-cnv -v \
        > "$OUT/gen_$p.txt" 2> "$OUT/log_$p.txt" < /dev/null

    echo "  gpu free before: $free_before"
    echo "  trace lines    : $(wc -l < "$OUT/trace_$p.csv")"
    {
        echo "### $p"
        grep -E "ubatch <= 8|ubatch > 8|uploaded|hit" "$OUT/log_$p.txt" | tail -8
    } | tee -a "$OUT/telemetry_2048.txt"
done

echo "finished : $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/run_info.txt"
echo "done, results in $OUT"
