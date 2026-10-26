#!/usr/bin/env bash
# EXP-0003: BENCHMARKS.md B1 for the sampled-slot count, on base a4699400e.
#
#   ./research/results/experiments/EXP-0003/bench.sh [model.gguf]
#
# Arms:
#   s16      LLAMA_MOE_CACHE_SAMPLE=16  the EXP-0002 policy
#   default  (none)                     this experiment, 32 sampled slots
#
# Both arms run in the same session, so the paired comparison is valid on this
# base whatever upstream did to the router.
#
# 1024 and 2048 MiB are controls: nothing samples at or above the crossover, so
# the arms must come out byte-identical there.  The 2048 row doubles as the
# base-change check - it should reproduce 61.38% / 58033.62 MiB exactly if the
# router is unchanged from d854d98b.

set -euo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

MODEL=${1:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/experiments/EXP-0003
BIN=./build/bin
SIZES="256 384 528 586 1024 2048"

mkdir -p "$OUT"
: > "$OUT/b1_raw.txt"
echo "mib,arm,hits,misses,hit_rate,uploaded_mib" > "$OUT/b1.csv"

{
    echo "commit  : $(git rev-parse --short HEAD) plus the uncommitted n_sample = 32"
    echo "base    : $(git rev-parse --short a4699400e)"
    echo "model   : $MODEL"
    echo "started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$OUT/b1_info.txt"

run() {
    local mib=$1 arm=$2
    shift 2
    local line
    line=$(env "$@" "$BIN/llama-completion" \
        -m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib "$mib" \
        -c 2048 -n 256 -f moe-bench/prompt.txt --temp 0 --seed 1 \
        -t 8 --no-warmup -no-cnv -v < /dev/null 2>&1 \
        | grep -E "ubatch <= 8" | tail -1)
    echo "mcmoe=$mib arm=$arm $line" | tee -a "$OUT/b1_raw.txt"
    echo "$line" | sed -E "s/.*hits = ([0-9]+), misses = ([0-9]+), hit rate = ([0-9.]+)%, uploaded = ([0-9.]+) MiB.*/$mib,$arm,\1,\2,\3,\4/" >> "$OUT/b1.csv"
}

for mib in $SIZES; do
    echo "== $mib MiB =="
    run "$mib" s16     LLAMA_MOE_CACHE_SAMPLE=16
    run "$mib" default LLAMA_MOE_CACHE_NONE=1
done

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/b1_info.txt"
echo "done, results in $OUT/b1.csv"
