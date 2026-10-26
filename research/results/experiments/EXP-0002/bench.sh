#!/usr/bin/env bash
# EXP-0002: BENCHMARKS.md B1 on hardware for three eviction arms.
#
#   ./research/results/experiments/EXP-0002/bench.sh [model.gguf]
#
# Arms:
#   sample0  LLAMA_MOE_CACHE_SAMPLE=0  upstream LRU
#   dist0    LLAMA_MOE_CACHE_DIST=0    EXP-0000 policy, victim by use count only
#   default  (none)                    this experiment, use count weighted by cyclic distance
#
# Above the crossover all three are the same code path, which is the control.
# One process at a time: each maps the whole 20.8 GiB model.

set -euo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

MODEL=${1:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/experiments/EXP-0002
BIN=./build/bin
SIZES="256 384 528 586 645 1024 2048"

mkdir -p "$OUT"
: > "$OUT/b1_raw.txt"
echo "mib,arm,hits,misses,hit_rate,uploaded_mib" > "$OUT/b1.csv"

{
    echo "commit  : $(git rev-parse --short HEAD)"
    echo "model   : $MODEL"
    echo "free RAM: $(free -g 2>/dev/null | awk '/Mem:/{print $7" GiB"}' || echo 'see run notes')"
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
    # llama_moe_cache: ubatch <= 8: hits = N, misses = N, hit rate = X%, uploaded = Y MiB
    echo "$line" | sed -E "s/.*hits = ([0-9]+), misses = ([0-9]+), hit rate = ([0-9.]+)%, uploaded = ([0-9.]+) MiB.*/$mib,$arm,\1,\2,\3,\4/" >> "$OUT/b1.csv"
}

for mib in $SIZES; do
    echo "== $mib MiB =="
    run "$mib" sample0  LLAMA_MOE_CACHE_SAMPLE=0
    run "$mib" dist0    LLAMA_MOE_CACHE_DIST=0
    run "$mib" default  LLAMA_MOE_CACHE_NONE=1
done

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/b1_info.txt"
echo "done, results in $OUT/b1.csv"
