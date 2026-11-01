#!/usr/bin/env bash
# Checks after merging upstream a91eab67d into the branch (merge 2ca33f0d7, from a88e64081).
#
#   ./research/results/baseline/rebase-2ca33f0d7/check.sh [model.gguf]
#
#   1. B1, the shipping policy at every size and upstream LRU at 528 and 2048 MiB.
#      2048 MiB must reproduce 61.38% / 58033.62 MiB (hits 50092, misses 31516),
#      the standing router check (KNOWLEDGE_BASE.md).
#   2. B3 with the profile and online pinning arms, then every output diffed
#      against the pre-merge B3 outputs of EXP-0006 (same prompt, same flags).
#   3. The buffer-size log lines of one 2048 MiB run, pre-merge build against merged.
#   4. test-backend-ops on MUL_MAT_ID (upstream's new MoE fusion cases).
#   5. A throughput sanity check, llama-bench pp512 + tg64 at 2048 MiB, merged and
#      pre-merge builds in ABBA order. Corroboration only (B2 noise floor 1.3 to 3.9%).

set -uo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

MODEL=${1:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/baseline/rebase-2ca33f0d7
CAP=C:/nsys/rebase_2ca33f0d7
BIN=./build/bin
OLD=C:/nsys/premerge_a88e64081/bin
PRE_B3=C:/nsys/exp0006/correctness256
PROFILE=C:/nsys/exp0005/profile_bcd.csv

mkdir -p "$OUT" "$CAP"

if tasklist //FI "IMAGENAME eq llama-completion.exe" //NH | grep -qi llama-completion; then
    echo "another llama-completion is running, stopping" >&2
    exit 1
fi

{
    echo "merge   : $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
    echo "parents : $(git log -1 --format=%p)"
    echo "model   : $MODEL"
    echo "avail   : $(powershell -NoProfile -Command '[math]::Round((Get-Counter "\Memory\Available MBytes").CounterSamples[0].CookedValue)') MiB"
    echo "started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$OUT/info.txt"

# 1. B1
echo "mib,arm,hits,misses,hit_rate,uploaded_mib" > "$OUT/b1.csv"
: > "$OUT/b1_raw.txt"
b1() {
    local mib=$1 arm=$2
    shift 2
    local log="$CAP/b1_${mib}_${arm}.log" line
    env "$@" "$BIN/llama-completion" -m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib "$mib" \
        -c 2048 -n 256 -f moe-bench/prompt.txt --temp 0 --seed 1 \
        -t 8 --no-warmup -no-cnv -v < /dev/null > "$CAP/b1_${mib}_${arm}.out" 2> "$log"
    line=$(grep -E "ubatch <= 8" "$log" | tail -1)
    echo "mcmoe=$mib arm=$arm $line" | tee -a "$OUT/b1_raw.txt"
    echo "$line" | sed -E "s/.*hits = ([0-9]+), misses = ([0-9]+), hit rate = ([0-9.]+)%, uploaded = ([0-9.]+) MiB.*/$mib,$arm,\1,\2,\3,\4/" >> "$OUT/b1.csv"
}
for mib in 2048 1024 586 528 384 256; do
    b1 "$mib" default LLAMA_MOE_CACHE_NONE=1
done
b1 528 lru LLAMA_MOE_CACHE_SAMPLE=0
b1 2048 lru LLAMA_MOE_CACHE_SAMPLE=0

# 2. B3, then across the merge
PIN_PROFILE="$PROFILE" NGEN=256 OUT="$CAP/correctness256" \
    ./moe-bench/check-correctness.sh "$MODEL" > "$OUT/b3_correctness_gpu256.txt" 2>&1
echo "exit $?" >> "$OUT/b3_correctness_gpu256.txt"
{
    echo "== merged B3 outputs against the pre-merge ones ($PRE_B3) =="
    for f in "$PRE_B3"/*.txt; do
        name=$(basename "$f")
        if [ ! -f "$CAP/correctness256/$name" ]; then
            echo "  MISSING $name"
        elif cmp -s "$f" "$CAP/correctness256/$name"; then
            echo "  SAME    $name ($(wc -c < "$f") bytes)"
        else
            echo "  DIFFER  $name at $(cmp "$f" "$CAP/correctness256/$name" | sed -E 's/.*byte ([0-9]+).*/byte \1/')"
        fi
    done
} | tee "$OUT/b3_across_merge.txt"

# 3. buffer sizes, the same 2048 MiB run on the pre-merge build
"$OLD/llama-completion" -m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib 2048 \
    -c 2048 -n 256 -f moe-bench/prompt.txt --temp 0 --seed 1 \
    -t 8 --no-warmup -no-cnv -v < /dev/null > "$CAP/b1_2048_premerge.out" 2> "$CAP/b1_2048_premerge.log"
pat='buffer size|MoE cache size|ubatch <= 8|large batches|graph splits'
{
    echo "== pre-merge a88e64081 =="
    grep -E "$pat" "$CAP/b1_2048_premerge.log"
    echo "== merged =="
    grep -E "$pat" "$CAP/b1_2048_default.log"
    echo "== stdout =="
    cmp -s "$CAP/b1_2048_premerge.out" "$CAP/b1_2048_default.out" && echo "  identical" || echo "  DIFFER"
} | tee "$OUT/buffers_across_merge.txt"

# 4. MUL_MAT_ID op tests
"$BIN/test-backend-ops" test -o MUL_MAT_ID -b CUDA0 > "$CAP/test_backend_ops_mmid.txt" 2>&1
echo "exit $?" >> "$CAP/test_backend_ops_mmid.txt"
grep -E "tests passed|FAIL|Backend CUDA0|exit" "$CAP/test_backend_ops_mmid.txt" | tail -8 | tee "$OUT/test_backend_ops_mmid.txt"

# 5. throughput sanity, ABBA
: > "$OUT/bench_abba.md"
bench() {
    local tag=$1 bindir=$2
    echo "== $tag ($bindir) avail $(powershell -NoProfile -Command '[math]::Round((Get-Counter "\Memory\Available MBytes").CounterSamples[0].CookedValue)') MiB" | tee -a "$OUT/bench_abba.md"
    "$bindir/llama-bench" -m "$MODEL" -ngl 99 -ncmoe 40 -mcmoe 2048 -p 512 -n 64 -r 3 -t 8 -o md 2>/dev/null \
        | tee -a "$OUT/bench_abba.md"
}
bench merged_a "$BIN"
bench premerge_b "$OLD"
bench premerge_b2 "$OLD"
bench merged_a2 "$BIN"

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/info.txt"
