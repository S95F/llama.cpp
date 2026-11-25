#!/usr/bin/env bash
# Checks after merging upstream cddaa915a into the branch (merge 10c477876, from 980791788).
#
#   ./research/results/baseline/rebase-10c477876/check.sh [model.gguf]
#
#   1. B1, the shipping policy and the plain LRU arm (LLAMA_MOE_CACHE_SAMPLE=0) at
#      every size. The router changed upstream, so the old 2048 MiB row
#      (61.38% / 58033.62 MiB) is not expected to reproduce; this re-records it.
#   2. B3 with the profile and online pinning arms, then every output diffed
#      against the pre-merge B3 outputs of EXP-0007 (same prompt, same flags).
#   3. test-backend-ops on MUL_MAT_ID.
# Decode timing across the merge is bench.sh.

set -uo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

MODEL=${1:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/baseline/rebase-10c477876
CAP=C:/nsys/rebase_10c477876
BIN=./build/bin
PRE_B3=C:/nsys/exp0007/correctness256
PROFILE=C:/nsys/exp0005/profile_bcd.csv

mkdir -p "$OUT" "$CAP"

if tasklist //FI "IMAGENAME eq llama-completion.exe" //NH | grep -qi llama-completion; then
    echo "another llama-completion is running, stopping" >&2
    exit 1
fi

{
    echo "merge   : $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
    echo "parents : $(git log -1 --format=%p 10c477876)"
    echo "model   : $MODEL"
    echo "avail   : $(powershell -NoProfile -Command '[math]::Round((Get-Counter "\Memory\Available MBytes").CounterSamples[0].CookedValue)') MiB"
    echo "game    : $(tasklist //FI "IMAGENAME eq VIractal.exe" //NH | grep -i viractal || echo "no VIractal.exe")"
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
    echo "$line" | sed -E "s/.*hits = ([0-9]+), misses = ([0-9]+).*hit rate = ([0-9.]+)%, uploaded = ([0-9.]+) MiB.*/$mib,$arm,\1,\2,\3,\4/" >> "$OUT/b1.csv"
}
for mib in 2048 1024 645 586 528 384 256; do
    b1 "$mib" default LLAMA_MOE_CACHE_NONE=1
    b1 "$mib" lru LLAMA_MOE_CACHE_SAMPLE=0
done

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

# 3. MUL_MAT_ID op tests
"$BIN/test-backend-ops" test -o MUL_MAT_ID -b CUDA0 > "$CAP/test_backend_ops_mmid.txt" 2>&1
echo "exit $?" >> "$CAP/test_backend_ops_mmid.txt"
grep -E "tests passed|FAIL|Backend CUDA0|exit" "$CAP/test_backend_ops_mmid.txt" | tail -8 | tee "$OUT/test_backend_ops_mmid.txt"

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/info.txt"
