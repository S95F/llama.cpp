#!/usr/bin/env bash
# Across merge 10c477876: the buffer sizes of one 2048 MiB run, then decode with B2's
# own protocol (-p 0 -n 64 -r 4) at 2048 MiB, merged (A) against pre-merge 980791788 (B),
# ABBA then BAAB. Same form as rebase-2ca33f0d7/bench_r3.sh.
#
#   ./research/results/baseline/rebase-10c477876/bench.sh [model.gguf]
#
# Pinning is set explicitly (the user environment sets GGML_CUDA_REGISTER_HOST=1, and
# the earlier merge checks inherited it): both arms pin the default cap in address order.

set -uo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"
export GGML_CUDA_REGISTER_HOST=1

MODEL=${1:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
DIR=research/results/baseline/rebase-10c477876
CAP=C:/nsys/rebase_10c477876
OUT=$DIR/bench.md
JL=$DIR/bench.jsonl
A=./build/bin
B=C:/nsys/premerge_980791788/build/bin

avail() {
    powershell -NoProfile -Command '[math]::Round((Get-Counter "\Memory\Available MBytes").CounterSamples[0].CookedValue)'
}

gpu() {
    nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader
}

game() {
    tasklist //FI "IMAGENAME eq VIractal.exe" //NH | grep -i viractal || echo "no VIractal.exe"
}

if tasklist //NH | grep -qiE "llama-(completion|bench)"; then
    echo "another llama process is running, stopping" >&2
    exit 1
fi

# buffer sizes, the B1 2048 MiB run on the pre-merge build (the merged one is check.sh's)
GGML_CUDA_REGISTER_HOST=0 "$B/llama-completion" -m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib 2048 \
    -c 2048 -n 256 -f moe-bench/prompt.txt --temp 0 --seed 1 \
    -t 8 --no-warmup -no-cnv -v < /dev/null > "$CAP/b1_2048_premerge.out" 2> "$CAP/b1_2048_premerge.log"
pat='buffer size|MoE cache size|ubatch <= 8|large batches|graph splits'
{
    echo "== pre-merge 980791788 =="
    grep -E "$pat" "$CAP/b1_2048_premerge.log"
    echo "== merged =="
    grep -E "$pat" "$CAP/b1_2048_default.log"
    echo "== stdout =="
    cmp -s "$CAP/b1_2048_premerge.out" "$CAP/b1_2048_default.out" && echo "  identical" || echo "  DIFFER"
} > "$DIR/buffers_across_merge.txt"

: > "$OUT"
: > "$JL"
echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ), GGML_CUDA_REGISTER_HOST=$GGML_CUDA_REGISTER_HOST" | tee -a "$OUT"
n=0
bench() {
    local tag=$1 bindir=$2 err
    n=$((n + 1))
    err=$(mktemp)
    echo "== $n $tag ($bindir) avail $(avail) MiB, gpu idle: $(gpu), $(game)" | tee -a "$OUT"
    "$bindir/llama-bench" -m "$MODEL" -ngl 99 -ncmoe 40 -mcmoe 2048 -p 0 -n 64 -r 4 -t 8 \
        -o md -oe jsonl 2> "$err" | tee -a "$OUT"
    grep '^{' "$err" | sed "s/^{/{\"arm\": \"$tag\", \"seq\": $n, /" >> "$JL"
    rm -f "$err"
}
for tag in A B B A B A A B; do
    if [ "$tag" = A ]; then bench merged "$A"; else bench premerge "$B"; fi
done
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ), gpu idle: $(gpu)" | tee -a "$OUT"
