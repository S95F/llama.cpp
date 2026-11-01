#!/usr/bin/env bash
# Decode across the merge with B2's own protocol (-p 0 -n 64 -r 4) at 2048 MiB.
# bench_r2.sh ran pp512 first in the same process, which is not B2's protocol.
#
#   ./research/results/baseline/rebase-2ca33f0d7/bench_r3.sh [model.gguf]
#
# Merged (A) and pre-merge (B) builds, ABBA then BAAB. The md table goes to
# bench_r3.md, the per-rep samples (jsonl) to bench_r3.jsonl.

set -uo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

MODEL=${1:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
DIR=research/results/baseline/rebase-2ca33f0d7
OUT=$DIR/bench_r3.md
JL=$DIR/bench_r3.jsonl
A=./build/bin
B=C:/nsys/premerge_a88e64081/bin

avail() {
    powershell -NoProfile -Command '[math]::Round((Get-Counter "\Memory\Available MBytes").CounterSamples[0].CookedValue)'
}

gpu() {
    nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader
}

game() {
    tasklist //FI "IMAGENAME eq VIractal.exe" //NH | grep -i viractal || echo "no VIractal.exe"
}

: > "$OUT"
: > "$JL"
echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT"
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
