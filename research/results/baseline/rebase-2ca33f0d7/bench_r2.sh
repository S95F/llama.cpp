#!/usr/bin/env bash
# Re-run of check.sh step 5, whose timings a game contaminated (NOTE_game.txt).
#
#   ./research/results/baseline/rebase-2ca33f0d7/bench_r2.sh [model.gguf]
#
# B2 settings at 2048 MiB, pp512 and tg64, merged and pre-merge builds in ABBA
# order. The GPU state and any running game are logged before each arm.

set -uo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

MODEL=${1:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/baseline/rebase-2ca33f0d7/bench_abba_tg_r2.md
BIN=./build/bin
OLD=C:/nsys/premerge_a88e64081/bin

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
echo "started $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT"
bench() {
    local tag=$1 bindir=$2
    {
        echo "== $tag ($bindir) avail $(avail) MiB, gpu idle: $(gpu), $(game)"
    } | tee -a "$OUT"
    "$bindir/llama-bench" -m "$MODEL" -ngl 99 -ncmoe 40 -mcmoe 2048 -p 512 -n 64 -r 4 -t 8 -o md 2>/dev/null \
        | tee -a "$OUT"
}
bench merged_a "$BIN"
bench premerge_b "$OLD"
bench premerge_b2 "$OLD"
bench merged_a2 "$BIN"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ), gpu idle: $(gpu)" | tee -a "$OUT"
