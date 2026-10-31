#!/usr/bin/env bash
# EXP-0004: Nsight Systems timelines of generation, to measure whether the
# expert uploads overlap GPU compute.
#
#   ./research/results/experiments/EXP-0004/bench_timeline.sh [arm ...]
#
# Each arm is <cache MiB>-<pageable|pinned>. Every arm runs three times in an
# ABA order across the list (traced, untraced, traced), so the untraced run
# gives the tracing overhead and the two traced runs give the repeatability.
# The .nsys-rep and .sqlite files are large and stay outside the repository in
# $CAP; the per-token JSON from analyze_timeline.py is what gets committed.

set -euo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
MODEL=${MODEL:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/experiments/EXP-0004
CAP=${CAP:-C:/nsys/exp0004}
BIN=C:/code/llama.cpp-fork/build/bin/llama-completion.exe
PY=/c/Python313/python
ARMS=${*:-"256-pageable 586-pageable 1024-pageable 2048-pageable 1024-pinned 2048-pinned"}

mkdir -p "$OUT" "$CAP"

ARGS=(-m "$MODEL" -ngl 99 -ncmoe 40 -c 2048 -n 128 -f moe-bench/prompt.txt --temp 0 --seed 1 -t 8 --no-warmup -no-cnv -lv 4)

free_gib() {
    powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)"
}

arm_env() {
    case $1 in
        *-pinned)   echo "GGML_CUDA_REGISTER_HOST=1" ;;
        *-pageable) echo "GGML_CUDA_REGISTER_HOST=0" ;;
        *) echo "bad arm $1" >&2; exit 1 ;;
    esac
}

traced() {
    local arm=$1 rep=$2 mib=${1%%-*}
    local tag="${arm}_t${rep}"
    echo "== traced $tag, free RAM $(free_gib) GiB"
    env $(arm_env "$arm") "$NSYS" profile --trace=cuda-sw,nvtx --sample=none --cpuctxsw=none \
        --cuda-memory-usage=false --cuda-graph-trace=graph:host-only --gpu-metrics-devices=none \
        --force-overwrite=true --export=sqlite --output="$CAP/$tag" \
        "$BIN" "${ARGS[@]}" --moe-cache-mib "$mib" < /dev/null > "$CAP/nsys_$tag.txt" 2>&1
    "$PY" -I "$OUT/analyze_timeline.py" "$CAP/$tag.sqlite" --out "$OUT/timeline_$tag.json" --log-out "$OUT/log_$tag.txt" | tee "$OUT/summary_$tag.txt"
}

untraced() {
    local arm=$1 mib=${1%%-*}
    local tag="${arm}_u"
    echo "== untraced $tag, free RAM $(free_gib) GiB"
    env $(arm_env "$arm") "$BIN" "${ARGS[@]}" --moe-cache-mib "$mib" < /dev/null > "$OUT/log_$tag.txt" 2>&1
    grep -E "eval time|ubatch <= 8|registered" "$OUT/log_$tag.txt" | tee "$OUT/summary_$tag.txt"
}

{
    echo "commit  : $(git rev-parse --short HEAD)"
    echo "model   : $MODEL"
    echo "nsys    : $("$NSYS" --version)"
    echo "arms    : $ARMS"
    echo "started : $(date -u +%Y-%m-%dT%H:%M:%SZ), free RAM $(free_gib) GiB"
} | tee -a "$OUT/info.txt"

for arm in $ARMS; do traced "$arm" 1; done
for arm in $ARMS; do untraced "$arm"; done
for arm in $ARMS; do traced "$arm" 2; done

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ), free RAM $(free_gib) GiB" | tee -a "$OUT/info.txt"
