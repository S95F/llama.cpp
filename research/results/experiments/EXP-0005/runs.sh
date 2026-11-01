#!/usr/bin/env bash
# EXP-0005: pinning arms at 2048 MiB, every run from the same page-cache state.
#
#   ./research/results/experiments/EXP-0005/runs.sh [<arm>_r<rep> ...]
#
# Before each run evict.py pushes the model out of the page cache (no admin
# rights, so by memory pressure, not by emptying the standby list) and
# probe.py records how much of it is left. Each run is a 1024-token Nsight
# Systems timeline (BENCHMARKS.md B7) with the paging counters alongside and a
# watchdog that stops it if the desktop runs out of memory.
#
# Arms:
#   off        pageable experts
#   addr7842   default pinning: RAM/4 (7842 MiB here), whole tensors in address order
#   addr15000  LLAMA_MOE_PIN_MIB=15000, address order
#   prof7842   default cap, experts ordered by LLAMA_MOE_PIN_PROFILE
#   prof15000  15000 MiB, experts ordered by the profile
#   none       --load-mode none (ROADMAP R5): the experts in a pinned host buffer
#
# The profile is the selection traces of prompts b, c and d (EXP-0001); the runs
# generate from prompt a (moe-bench/prompt.txt), so the profile is held out.

set -euo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
MODEL=${MODEL:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/experiments/EXP-0005
CAP=${CAP:-C:/nsys/exp0005}
BIN=C:/code/llama.cpp-fork/build/bin/llama-completion.exe
PY=/c/Python313/python
TL=research/results/experiments/EXP-0004/analyze_timeline.py
PROFILE=$CAP/profile_bcd.csv
RUNS=${*:-"off_r1 addr7842_r1 prof7842_r1 addr15000_r1 prof15000_r1 none_r1 none_r2 prof15000_r2 addr15000_r2 prof7842_r2 addr7842_r2 off_r2"}

ARGS=(-m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib 2048 -c 2048 -n 1024 -f moe-bench/prompt.txt --temp 0 --seed 1 -t 8 --no-warmup -no-cnv -lv 4)

mkdir -p "$CAP"
E1=research/results/experiments/EXP-0001
cat "$E1/trace_prompt_b.csv" "$E1/trace_prompt_c.csv" "$E1/trace_prompt_d.csv" > "$PROFILE"

arm_env() {
    case $1 in
        off)       echo "GGML_CUDA_REGISTER_HOST=0" ;;
        addr7842)  echo "GGML_CUDA_REGISTER_HOST=1" ;;
        addr15000) echo "GGML_CUDA_REGISTER_HOST=1 LLAMA_MOE_PIN_MIB=15000" ;;
        prof7842)  echo "GGML_CUDA_REGISTER_HOST=1 LLAMA_MOE_PIN_PROFILE=$PROFILE" ;;
        prof15000) echo "GGML_CUDA_REGISTER_HOST=1 LLAMA_MOE_PIN_MIB=15000 LLAMA_MOE_PIN_PROFILE=$PROFILE" ;;
        none)      echo "GGML_CUDA_REGISTER_HOST=0" ;;
        *) echo "bad arm $1" >&2; exit 1 ;;
    esac
}

arm_args() {
    case $1 in
        none) echo "--load-mode none" ;;
        *) echo "" ;;
    esac
}

{
    echo "commit  : $(git rev-parse --short HEAD) (+ working tree: $(git status --short src | tr '\n' ' '))"
    echo "model   : $MODEL"
    echo "nsys    : $("$NSYS" --version)"
    echo "runs    : $RUNS"
    echo "started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "$OUT/info.txt"

for tag in $RUNS; do
    arm=${tag%_r*}
    env_=$(arm_env "$arm")
    echo "== $tag: $env_ $(arm_args "$arm")" | tee -a "$OUT/info.txt"

    "$PY" "$OUT/evict.py" | tee "$OUT/pre_$tag.txt"
    "$PY" "$OUT/probe.py" "$MODEL" --seed "$(echo "$tag" | cksum | cut -d" " -f1)" | tee -a "$OUT/pre_$tag.txt"
    cat "$OUT/pre_$tag.txt" >> "$OUT/info.txt"

    typeperf "\\Memory\\Pages Input/sec" "\\Memory\\Page Reads/sec" "\\Memory\\Transition Faults/sec" \
        "\\Memory\\Page Faults/sec" "\\Memory\\Available MBytes" "\\PhysicalDisk(_Total)\\Disk Read Bytes/sec" \
        -si 1 -y -o "$(cygpath -w "$OUT")\\paging_$tag.csv" > /dev/null 2>&1 &
    "$PY" "$OUT/watchdog.py" --out "$OUT/low_$tag.txt" &
    wd=$!
    echo "start_unix $(date +%s.%N)" > "$OUT/wall_$tag.txt"
    # shellcheck disable=SC2046
    env $env_ "$NSYS" profile --trace=cuda-sw,nvtx --sample=none --cpuctxsw=none \
        --cuda-memory-usage=false --cuda-graph-trace=graph:host-only --gpu-metrics-devices=none \
        --force-overwrite=true --export=sqlite --output="$CAP/$tag" \
        "$BIN" "${ARGS[@]}" $(arm_args "$arm") < /dev/null > "$CAP/nsys_$tag.txt" 2>&1 || echo "run failed: $tag" | tee -a "$OUT/info.txt"
    echo "end_unix $(date +%s.%N)" >> "$OUT/wall_$tag.txt"
    wait $wd || true
    taskkill //F //IM typeperf.exe > /dev/null 2>&1 || true
    cat "$OUT/low_$tag.txt" | tee -a "$OUT/info.txt"

    if [ -f "$CAP/$tag.sqlite" ]; then
        "$PY" "$TL" "$CAP/$tag.sqlite" --out "$OUT/timeline_$tag.json" --log-out "$OUT/log_$tag.txt" > "$OUT/summary_$tag.txt" 2>&1 \
            || echo "analysis failed: $tag" | tee -a "$OUT/info.txt"
        "$PY" "$OUT/summarize.py" "$tag" --cap "$CAP" | tail -1 || echo "summary failed: $tag"
    fi
done

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/info.txt"
"$PY" "$OUT/summarize.py" --table --cap "$CAP"
