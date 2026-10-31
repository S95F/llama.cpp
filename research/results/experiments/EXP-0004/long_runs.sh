#!/usr/bin/env bash
# EXP-0004, second batch: 1024-token traced runs, to see the decomposition once
# the first-touch page faults of a fresh process have died away, with the OS
# paging counters sampled alongside so the slow copies can be tied to disk reads.
#
#   ./research/results/experiments/EXP-0004/long_runs.sh [arm ...]
#
# Then the standing rebase check (BENCHMARKS.md B1 at 2048 MiB).

set -euo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
MODEL=${MODEL:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/experiments/EXP-0004
CAP=${CAP:-C:/nsys/exp0004}
BIN=C:/code/llama.cpp-fork/build/bin/llama-completion.exe
PY=/c/Python313/python
ARMS=${*:-"2048-pageable 2048-pinned 586-pageable"}

ARGS=(-m "$MODEL" -ngl 99 -ncmoe 40 -c 2048 -n 1024 -f moe-bench/prompt.txt --temp 0 --seed 1 -t 8 --no-warmup -no-cnv -lv 4)

free_gib() {
    powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)"
}

for arm in $ARMS; do
    mib=${arm%%-*}
    case $arm in
        *-pinned) pin=1 ;;
        *) pin=0 ;;
    esac
    tag="${arm}_long"
    echo "== traced $tag, free RAM $(free_gib) GiB" | tee -a "$OUT/info.txt"

    # paging counters once a second, as wall-clock rows, until the run ends
    typeperf "\\Memory\\Pages Input/sec" "\\Memory\\Page Reads/sec" "\\Memory\\Transition Faults/sec" \
        "\\Memory\\Page Faults/sec" "\\PhysicalDisk(_Total)\\Disk Read Bytes/sec" \
        -si 1 -y -o "$(cygpath -w "$OUT")\\paging_$tag.csv" > /dev/null 2>&1 &
    tp=$!
    echo "start_unix $(date +%s.%N)" > "$OUT/wall_$tag.txt"
    env GGML_CUDA_REGISTER_HOST=$pin "$NSYS" profile --trace=cuda-sw,nvtx --sample=none --cpuctxsw=none \
        --cuda-memory-usage=false --cuda-graph-trace=graph:host-only --gpu-metrics-devices=none \
        --force-overwrite=true --export=sqlite --output="$CAP/$tag" \
        "$BIN" "${ARGS[@]}" --moe-cache-mib "$mib" < /dev/null > "$CAP/nsys_$tag.txt" 2>&1
    echo "end_unix $(date +%s.%N)" >> "$OUT/wall_$tag.txt"
    sleep 2
    taskkill //F //IM typeperf.exe > /dev/null 2>&1 || kill $tp 2> /dev/null || true
    "$PY" -I "$OUT/analyze_timeline.py" "$CAP/$tag.sqlite" --out "$OUT/timeline_$tag.json" --log-out "$OUT/log_$tag.txt" | tee "$OUT/summary_$tag.txt"
done

# standing rebase check, the EXP-0003 B1 command for the default arm at 2048 MiB
echo "== B1 2048 MiB rebase check, free RAM $(free_gib) GiB" | tee -a "$OUT/info.txt"
"$BIN" -m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib 2048 \
    -c 2048 -n 256 -f moe-bench/prompt.txt --temp 0 --seed 1 \
    -t 8 --no-warmup -no-cnv -v < /dev/null 2>&1 | grep -E "ubatch <= 8" | tail -1 | tee "$OUT/b1_2048_rebase_check.txt"
