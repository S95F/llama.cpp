#!/usr/bin/env bash
# EXP-0007: one batched prefetch per layer of the missed experts' pageable pages
# (ROADMAP R12), the fault attribution of Q18, and a second launch headroom (R16b).
#
#   ./research/results/experiments/EXP-0007/runs.sh [<arm>_r<rep> ...]
#
# As EXP-0006's runs.sh (evict, probe, balloon, monitor, typeperf, Nsight), plus:
#   - every run writes the selection trace (LLAMA_MOE_CACHE_TRACE) and the host
#     time of every upload and prefetch (LLAMA_MOE_COPY_LOG) to $CAP;
#   - the arm sets the balloon target: -lo arms hold $TARGET_LO MiB available;
#   - any run that goes below the monitor's 300 MiB floor or is killed stops the
#     session (EXP-0006's warm-up checked only for a kill);
#   - the cache's variables are cleared first: GGML_CUDA_REGISTER_HOST=1 is set in
#     the user's environment, so an arm that does not set it would pin anyway;
#   - a tag's earlier outputs are deleted before it runs, so a failed run cannot be
#     joined with a previous attempt's files.
#
# Arms, all with a 2048 MiB cache:
#   prof       GGML_CUDA_REGISTER_HOST=1, default cap, file profile order (as EXP-0006)
#   addr       GGML_CUDA_REGISTER_HOST=1, default cap, address order
#   pg         nothing pinned, 128 tokens (every expert copy pageable, cold)
#   <arm>-pf   the same with LLAMA_MOE_PREFETCH=1 (every pageable missed expert)
#   <arm>-pq   the same with LLAMA_MOE_PREFETCH=2 (those whose middle page is not in the working set)
#   <arm>-lo   the same at the lower launch headroom
# A tag is <arm>_r<rep>; <arm>_w is a warm-up run that the analysis ignores.

set -euo pipefail

unset GGML_CUDA_REGISTER_HOST LLAMA_MOE_PREFETCH LLAMA_MOE_PIN_PROFILE LLAMA_MOE_PIN_MIB LLAMA_MOE_PIN_ONLINE     LLAMA_MOE_CACHE_SAMPLE LLAMA_MOE_CACHE_DIST LLAMA_MOE_CACHE_SLRU_RAMP LLAMA_MOE_CACHE_LOG LLAMA_MOE_CACHE_TRACE LLAMA_MOE_COPY_LOG

export PATH="/c/Users/steve/cuda126/bin:$PATH"

NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
MODEL=${MODEL:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/experiments/EXP-0007
E5=research/results/experiments/EXP-0005
E6=research/results/experiments/EXP-0006
CAP=${CAP:-C:/nsys/exp0007}
BIN=C:/code/llama.cpp-fork/build/bin/llama-completion.exe
PY=/c/Python313/python
TL=research/results/experiments/EXP-0004/analyze_timeline.py
PROFILE=$CAP/profile_bcd.csv
TARGET_HI=${TARGET_HI:-19968}
TARGET_LO=${TARGET_LO:-18944}
FLOOR=300
# mirror order: each pair swaps its order between the rounds and every arm has the same mean position;
# prof-lo sits next to prof in both rounds; then the cold pageable pair, ABBA
RUNS=${*:-"prof-lo_w prof-lo_r1 prof_r1 prof-pf_r1 prof-pq_r1 addr-pq_r1 addr-pf_r1 addr_r1 addr_r2 addr-pf_r2 addr-pq_r2 prof-pq_r2 prof-pf_r2 prof_r2 prof-lo_r2 pg_r1 pg-pf_r1 pg-pf_r2 pg_r2"}

mkdir -p "$CAP"
# never leave a balloon holding memory behind
stop=""
trap '[ -n "$stop" ] && touch "$stop"' EXIT
E1=research/results/experiments/EXP-0001
cat "$E1/trace_prompt_b.csv" "$E1/trace_prompt_c.csv" "$E1/trace_prompt_d.csv" > "$PROFILE"

arm_env() {
    local a=${1%-lo} e=""
    case $a in
        *-pf) e="LLAMA_MOE_PREFETCH=1"; a=${a%-pf} ;;
        *-pq) e="LLAMA_MOE_PREFETCH=2"; a=${a%-pq} ;;
    esac
    case $a in
        prof) echo "$e GGML_CUDA_REGISTER_HOST=1 LLAMA_MOE_PIN_PROFILE=$PROFILE" ;;
        addr) echo "$e GGML_CUDA_REGISTER_HOST=1" ;;
        pg)   echo "$e GGML_CUDA_REGISTER_HOST=0" ;;
        *) echo "bad arm $1" >&2; exit 1 ;;
    esac
}

arm_target() {
    case $1 in
        *-lo) echo "$TARGET_LO" ;;
        *)    echo "$TARGET_HI" ;;
    esac
}

arm_ngen() {
    case $1 in
        pg*) echo 128 ;;
        *)   echo 1024 ;;
    esac
}

# the balloon line says "locked yes" and "available now N MiB" with N within 192 MiB of the target
balloon_ok() {
    awk -v t="$2" '{ for (i = 1; i < NF; i++) if ($i == "now") n = $(i + 1) } END { exit !(n != "" && n - t <= 192 && t - n <= 192 && $0 ~ /locked yes/) }' "$1"
}

{
    echo "commit  : $(git rev-parse --short HEAD) (+ working tree: $(git status --short src | tr '\n' ' '))"
    echo "model   : $MODEL"
    echo "nsys    : $("$NSYS" --version)"
    echo "targets : $TARGET_HI MiB available at launch, $TARGET_LO for -lo arms"
    echo "runs    : $RUNS"
    echo "env     : $(env | grep -E '^(LLAMA|GGML)' | tr '
' ' ')"
    echo "started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "$OUT/info.txt"

for tag in $RUNS; do
    arm=${tag%_*}
    env_="$(arm_env "$arm") LLAMA_MOE_CACHE_TRACE=$CAP/trace_$tag.csv LLAMA_MOE_COPY_LOG=$CAP/copies_$tag.csv"
    target=$(arm_target "$arm")
    ngen=$(arm_ngen "$arm")
    echo "== $tag: $env_ -n $ngen, target $target MiB" | tee -a "$OUT/info.txt"

    if tasklist //FI "IMAGENAME eq llama-completion.exe" //NH | grep -qi llama-completion; then
        echo "stopped before $tag: another llama-completion.exe is running" | tee -a "$OUT/info.txt"
        exit 1
    fi

    rm -f "$OUT/flag_$tag.txt" "$CAP/trace_$tag.csv" "$CAP/copies_$tag.csv" "$CAP/$tag.sqlite" "$CAP/$tag.nsys-rep"         "$CAP/out_$tag.txt" "$OUT/timeline_$tag.json" "$OUT/log_$tag.txt" "$OUT/summary_$tag.txt"
    for attempt in 1 2 3; do
        "$PY" "$E5/evict.py" | tee "$OUT/pre_$tag.txt"
        "$PY" "$E5/probe.py" "$MODEL" --seed "$(echo "$tag" | cksum | cut -d" " -f1)" | tee -a "$OUT/pre_$tag.txt"

        stop="$CAP/stop_$tag.flag"
        rm -f "$stop" "$OUT/balloon_$tag.txt"
        "$PY" "$E6/balloon.py" --target "$target" --out "$OUT/balloon_$tag.txt" --stop "$stop" &
        bl=$!
        for _ in $(seq 1 120); do
            [ -s "$OUT/balloon_$tag.txt" ] && break
            sleep 0.5
        done
        cat "$OUT/balloon_$tag.txt" | tee -a "$OUT/pre_$tag.txt"
        balloon_ok "$OUT/balloon_$tag.txt" "$target" && break
        echo "balloon off target (attempt $attempt): $(cat "$OUT/balloon_$tag.txt")" | tee -a "$OUT/info.txt"
        if [ "$attempt" = 3 ]; then
            echo "launch headroom off target: $(cat "$OUT/balloon_$tag.txt")" > "$OUT/flag_$tag.txt"
            break
        fi
        touch "$stop"
        wait $bl || true
    done
    cat "$OUT/pre_$tag.txt" >> "$OUT/info.txt"

    typeperf "\\Memory\\Pages Input/sec" "\\Memory\\Page Reads/sec" "\\Memory\\Transition Faults/sec" \
        "\\Memory\\Page Faults/sec" "\\Memory\\Available MBytes" "\\PhysicalDisk(_Total)\\Disk Read Bytes/sec" \
        "\\Memory\\Standby Cache Normal Priority Bytes" "\\Memory\\Standby Cache Reserve Bytes" \
        "\\Memory\\Standby Cache Core Bytes" "\\Memory\\Modified Page List Bytes" "\\Memory\\Free & Zero Page List Bytes" \
        "\\Process(_Total)\\Working Set" \
        -si 1 -y -o "$(cygpath -w "$OUT")\\paging_$tag.csv" > /dev/null 2>&1 &
    "$PY" "$E6/monitor.py" --min "$FLOOR" --csv "$OUT/proc_$tag.csv" --out "$OUT/low_$tag.txt" &
    wd=$!
    echo "start_unix $(date +%s.%N)" > "$OUT/wall_$tag.txt"
    # shellcheck disable=SC2046
    env $env_ "$NSYS" profile --trace=cuda-sw,nvtx --sample=none --cpuctxsw=none \
        --cuda-memory-usage=false --cuda-graph-trace=graph:host-only --gpu-metrics-devices=none \
        --force-overwrite=true --export=sqlite --output="$CAP/$tag" \
        "$BIN" -m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib 2048 -c 2048 -n "$ngen" --temp 0 --seed 1 -t 8 \
        --no-warmup -no-cnv -lv 4 -f moe-bench/prompt.txt < /dev/null > "$CAP/nsys_$tag.txt" 2>&1 || echo "run failed: $tag" | tee -a "$OUT/info.txt"
    echo "end_unix $(date +%s.%N)" >> "$OUT/wall_$tag.txt"
    wait $wd || true
    touch "$stop"
    wait $bl || true
    taskkill //F //IM typeperf.exe > /dev/null 2>&1 || true
    cat "$OUT/low_$tag.txt" | tee -a "$OUT/info.txt"

    if [ -f "$CAP/$tag.sqlite" ]; then
        "$PY" "$TL" "$CAP/$tag.sqlite" --out "$OUT/timeline_$tag.json" --log-out "$OUT/log_$tag.txt" > "$OUT/summary_$tag.txt" 2>&1 \
            || echo "analysis failed: $tag" | tee -a "$OUT/info.txt"
        "$PY" "$OUT/summarize.py" "$tag" --cap "$CAP" | tail -1 || echo "summary failed: $tag"
        "$PY" -c "import json,sys; o=json.load(open(sys.argv[1]))[sys.argv[2]].get('stdout',{}); print('stdout', o.get('chars'), 'chars, </think> at', o.get('think_end'), 'sha256', o.get('sha256'))" \
            "$OUT/runs.json" "$tag" | tee -a "$OUT/info.txt" || true
    fi
    # stop the session at a kill or below the floor, so the desktop is not starved again
    low=$(sed -n 's/^lowest available \([0-9]*\) MiB.*/\1/p' "$OUT/low_$tag.txt" 2> /dev/null)
    if grep -q KILLED "$OUT/low_$tag.txt" || [ "${low:-0}" -lt "$FLOOR" ]; then
        echo "stopped after $tag: $(cat "$OUT/low_$tag.txt")" | tee -a "$OUT/info.txt"
        exit 1
    fi
done

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/info.txt"
"$PY" "$OUT/summarize.py" --table --cap "$CAP"
