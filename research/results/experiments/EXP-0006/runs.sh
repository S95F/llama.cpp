#!/usr/bin/env bash
# EXP-0006: an online selection profile (ROADMAP R17) against the file profile
# and address order, with the available RAM held fixed at launch (R16).
#
#   ./research/results/experiments/EXP-0006/runs.sh [<arm>_r<rep> ...]
#
# As EXP-0005's runs.sh (evict, probe, typeperf, Nsight Systems), plus:
#   - balloon.py holds the available RAM at $TARGET MiB from just before the
#     launch to the end of the run, so every run starts with the same headroom;
#     a balloon that misses the target is retried after a new eviction, and a
#     run that still misses it is flagged in flag_<tag>.txt;
#   - monitor.py replaces the watchdog and also records the process's working
#     set, private bytes and page faults every second;
#   - typeperf also records the standby, modified and free page lists and the
#     working set of all processes;
#   - the session stops if another llama-completion.exe is running.
#
# Arms, all with GGML_CUDA_REGISTER_HOST=1 and a 2048 MiB cache:
#   addr        default cap (RAM/4), whole tensors in address order
#   prof        default cap, ordered by the file profile (LLAMA_MOE_PIN_PROFILE)
#   on64        default cap, LLAMA_MOE_PIN_ONLINE=64
#   on128       default cap, LLAMA_MOE_PIN_ONLINE=128
#   prof15k     LLAMA_MOE_PIN_MIB=15000, file profile
#   on64x15k    15000 MiB, online after 64
#   on128x15k   15000 MiB, online after 128
#   ood-<arm>   addr, prof, on128 at the default cap on prompt_ood.txt, with
#               --ignore-eos (the prompt never ran before, keep 1023 passes)
# A tag is <arm>_r<rep>; <arm>_w is a warm-up run that analyze.py ignores.
#
# The profile is the selection traces of prompts b, c and d (EXP-0001), as in
# EXP-0005. The runs generate from prompt a (moe-bench/prompt.txt), or from
# prompt_ood.txt, a children's story answered after an empty think block.

set -euo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
MODEL=${MODEL:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/experiments/EXP-0006
E5=research/results/experiments/EXP-0005
CAP=${CAP:-C:/nsys/exp0006}
BIN=C:/code/llama.cpp-fork/build/bin/llama-completion.exe
PY=/c/Python313/python
TL=research/results/experiments/EXP-0004/analyze_timeline.py
PROFILE=$CAP/profile_bcd.csv
TARGET=${TARGET:-19968}
# addr sits next to each online arm in both rounds (ABBA); a warm-up run first
RUNS=${*:-"on64x15k_w prof_r1 on64_r1 addr_r1 on128_r1 prof15k_r1 on64x15k_r1 on128x15k_r1 on128x15k_r2 on64x15k_r2 prof15k_r2 on128_r2 addr_r2 on64_r2 prof_r2 ood-addr_r1 ood-prof_r1 ood-on128_r1 ood-on128_r2 ood-prof_r2 ood-addr_r2"}

ARGS=(-m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib 2048 -c 2048 -n 1024 --temp 0 --seed 1 -t 8 --no-warmup -no-cnv -lv 4)

mkdir -p "$CAP"
# never leave a balloon holding memory behind
stop=""
trap '[ -n "$stop" ] && touch "$stop"' EXIT
E1=research/results/experiments/EXP-0001
cat "$E1/trace_prompt_b.csv" "$E1/trace_prompt_c.csv" "$E1/trace_prompt_d.csv" > "$PROFILE"

arm_env() {
    local e="GGML_CUDA_REGISTER_HOST=1"
    case ${1#ood-} in
        addr)      echo "$e" ;;
        prof)      echo "$e LLAMA_MOE_PIN_PROFILE=$PROFILE" ;;
        on64)      echo "$e LLAMA_MOE_PIN_ONLINE=64" ;;
        on128)     echo "$e LLAMA_MOE_PIN_ONLINE=128" ;;
        prof15k)   echo "$e LLAMA_MOE_PIN_MIB=15000 LLAMA_MOE_PIN_PROFILE=$PROFILE" ;;
        on64x15k)  echo "$e LLAMA_MOE_PIN_MIB=15000 LLAMA_MOE_PIN_ONLINE=64" ;;
        on128x15k) echo "$e LLAMA_MOE_PIN_MIB=15000 LLAMA_MOE_PIN_ONLINE=128" ;;
        *) echo "bad arm $1" >&2; exit 1 ;;
    esac
}

arm_prompt() {
    case $1 in
        ood-*) echo "$OUT/prompt_ood.txt" ;;
        *)     echo "moe-bench/prompt.txt" ;;
    esac
}

# the balloon line says "locked yes" and "available now N MiB" with N within 192 MiB of the target
balloon_ok() {
    awk -v t="$TARGET" '{ for (i = 1; i < NF; i++) if ($i == "now") n = $(i + 1) } END { exit !(n != "" && n - t <= 192 && t - n <= 192 && $0 ~ /locked yes/) }' "$1"
}

{
    echo "commit  : $(git rev-parse --short HEAD) (+ working tree: $(git status --short src | tr '\n' ' '))"
    echo "model   : $MODEL"
    echo "nsys    : $("$NSYS" --version)"
    echo "target  : $TARGET MiB available at launch"
    echo "runs    : $RUNS"
    echo "started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "$OUT/info.txt"

for tag in $RUNS; do
    arm=${tag%_*}
    env_=$(arm_env "$arm")
    prompt=$(arm_prompt "$arm")
    extra=()
    case $arm in ood-*) extra=(--ignore-eos) ;; esac
    echo "== $tag: $env_ ${extra[*]} -f $prompt" | tee -a "$OUT/info.txt"

    if tasklist //FI "IMAGENAME eq llama-completion.exe" //NH | grep -qi llama-completion; then
        echo "stopped before $tag: another llama-completion.exe is running" | tee -a "$OUT/info.txt"
        exit 1
    fi

    rm -f "$OUT/flag_$tag.txt"
    for attempt in 1 2 3; do
        "$PY" "$E5/evict.py" | tee "$OUT/pre_$tag.txt"
        "$PY" "$E5/probe.py" "$MODEL" --seed "$(echo "$tag" | cksum | cut -d" " -f1)" | tee -a "$OUT/pre_$tag.txt"

        stop="$CAP/stop_$tag.flag"
        rm -f "$stop" "$OUT/balloon_$tag.txt"
        "$PY" "$OUT/balloon.py" --target "$TARGET" --out "$OUT/balloon_$tag.txt" --stop "$stop" &
        bl=$!
        for _ in $(seq 1 120); do
            [ -s "$OUT/balloon_$tag.txt" ] && break
            sleep 0.5
        done
        cat "$OUT/balloon_$tag.txt" | tee -a "$OUT/pre_$tag.txt"
        balloon_ok "$OUT/balloon_$tag.txt" && break
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
    "$PY" "$OUT/monitor.py" --csv "$OUT/proc_$tag.csv" --out "$OUT/low_$tag.txt" &
    wd=$!
    echo "start_unix $(date +%s.%N)" > "$OUT/wall_$tag.txt"
    # shellcheck disable=SC2046
    env $env_ "$NSYS" profile --trace=cuda-sw,nvtx --sample=none --cpuctxsw=none \
        --cuda-memory-usage=false --cuda-graph-trace=graph:host-only --gpu-metrics-devices=none \
        --force-overwrite=true --export=sqlite --output="$CAP/$tag" \
        "$BIN" "${ARGS[@]}" "${extra[@]}" -f "$prompt" < /dev/null > "$CAP/nsys_$tag.txt" 2>&1 || echo "run failed: $tag" | tee -a "$OUT/info.txt"
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
    # the warm-up checks that 15000 MiB still registers under the balloon without going below the floor
    if [ "${tag##*_}" = w ]; then
        reg=$(sed -n 's/.*registered \([0-9]*\)\.[0-9]* MiB of host experts.*/\1/p' "$OUT/log_$tag.txt" 2> /dev/null | tail -1)
        if [ "${reg:-0}" -lt 14800 ] || grep -q KILLED "$OUT/low_$tag.txt"; then
            echo "stopped after the warm-up: registered ${reg:-0} MiB, $(cat "$OUT/low_$tag.txt")" | tee -a "$OUT/info.txt"
            exit 1
        fi
    fi
done

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/info.txt"
"$PY" "$OUT/summarize.py" --table --cap "$CAP"
