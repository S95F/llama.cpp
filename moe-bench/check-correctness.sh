#!/usr/bin/env bash
# The MoE cache must not change what the model generates: it only moves expert
# weights into VRAM, the kernels and the values are the same.  This greedily
# decodes the same prompt under every cache configuration and under the
# unmodified upstream build, and diffs the output.
#
#   ./moe-bench/check-correctness.sh <model.gguf> [baseline_bin_dir]
#
# baseline_bin_dir is the bin/ of a build of upstream master.  Without it only
# the configurations of this build are compared against each other.

set -uo pipefail

MODEL=${1:?usage: check-correctness.sh <model.gguf> [baseline_bin_dir]}
BASE_BIN=${2:-}
BIN=${BIN:-./build/bin}
OUT=${OUT:-moe-bench/out/correctness}
NCMOE=${NCMOE:-40}
THREADS=${THREADS:-8}
NGEN=${NGEN:-256}

# experts the cache does not serve would run on the CPU, whose rounding differs from CUDA's and flips a greedy token
# within a few hundred tokens: offload them too, so that only the movement of the experts differs (EXP-0005)
export GGML_OP_OFFLOAD_MIN_BATCH=${GGML_OP_OFFLOAD_MIN_BATCH:-1}

mkdir -p "$OUT"
rm -f "$OUT"/*.txt
PROMPT=moe-bench/prompt.txt

# run <name> <bin dir> <env assignment|-> <extra args...>
run() {
    local name=$1; shift
    local bindir=$1; shift
    local envs=$1; shift
    echo "-- $name ${envs#-}"
    local -a pre=(env)
    [ "$envs" != "-" ] && pre+=("$envs")
    "${pre[@]}" "$bindir/llama-completion" -m "$MODEL" -ngl 99 -ncmoe "$NCMOE" \
        -c 2048 -n "$NGEN" -f "$PROMPT" --temp 0 --seed 1 -t "$THREADS" \
        --no-warmup -no-cnv "$@" < /dev/null 2>/dev/null \
        | sed -e 's/\r$//' > "$OUT/$name.txt"
    wc -c < "$OUT/$name.txt"
}

# reference: cache disabled, this build
run cache-off "$BIN" - --moe-cache-mib 0

# cache on, above and below the one-token threshold, eviction auto / forced off
run cache2048-lru    "$BIN" LLAMA_MOE_CACHE_SAMPLE=0  --moe-cache-mib 2048
run cache2048-auto   "$BIN" -                         --moe-cache-mib 2048
run cache528-lru     "$BIN" LLAMA_MOE_CACHE_SAMPLE=0  --moe-cache-mib 528
run cache528-auto    "$BIN" -                         --moe-cache-mib 528
run cache2048-s16    "$BIN" LLAMA_MOE_CACHE_SAMPLE=16 --moe-cache-mib 2048

# victim scored by use count only, and a cache small enough to starve a group of slots
run cache528-dist0   "$BIN" LLAMA_MOE_CACHE_DIST=0    --moe-cache-mib 528
run cache256-auto    "$BIN" -                         --moe-cache-mib 256

# pinned host expert pages must not change anything either
run cache1024-pinned "$BIN" GGML_CUDA_REGISTER_HOST=1 --moe-cache-mib 1024

# nor pinning by expert, ordered by a selection trace (PIN_PROFILE, the LLAMA_MOE_CACHE_TRACE format)
if [ -n "${PIN_PROFILE:-}" ]; then
    LLAMA_MOE_PIN_PROFILE="$PIN_PROFILE" run cache2048-pinprof "$BIN" GGML_CUDA_REGISTER_HOST=1 --moe-cache-mib 2048
fi

if [ -n "$BASE_BIN" ]; then
    run upstream-off  "$BASE_BIN" - --moe-cache-mib 0
    run upstream-2048 "$BASE_BIN" - --moe-cache-mib 2048
fi

echo
echo "== diff against cache-off =="
fail=0
for f in "$OUT"/*.txt; do
    n=$(basename "$f" .txt)
    [ "$n" = "cache-off" ] && continue
    if diff -q "$OUT/cache-off.txt" "$f" > /dev/null; then
        echo "  OK    $n"
    else
        echo "  DIFF  $n"
        diff "$OUT/cache-off.txt" "$f" | head -10
        fail=1
    fi
done
exit $fail
