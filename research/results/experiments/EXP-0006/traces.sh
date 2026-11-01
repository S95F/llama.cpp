#!/usr/bin/env bash
# EXP-0006, before the timed runs: one untimed 1024-token selection trace per
# prompt (no pinning, no Nsight Systems; routing and the cache do not depend on
# either), then coverage.py's offline prediction on the windows runs.sh measures.
#
#   ./research/results/experiments/EXP-0006/traces.sh

set -euo pipefail

export PATH="/c/Users/steve/cuda126/bin:$PATH"

MODEL=${MODEL:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/experiments/EXP-0006
E1=research/results/experiments/EXP-0001
BIN=C:/code/llama.cpp-fork/build/bin/llama-completion.exe
PY=/c/Python313/python

ARGS=(-m "$MODEL" -ngl 99 -ncmoe 40 --moe-cache-mib 2048 -c 2048 -n 1024 --temp 0 --seed 1 -t 8 --no-warmup -no-cnv)

LLAMA_MOE_CACHE_TRACE="$OUT/trace_a_1024.csv" "$BIN" "${ARGS[@]}" -f moe-bench/prompt.txt < /dev/null \
    > "$OUT/gen_a_1024.txt" 2> "$OUT/trace_a_1024.log"
LLAMA_MOE_CACHE_TRACE="$OUT/trace_ood_1024.csv" "$BIN" "${ARGS[@]}" --ignore-eos -f "$OUT/prompt_ood.txt" < /dev/null \
    > "$OUT/gen_ood_1024.txt" 2> "$OUT/trace_ood_1024.log"

"$PY" "$OUT/coverage.py" "$MODEL" moe-bench/layout-qwen35moe-35b-a3b-ud-q4kxl.json \
    --runs a="$OUT/trace_a_1024.csv" ood="$OUT/trace_ood_1024.csv" \
    --profile "$E1/trace_prompt_b.csv" "$E1/trace_prompt_c.csv" "$E1/trace_prompt_d.csv" \
    --json "$OUT/coverage_1024.json" 2>&1 | tee "$OUT/coverage_1024.txt"
