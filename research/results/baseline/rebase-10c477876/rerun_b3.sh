#!/usr/bin/env bash
# B3 again after lowering the online pinning row from 15000 to 14000 MiB (b3_note.txt).
set -uo pipefail
export PATH="/c/Users/steve/cuda126/bin:$PATH"
MODEL=${1:-C:/code/Codey/models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
OUT=research/results/baseline/rebase-10c477876
CAP=C:/nsys/rebase_10c477876
echo "avail $(powershell -NoProfile -Command '[math]::Round((Get-Counter "\Memory\Available MBytes").CounterSamples[0].CookedValue)') MiB, $(tasklist //FI "IMAGENAME eq VIractal.exe" //NH | grep -i viractal || echo "no VIractal.exe")" > "$OUT/b3_correctness_gpu256.txt"
PIN_PROFILE=C:/nsys/exp0005/profile_bcd.csv NGEN=256 OUT="$CAP/correctness256b" \
    ./moe-bench/check-correctness.sh "$MODEL" >> "$OUT/b3_correctness_gpu256.txt" 2>&1
echo "exit $?" >> "$OUT/b3_correctness_gpu256.txt"
