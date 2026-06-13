#!/usr/bin/env bash
# Probe 4 re-verification: backend sampling attach overhead on tiny repairs.
# Doc: 3-mask ON 109 / OFF 87 (+23); 6-mask ON 197 / OFF 160 (+37);
#      12-mask ON 364 / OFF 383 (-19). Crossover ~10-12 masks.
# Warm + median-of-5 per cell (tiny-repair timings are noisy).
set -uo pipefail
ROOT=/home/car/projects/llama.cpp
DREAM=/home/car/models/dream7b/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf
OUT=$ROOT/docs/dllms/throughput-plans/07_layer_f/probe4-tinyrepair-onoff
ctx='defmodule M do\n  def f(x) do\n    HOLE\n  end\nend\n'

measure () {  # $1 = arm label, $2... = extra server flags
  local label="$1"; shift
  "$ROOT/build/bin/llama-diffusion-server" -m "$DREAM" -ub 512 -ngl 99 \
    --diffusion-eps 0.001 --diffusion-steps 128 --temp 0.2 --top-k 40 "$@" \
    --host 127.0.0.1 --port 8080 > "$OUT/server-$label.log" 2>&1 &
  local SRV=$!
  for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  MP=$(curl -s http://127.0.0.1:8080/health | python3 -c "import sys,json;print(json.load(sys.stdin).get('mask_piece',''))")
  # warmup
  curl -s http://127.0.0.1:8080/generate -d '{"prompt":"hi","n_gen":32,"steps":4,"seed":1}' >/dev/null
  echo "=== arm=$label ==="
  for K in 3 6 12; do
    HOLE=$(python3 -c "print('$MP'*$K)")
    CANVAS=$(python3 -c "print('$ctx'.replace('HOLE','$HOLE'))")
    vals=()
    for rep in 1 2 3 4 5; do
      ms=$(curl -s http://127.0.0.1:8080/generate -H 'Content-Type: application/json' \
        -d "{\"prompt\":$(python3 -c "import json;print(json.dumps('''$CANVAS'''))"),\"infill\":true,\"steps\":16,\"conf_threshold\":0.9,\"temp\":0.2,\"top_k\":40}" \
        | python3 -c "import sys,json;print(f'{json.load(sys.stdin)[\"ms_total\"]:.1f}')")
      vals+=("$ms")
    done
    med=$(python3 -c "import statistics;print(f'{statistics.median([${vals[*]// /,}]):.1f}')" 2>/dev/null || python3 -c "import statistics;print(f'{statistics.median([$(IFS=,;echo "${vals[*]}")]):.1f}')")
    echo "masks=$K median_ms=$med  reps=[${vals[*]}]"
  done
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
}

measure ON
measure OFF --no-backend-sampling
echo "done $(date -Iseconds)"
