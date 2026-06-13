#!/usr/bin/env bash
# Probe 1 re-verification: infill wall vs canvas size, fixed 4-mask hole.
# Doc: ~23 prompt tok -> 133 ms; ~292 prompt tok -> 782 ms; 120-line case 500'd
#      (canvas > ubatch 512) => ~2.3 ms/token above ~80 ms floor.
set -uo pipefail
ROOT=/home/car/projects/llama.cpp
DREAM=/home/car/models/dream7b/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf
OUT=$ROOT/docs/dllms/throughput-plans/07_layer_f/probe1-infill-vs-canvas

"$ROOT/build/bin/llama-diffusion-server" -m "$DREAM" -ub 512 -ngl 99 \
  --diffusion-eps 0.001 --diffusion-steps 128 --temp 0.2 --top-k 40 \
  --host 127.0.0.1 --port 8080 > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
MP=$(curl -s http://127.0.0.1:8080/health | python3 -c "import sys,json;print(json.load(sys.stdin).get('mask_piece',''))")
curl -s http://127.0.0.1:8080/generate -d '{"prompt":"hi","n_gen":32,"steps":4,"seed":1}' >/dev/null  # warmup

echo "=== 4-mask hole, growing fixed code context (median of 3) ==="
for L in 1 20 70 130; do
  CANVAS=$(python3 -c "
mp='$MP'
filler='\n'.join('  y%d = %d * 2 + 1'%(i,i) for i in range($L))
print('defmodule M do\n  def f(x) do\n'+filler+'\n    '+mp*4+'\n  end\nend\n')
")
  vals=(); ntok=""
  for rep in 1 2 3; do
    R=$(curl -s -w '\nHTTP%{http_code}' http://127.0.0.1:8080/generate -H 'Content-Type: application/json' \
        -d "{\"prompt\":$(python3 -c "import json,sys;print(json.dumps('''$CANVAS'''))"),\"infill\":true,\"steps\":16,\"conf_threshold\":0.9,\"temp\":0.2,\"top_k\":40}")
    code=$(echo "$R" | grep -o 'HTTP[0-9]*' | tail -1)
    if [ "$code" = "HTTP200" ]; then
      body=$(echo "$R" | sed 's/HTTP[0-9]*$//')
      ms=$(echo "$body" | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'{d[\"ms_total\"]:.1f}')")
      ntok=$(echo "$body" | python3 -c "import sys,json;print(json.load(sys.stdin).get('n_prompt_tokens','?'))")
      vals+=("$ms")
    else
      echo "lines=$L -> $code (canvas exceeded ubatch 512)"; vals=(); break
    fi
  done
  [ ${#vals[@]} -gt 0 ] && { med=$(python3 -c "import statistics;print(f'{statistics.median([$(IFS=,;echo "${vals[*]}")]):.1f}')"); echo "lines=$L prompt_tokens=$ntok median_ms=$med reps=[${vals[*]}]"; }
done
echo "done $(date -Iseconds)"
