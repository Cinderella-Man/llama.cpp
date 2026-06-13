#!/usr/bin/env bash
# Probe 7 re-verification: prove "sampling time per step" is mis-attributed
# GPU-forward wait (llama_decode is async). Requires probe7.diff applied + rebuild.
# Doc: 256-canvas, sampling time/step 154.94 ms -> 0.06 ms with DIFF_SYNC_AFTER_DECODE,
# per-step total UNCHANGED (~160 ms).
set -uo pipefail
ROOT=/home/car/projects/llama.cpp
DREAM=/home/car/models/dream7b/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf
OUT=$ROOT/docs/dllms/throughput-plans/07_layer_f/probe7-async-decode
P='Write an extremely long detailed essay about distributed systems and never stop writing paragraphs'

run_arm () {  # $1 = label, env DIFF_SYNC_AFTER_DECODE inherited
  "$ROOT/build/bin/llama-diffusion-server" -m "$DREAM" -ub 512 -ngl 99 \
    --diffusion-eps 0.001 --diffusion-steps 128 --temp 0.2 --top-k 40 \
    --host 127.0.0.1 --port 8080 > "$OUT/server-$1.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  # 256-canvas, thr 0.95, steps 40 (matches doc's full-canvas condition)
  curl -s http://127.0.0.1:8080/generate -H 'Content-Type: application/json' \
    -d "{\"prompt\":$(python3 -c "import json;print(json.dumps('$P'))"),\"n_gen\":256,\"steps\":40,\"conf_threshold\":0.95,\"temp\":0.2,\"top_k\":40}" >/dev/null
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  echo "=== arm=$1 (DIFF_SYNC_AFTER_DECODE=${DIFF_SYNC_AFTER_DECODE:-unset}) ==="
  grep "time per step" "$OUT/server-$1.log" | tail -1
}

unset DIFF_SYNC_AFTER_DECODE; run_arm off
export DIFF_SYNC_AFTER_DECODE=1; run_arm on
