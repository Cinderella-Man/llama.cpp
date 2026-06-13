#!/usr/bin/env bash
# Probe 3 re-verification: per-step cost vs canvas size.
# Doc: full-canvas steps (thr 0.95, canvas-filling prompt), per-step avg
#   64->65ms 128->101ms 256->130ms 448->125ms  => ~15ms floor + ~0.4ms/token
#   (the REFUTATION of "canvas-independent": per-step grows with canvas).
# Plus step-count mechanics via infill (3/6/12-mask -> 4/7/13 steps @thr0.9;
#   early_commit 0.5 collapses 12-mask -> ~4 steps).
set -uo pipefail
ROOT=/home/car/projects/llama.cpp
DREAM=/home/car/models/dream7b/Dream-org_Dream-v0-Instruct-7B-Q4_K_M.gguf
OUT=$ROOT/docs/dllms/throughput-plans/07_layer_f/probe3-perstep-costmodel
MASK='<|mask|>'   # placeholder; replaced below with the model's mask piece

"$ROOT/build/bin/llama-diffusion-server" -m "$DREAM" \
  -ub 512 -ngl 99 --diffusion-eps 0.001 --diffusion-steps 128 \
  --temp 0.2 --top-k 40 --host 127.0.0.1 --port 8080 \
  > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
MP=$(curl -s http://127.0.0.1:8080/health | python3 -c "import sys,json;print(json.load(sys.stdin).get('mask_piece',''))")
echo "mask_piece=[$MP]"

# ---- per-step vs canvas (full-canvas generation, thr 0.95, steps 40) ----
PROMPT='Write an extremely long and detailed technical essay about distributed systems, consensus algorithms, replication, fault tolerance, and networking. Keep writing many paragraphs without stopping.'
echo "=== PER-STEP vs CANVAS (thr 0.95, steps 40) ==="
for N in 64 128 256 448; do
  R=$(curl -s http://127.0.0.1:8080/generate -H 'Content-Type: application/json' \
      -d "{\"prompt\":$(python3 -c "import json,sys;print(json.dumps('$PROMPT'))"),\"n_gen\":$N,\"steps\":40,\"conf_threshold\":0.95,\"temp\":0.2,\"top_k\":40}")
  echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'canvas=$N ms_total={d[\"ms_total\"]:.1f} steps={d[\"steps_done\"]} per_step={d[\"ms_total\"]/max(d[\"steps_done\"],1):.1f}ms')"
done

# ---- step-count vs mask count (infill, thr 0.9) ----
echo "=== STEP COUNT vs MASKS (infill, thr 0.9) ==="
ctx='defmodule M do\n  def f(x) do\n    HOLE\n  end\nend\n'
for K in 3 6 12; do
  HOLE=$(python3 -c "print('$MP'*$K)")
  CANVAS=$(python3 -c "print('$ctx'.replace('HOLE','$HOLE'))")
  for EC in 0 0.5; do
    R=$(curl -s http://127.0.0.1:8080/generate -H 'Content-Type: application/json' \
        -d "{\"prompt\":$(python3 -c "import json;print(json.dumps('''$CANVAS'''))"),\"infill\":true,\"steps\":16,\"conf_threshold\":0.9,\"early_commit\":$EC,\"temp\":0.2,\"top_k\":40}")
    echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);print(f'masks=$K early_commit=$EC steps={d[\"steps_done\"]} ms={d[\"ms_total\"]:.1f}')" 2>/dev/null || echo "masks=$K ec=$EC ERR: $R"
  done
done
echo "done $(date -Iseconds)"
