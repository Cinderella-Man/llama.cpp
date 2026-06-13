#!/usr/bin/env bash
# F2 STEP 2 dflash closed-loop re-verification.
set -uo pipefail
ROOT=/home/car/projects/llama.cpp
OUT=$ROOT/docs/dllms/throughput-plans/07_layer_f/f2-dflash-closedloop
QWEN=/home/car/models/qwen7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf
FDLLM=/home/car/models/fast-dllm-v2-1.5b/fast-dllm-v2-1.5b-Q4_K_M.gguf

# 8 unique cases from the original drafts file (id = base before first '-')
python3 -c "
import json
seen={}; out=[]
for l in open('/tmp/f2_drafts.jsonl'):
    j=json.loads(l); base=j['id'].split('-')[0]
    if base not in seen: seen[base]=1; out.append({'id':base,'user':j['user']})
open('$OUT/cases8.jsonl','w').write('\n'.join(json.dumps(o) for o in out)+'\n')
"
echo "=== AR baseline ==="
"$ROOT/build/bin/llama-dflash" -m "$QWEN" --model-draft "$FDLLM" -ngl 99 \
  --in "$OUT/cases8.jsonl" --ar 2>&1 | grep -iE "tok/s|baseline"
for DL in 32 16 8; do
  echo "=== dflash draft-len $DL ==="
  "$ROOT/build/bin/llama-dflash" -m "$QWEN" --model-draft "$FDLLM" -ngl 99 \
    --in "$OUT/cases8.jsonl" --draft-len $DL --max-tokens 256 2>&1 \
    | grep -iE "dflash:|accepted/round|efficiency"
done
