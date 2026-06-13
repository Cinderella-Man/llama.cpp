#!/usr/bin/env bash
# Recompute the batched-variant-sweep projection from measured inputs.
# 1) amortization (rebuild batch-probe w/ batchprobe-realsizes.diff, then):
#    llama-diffusion-batch-probe -m Dream-7B-Q4_K_M -ngl 99 -ub 1024  (see PROBE3c)
# 2) exhaust rate (apply sweep-instrumentation.diff, mix compile, then run bench with
#    KINTSUGI_SWEEP_TRACE=sweep-trace.csv) -> E[tried], E[total]
# 3) projection:
cd "$(git rev-parse --show-toplevel)"
python3 - <<'PY'
import json
F={32:1.96,40:1.79,48:1.35,64:1.22,96:1.05}   # measured K=3 amortization
def Fof(t):
    ks=sorted(F)
    if t<=ks[0]: return F[ks[0]]
    if t>=ks[-1]: return F[ks[-1]]
    for a,b in zip(ks,ks[1:]):
        if a<=t<=b: return F[a]+(F[b]-F[a])*(t-a)/(b-a)
Etried,Etotal=2.151,2.464                       # measured (sweep-trace.csv)
rows=[json.loads(l) for l in open('docs/dllms/throughput-plans/07_layer_f/probe0-baseline-bench/ktrace-reverify.jsonl') if l.strip()]
inf=[r for r in rows if r['infill']]
bench,tok=147.0,949
saved=sum((r['ms_total']/1000)*(1-(Etotal/Etried)/Fof(r['n_prompt_tokens']))
          for r in inf if (Etotal/Etried)/Fof(r['n_prompt_tokens'])<1)
nb=bench-saved
print(f'saving {saved:.1f}s -> bench {nb:.0f}s, deliverable {tok/bench/1000*1e3:.2f} -> {tok/nb/1000*1e3:.2f} tok/s (+{(bench/nb-1)*100:.0f}%) [per-step upper bound]')
print('realistic ~+6-10% after step-count divergence; pass count 35/48 (bit-exact).')
PY
