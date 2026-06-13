#!/usr/bin/env python3
# Reproduce Probe 0's DRAFT/INFILL decomposition from the KINTSUGI_TRACE JSONL.
import json, sys, statistics as st

path = sys.argv[1] if len(sys.argv) > 1 else "ktrace-reverify.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
# warmup "hi" request is the first line; the doc's 421 is the engine-call count.
def pct(xs, p):
    xs = sorted(xs); k = (len(xs)-1)*p/100.0
    f = int(k); c = min(f+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)

draft = [r for r in rows if not r["infill"]]
infill = [r for r in rows if r["infill"]]
# exclude the warmup "hi" from drafts for the production decomposition
draft_nowarm = [r for r in draft if r["prompt"] != "hi"]

print(f"total trace lines        : {len(rows)}  (doc: 421 engine calls)")
print(f"DRAFT calls (incl warmup): {len(draft)}")
print(f"DRAFT calls (excl warmup): {len(draft_nowarm)}  (doc: 58)")
print(f"INFILL calls             : {len(infill)}  (doc: 363)")
print()
def summ(name, rs, doc_wall, doc_med):
    wall = sum(r["ms_total"] for r in rs)/1000.0
    med = st.median(r["ms_total"] for r in rs)
    print(f"{name:7s} | calls {len(rs):3d} | wall {wall:6.1f}s (doc {doc_wall}) | median {med:6.1f}ms (doc {doc_med})")
summ("DRAFT", draft_nowarm, "52.1s", "637")
summ("INFILL", infill, "73.8s", "111")
tot = sum(r["ms_total"] for r in rows)/1000.0
tot_nw = sum(r["ms_total"] for r in (draft_nowarm+infill))/1000.0
print(f"engine wall (all lines)  : {tot:.1f}s")
print(f"engine wall (excl warmup): {tot_nw:.1f}s  (doc: 125.9s)")
print(f"DRAFT share              : {sum(r['ms_total'] for r in draft_nowarm)/ (tot_nw*1000)*100:.0f}%  (doc: 41%)")
print(f"INFILL share             : {sum(r['ms_total'] for r in infill)/(tot_nw*1000)*100:.0f}%  (doc: 59%)")
print()
im = [r["ms_total"] for r in infill]
print("INFILL ms dist: min %.0f p10 %.0f median %.0f p90 %.0f max %.0f"
      % (min(im), pct(im,10), st.median(im), pct(im,90), max(im)))
print("   doc        : min 32  p10 40  median 111 p90 554 max 958")
dp = [r["n_prompt_tokens"] for r in draft_nowarm]
print("DRAFT prompt tokens: median %.0f max %d  (doc: median 63, max 75)" % (st.median(dp), max(dp)))
