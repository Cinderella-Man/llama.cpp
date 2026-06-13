#!/usr/bin/env python3
# Probe 2 re-verification from the Probe 0 KINTSUGI_TRACE.
# F9: draft prompt uniqueness + wall of re-sent prompts.
# F10: consecutive-infill char-overlap distribution + %>90% identical.
import json, sys, statistics as st, difflib

path = sys.argv[1]
rows = [json.loads(l) for l in open(path) if l.strip()]
draft = [r for r in rows if not r["infill"] and r["prompt"] != "hi"]
infill = [r for r in rows if r["infill"]]

# --- F9 ---
prompts = [r["prompt"] for r in draft]
uniq = set(prompts)
seen = set(); reseen_wall = 0.0; reseen_n = 0
for r in draft:
    if r["prompt"] in seen:
        reseen_n += 1; reseen_wall += r["ms_total"]
    seen.add(r["prompt"])
draft_wall = sum(r["ms_total"] for r in draft)/1000.0
print("=== F9 (draft prompt cache) ===")
print(f"draft calls           : {len(draft)}  (doc: 58)")
print(f"unique prompts        : {len(uniq)}  (doc: 11)")
print(f"re-sent (seen) calls  : {reseen_n}  (doc: 47)")
print(f"re-sent wall          : {reseen_wall/1000:.1f}s of {draft_wall:.1f}s  (doc: 51.9s of 52.1s)")

# --- F10 ---  consecutive-infill char overlap (SequenceMatcher ratio on canvas text)
def ov(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()
pairs = []
for i in range(1, len(infill)):
    pairs.append(ov(infill[i-1]["prompt"], infill[i]["prompt"]))
gt90 = sum(1 for x in pairs if x > 0.90)
def pct(xs, p):
    xs = sorted(xs); k=(len(xs)-1)*p/100.0; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)
print("\n=== F10 (cross-request canvas cache) ===")
print(f"consecutive-infill pairs : {len(pairs)}  (doc: 354)")
print(f"char-overlap median      : {st.median(pairs):.2f}  (doc: 0.92)")
print(f"char-overlap p90         : {pct(pairs,90):.2f}  (doc: 0.98)")
print(f">90% identical           : {gt90}/{len(pairs)} ({gt90/len(pairs)*100:.0f}%)  (doc: 208/354, 59%)")
