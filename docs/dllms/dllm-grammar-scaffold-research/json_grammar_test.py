#!/usr/bin/env python3
# Reproduce the paper's setting (Muendler et al. +7%): does grammar-constrained decode help
# a diffusion LLM in a domain where it's SEMANTICALLY COMPETENT (JSON)? Unlike Elixir, the
# model knows JSON well, so residual failures should be syntactic - where a grammar can help.
#
# A/B per prompt x seed: generate JSON with grammar OFF vs ON (grammars/json.gbnf, the
# COMPLETE generic JSON grammar - no subset-strictness). backend_sampling=false on both
# (grammar engages only on the CPU path). Metric: did we get valid JSON?
#   - OFF: is the model's output valid JSON (raw, and after extracting the first {...})?
#   - ON : the grammar forces valid JSON structure - is it valid? (sanity + latency)
# The KEY number is the BASELINE (OFF) validity: high => competent => no room (ceiling),
# low => room for grammar to help.
#
# Run: cd /home/car/projects/llama.cpp && python3 docs/dllms/dllm-grammar-scaffold-research/json_grammar_test.py

import json, urllib.request, os

URL = "http://127.0.0.1:8080"
GRAMMAR = open("/home/car/projects/llama.cpp/grammars/json.gbnf").read()
OUT = "/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/json-samples.jsonl"

PROMPTS = [
    "Output only a JSON object for a person with fields name, age, email, and city. JSON only.",
    "Output only a JSON object for a product: id, title, price, in_stock (boolean), tags (array of strings). JSON only.",
    "Output only a JSON object for a user with a nested address object (street, city, zip). JSON only.",
    "Output only a JSON array of 3 books, each with title, author, and year. JSON only.",
    "Output only a JSON config object: debug (boolean), retries (number), endpoints (array of strings). JSON only.",
    "Output only a JSON object for weather: temp, humidity, conditions, and a forecast array of {day, high, low}. JSON only.",
    "Output only a JSON object mapping three country names to their capital cities. JSON only.",
    "Output only a JSON object for an order: id, items (array of {name, qty, price}), total. JSON only.",
    "Output only a JSON object for a movie: title, year, genres (array), rating (number). JSON only.",
    "Output only a JSON object for a company: name, founded (number), employees (array of {name, role}). JSON only.",
]
SEEDS = [42, 142, 242]

def call(prompt, grammar, seed):
    body = {"prompt": prompt, "raw": False, "steps": 128, "conf_threshold": 0.6,
            "temp": 0.2, "top_k": 40, "eps": 0.001, "n_gen": 192, "seed": seed,
            "backend_sampling": False}
    if grammar:
        body["grammar"] = grammar
    req = urllib.request.Request(URL + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=180))
        return r.get("text", ""), r.get("ms_total", 0)
    except Exception as e:
        return "__ERROR__ " + str(e)[:80], 0

def extract_json(t):
    # first balanced {...} or [...]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i = t.find(open_c)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(t)):
            if t[j] == open_c: depth += 1
            elif t[j] == close_c:
                depth -= 1
                if depth == 0:
                    return t[i:j+1]
    return t.strip()

def valid(s):
    try:
        json.loads(s); return True
    except Exception:
        return False

io = open(OUT, "w")
off_raw = off_ext = on_raw = 0
n = 0
off_ms_sum = on_ms_sum = 0
for p in PROMPTS:
    for seed in SEEDS:
        n += 1
        ot, oms = call(p, None, seed)
        gt, gms = call(p, GRAMMAR, seed)
        off_ms_sum += oms; on_ms_sum += gms
        ovr = valid(ot); ove = valid(extract_json(ot)); gvr = valid(gt)
        off_raw += ovr; off_ext += ove; on_raw += gvr
        io.write(json.dumps({"prompt": p[:40], "seed": seed, "off_raw_valid": ovr,
                             "off_ext_valid": ove, "on_valid": gvr, "off_ms": oms, "on_ms": gms,
                             "off_text": ot, "on_text": gt}) + "\n")
        print(f"{p[:38]:38s} s={seed} OFF raw={int(ovr)} ext={int(ove)} | ON valid={int(gvr)} ({round(oms)}->{round(gms)}ms)")
io.close()

print(f"\n==== JSON grammar A/B ({n} generations, model on :8080) ====")
print(f"OFF raw output valid JSON : {off_raw}/{n}")
print(f"OFF extracted valid JSON  : {off_ext}/{n}   <- baseline competence (high => ceiling, no room)")
print(f"ON  (grammar) valid JSON  : {on_raw}/{n}")
print(f"grammar lift over OFF-extracted: {on_raw - off_ext:+d}")
print(f"latency: OFF {round(off_ms_sum)}ms  ON {round(on_ms_sum)}ms  ({on_ms_sum/max(off_ms_sum,1):.1f}x)")
print(f"-> {OUT}")
