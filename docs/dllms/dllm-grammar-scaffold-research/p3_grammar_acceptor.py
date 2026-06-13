#!/usr/bin/env python3
# P3: offline Elixir-grammar acceptor study.
# Runs the llama.cpp GBNF validator (test-gbnf-validator, the SAME grammar engine used at
# decode time) over real data to answer: would a grammar have caught the syntactic
# failures, and how often would it FALSE-REJECT valid Elixir?
#
# Truth table we want:
#   references (known-good)        -> should ACCEPT   (rejects = false-reject = the risk)
#   P0 parse-fail drafts           -> should REJECT   (accepts = missed by grammar)
#   P0 compile-fail drafts         -> should ACCEPT   (parse-valid; proves grammar != semantics)
#   P0 check-fail / pass drafts    -> should ACCEPT
#
# Usage: python3 p3_grammar_acceptor.py
import json, subprocess, tempfile, os, collections, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATOR = "/home/car/projects/llama.cpp/build/bin/test-gbnf-validator"
GRAMMAR = os.path.join(HERE, "elixir-subset.gbnf")
SAMPLES = os.path.join(HERE, "p0-samples.jsonl")
REFS = os.path.join(HERE, "references.json")

def accepts(code):
    with tempfile.NamedTemporaryFile("w", suffix=".ex", delete=False) as f:
        f.write(code); path = f.name
    try:
        out = subprocess.run([VALIDATOR, GRAMMAR, path], capture_output=True, text=True, timeout=20).stdout
    finally:
        os.unlink(path)
    return "is valid according" in out

def main():
    if not os.path.exists(VALIDATOR):
        sys.exit("missing validator: build test-gbnf-validator")
    refs = json.load(open(REFS))
    drafts = collections.defaultdict(list)
    for l in open(SAMPLES):
        r = json.loads(l); drafts[r["classB"]].append((f'{r["id"]}/{r["seed"]}', r["draftB"]))

    print("=== P3: Elixir GBNF subset acceptor (engine = llama.cpp test-gbnf-validator) ===\n")
    racc = [name for name, c in [("ref%d" % i, c) for i, c in enumerate(refs)] if accepts(c)]
    print(f"REFERENCES (valid Elixir): accepted {len(racc)}/{len(refs)}  "
          f"-> FALSE-REJECT = {len(refs)-len(racc)}/{len(refs)}")
    for i, c in enumerate(refs):
        if not accepts(c):
            print(f"  false-reject ref{i}: {c.splitlines()[0]} ...")

    print()
    for cls in ["parse", "compile", "check", "pass"]:
        cs = drafts.get(cls, [])
        acc = [name for name, c in cs if accepts(c)]
        verdict = "want REJECT (coverage)" if cls == "parse" else "want ACCEPT (parses)"
        print(f"class {cls:8s} n={len(cs):2d}: grammar-accepted {len(acc):2d}/{len(cs):2d}   [{verdict}]")
        if cls == "parse" and acc:
            print(f"           MISSED (accepted broken): {acc}")
    print("\n(parse-class accepted = grammar failed to catch a real syntax error;")
    print(" compile/check accepted = correct, grammar cannot judge semantics.)")

if __name__ == "__main__":
    main()
