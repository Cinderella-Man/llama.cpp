# Reproducing the paper's setting: grammar on JSON (a domain the model is competent at)

The user-chosen open branch: does grammar help these diffusion models in a domain where
they are SEMANTICALLY COMPETENT (the paper's C++/JSON), as opposed to Elixir where they are
capability-bound? Engine: P4b grammar re-applied + extended to non-infill generation
(`p4b-engine.diff` + the `gram_start = infill ? 0 : n_input` edit). Grammar:
`grammars/json.gbnf` (the COMPLETE generic JSON grammar - no subset-strictness). Model:
DiffuCoder-7B-cpGRPO (the paper's model). 2026-06-14.

## Findings

### 1. The domain HAS headroom (unlike Elixir) - the key contrast
With raw-mode JSON priming (`prompt ... :\n{"name":`), DiffuCoder emits valid JSON ~50%
(3/6 across prompts/styles). The failures are purely SYNTACTIC, exactly what a grammar
prevents:
- `"in_stock": true/false` (invalid literal)
- `"in_stock":": "true"}` (stray punctuation)
- `"email": "john@example.com.com"`, truncated `"city": "New Yor` (length)

This is the opposite of Elixir: there the model is SEMANTICALLY incompetent (writes
`String.count`, wrong algorithms - no CFG can fix), so grammar has nothing to grab. In JSON
the model is structurally competent with syntactic SLIPS - precisely where a grammar helps.
So the paper's +7% mechanism is plausible HERE, and our Elixir negative is confirmed
DOMAIN/CAPABILITY-specific, not "grammar never helps."

### 2. But the tractable frontier-grammar CANNOT capitalize on it
A/B over 10 prompts x 3 seeds (chat mode), grammar OFF vs ON: **0/30 valid both ways.**
Two reasons:
- Task framing: DiffuCoder-cpGRPO answers "output JSON" by writing a ```python``` script
  that BUILDS json - not raw JSON. (A code model does code.)
- Structural: the frontier-grammar constrains only the FIRST masked position; under
  diffusion's confidence-order commits, later tokens commit OUT OF ORDER and escape the
  constraint. Sanity: grammar forced `{` at position 1, then the model's prose committed
  after it -> `{ is the code to solve... }`, corrupt. Once the canvas drifts off-grammar
  the latch disables it. Net: it corrupts the start, then gives up.

### 3. Reproducing the paper REQUIRES the full algorithm
The paper's contribution is constraining OUT-OF-ORDER / multi-region diffusion via
CFG-intersection completability (accept a token at any masked slot only if a valid
completion still exists) + lookahead. The tractable frontier-only approximation is
structurally insufficient (confirmed empirically). To realize the grammar benefit one must
either (a) port the full eth-sri any-order CFG-intersection, or (b) force left-to-right
single-frontier commits under grammar (degrades diffusion to AR - slow, but enforces).
Neither is the small effort; catalog G11 "effort: large / research-grade" is vindicated.

## Verdict on the open branch

- The Elixir negative is DOMAIN-SPECIFIC: Elixir fails for SEMANTIC reasons (capability
  ceiling) that grammar can't touch; JSON has SYNTACTIC headroom that grammar (a real one)
  could. So "grammar never helps" would be wrong - it's "grammar doesn't help OUR
  (Elixir, capability-bound) stack."
- We did NOT obtain a positive grammar number, because the tractable frontier-grammar can't
  enforce structure under out-of-order commits, and the code-tuned model doesn't emit raw
  JSON. A definitive positive needs the full eth-sri port (or forced-AR) + a JSON-eliciting
  setup - a large effort, and one that does NOT help the Elixir goal (Elixir has no
  syntactic headroom to capture).

## Artifacts
`json_grammar_test.py` (A/B harness), `json-samples.jsonl` (per-generation), the raw-primed
baseline probe (inline). Engine grammar re-applied to source (revert with
`git checkout examples/diffusion/`; `p4b-engine.diff` + the non-infill one-liner capture it).
