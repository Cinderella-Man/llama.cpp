defmodule Kintsugi.Autofix do
  @moduledoc """
  Deterministic, zero-cost corrections of KNOWN model quirks, applied before any
  compile check or GPU repair. The harness principle: never spend a forward pass on
  something a regex fixes reliably.

  Every rule must be conservative: it only rewrites patterns that are ALWAYS invalid
  Elixir (so a correct program can never be damaged).
  """

  @doc "Apply all rules. Idempotent."
  def run(code) do
    code
    # `def f(x), do` at end of line: invalid mix of `, do:` one-liner syntax and a
    # do-block; Dream emits this constantly. The block form is the only valid reading.
    |> String.replace(~r/,\s*do[ \t]*$/m, " do")
    # doubled block opener (`def f(x), do do` -> after rule 1: `do do`)
    |> String.replace(~r/\bdo[ \t]+do\b/, "do")
    # stray markdown fences that survived extraction
    |> String.replace(~r/^\s*```\w*[ \t]*$/m, "")
  end
end
