defmodule Kintsugi.Autofix do
  @moduledoc """
  Extraction-artifact cleanup ONLY (stray markdown fences). All Elixir-syntax
  corrections live in Credence (`Credence.Syntax.FixDoBlockFusion` et al.) - the
  deterministic fixer is one shared, tested rule engine, not scattered regexes.
  """

  @doc "Strip stray markdown fences that survived code extraction. Idempotent."
  def run(code) do
    String.replace(code, ~r/^\s*`{3,}\w*[ \t]*$/m, "")
  end
end
