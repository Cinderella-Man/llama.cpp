defmodule Kintsugi do
  @moduledoc """
  Kintsugi: verify-and-repair code generation on diffusion LLMs.

  The loop this module implements (see docs/dllm-elixir-harness.md in the parent repo):

      draft (fast, confidence-threshold decode)
        -> compile (Kintsugi.Verifier, deterministic)
        -> if broken: mask the offending span (Kintsugi.Masker)
        -> infill ONLY the hole (diffusion regenerates masked positions, byte-identical
           elsewhere) -> recompile -> repeat

  Drafts cost ~a dozen forward passes; repairs cost a handful on a small canvas. The
  gold-filled cracks give the project its name.
  """

  alias Kintsugi.{Engine, Masker, Verifier}

  @default_opts %{
    "steps" => 128,
    "conf_threshold" => 0.6,
    "temp" => 0.2,
    "top_k" => 40,
    "eps" => 0.001
  }

  @doc """
  Generate compiling Elixir code for `instruction`. Repairs up to `max_repairs` times.

  Returns `{:ok, code, stats}` or `{:error, reason, stats}` where stats counts drafts,
  repairs, total ms and the repair history.
  """
  def forge(%Engine{} = eng, instruction, opts \\ %{}) do
    opts = Map.merge(@default_opts, Map.new(opts, fn {k, v} -> {to_string(k), v} end))
    max_repairs = Map.get(opts, "max_repairs", 4)
    check = Map.get(opts, "check")

    prompt =
      "Write Elixir code for the following task. Reply with ONLY a single ```elixir code block, no explanation.\n\nTask: " <>
        instruction

    case Engine.generate(eng, prompt, Map.drop(opts, ["max_repairs", "check"])) do
      {:ok, %{"text" => text, "ms_total" => ms}} ->
        code = extract_code(text)
        stats = %{drafts: 1, repairs: 0, ms_total: ms, history: []}
        repair_until_ok(eng, code, opts, check, max_repairs, stats)

      {:error, reason} ->
        {:error, reason, %{drafts: 0, repairs: 0, ms_total: 0, history: []}}
    end
  end

  @doc """
  Repair EXISTING code until it verifies (no drafting): the original kintsugi move -
  mask the broken statement, let diffusion fill the hole, recompile. Same opts as forge/3.
  """
  def heal(%Engine{} = eng, code, opts \\ %{}) do
    opts = Map.merge(@default_opts, Map.new(opts, fn {k, v} -> {to_string(k), v} end))
    max_repairs = Map.get(opts, "max_repairs", 4)
    check = Map.get(opts, "check")
    stats = %{drafts: 0, repairs: 0, ms_total: 0, history: []}
    repair_until_ok(eng, code, opts, check, max_repairs, stats)
  end

  @doc "One verify step: compile, then optionally run the check. Returns :ok | {:error, diags}."
  def verify(code, check) do
    with :ok <- Verifier.compile(code) do
      if check, do: Verifier.run(code, check), else: :ok
    end
  end

  @doc """
  One repair round: mask the first error's line and infill it. Masked diffusion fills
  EXACTLY the hole, so hole size is the whole game: the replaced line is tokenized
  (/tokenize) to get the true length, then a small sweep of hole sizes around it is tried
  and the first fill that compiles wins. Returns {:ok, new_code, ms_total_of_attempts}.
  """
  def repair(%Engine{} = eng, code, [%{line: line} | _] = _diags, opts \\ %{}) do
    lines = String.split(code, "\n")
    bad_line = Enum.at(lines, line - 1, "")

    n_tokens =
      case Engine.tokenize(eng, String.trim_leading(bad_line)) do
        {:ok, tokens} -> max(length(tokens), 4)
        _ -> Masker.hole_size(bad_line)
      end

    infill_opts =
      Map.merge(
        %{"steps" => 16, "conf_threshold" => Map.get(opts, "conf_threshold", 0.6)},
        Map.take(opts, ["temp", "top_k", "eps", "seed"])
      )

    # the model cannot end a fill early, so undershooting truncates valid code; sweep
    # exact, +2 and +40% before giving the outer loop a (possibly broken) best effort
    variants = Enum.uniq([n_tokens, n_tokens + 2, round(n_tokens * 1.4)])

    try_hole_variants(eng, code, line, variants, infill_opts, 0, nil)
  end

  defp try_hole_variants(_eng, _code, _line, [], _opts, ms_acc, last) do
    case last do
      nil -> {:error, :no_fill}
      text -> {:ok, text, ms_acc}
    end
  end

  defp try_hole_variants(eng, code, line, [n | rest], opts, ms_acc, _last) do
    canvas = Masker.mask_line(code, line, eng.mask_piece, n)

    case Engine.infill(eng, canvas, opts) do
      {:ok, %{"text" => text, "ms_total" => ms}} ->
        case Verifier.compile(text) do
          :ok -> {:ok, text, ms_acc + ms}
          {:error, _} -> try_hole_variants(eng, code, line, rest, opts, ms_acc + ms, text)
        end

      {:error, reason} ->
        if rest == [], do: {:error, reason}, else: try_hole_variants(eng, code, line, rest, opts, ms_acc, nil)
    end
  end

  # -- internals -----------------------------------------------------------------------

  defp repair_until_ok(_eng, code, _opts, check, 0 = _budget, stats) do
    case verify(code, check) do
      :ok -> {:ok, code, stats}
      {:error, diags} -> {:error, {:still_broken, diags}, stats}
    end
  end

  defp repair_until_ok(eng, code, opts, check, budget, stats) do
    case verify(code, check) do
      :ok ->
        {:ok, code, stats}

      {:error, diags} ->
        round_opts = Map.update(opts, "seed", stats.repairs + 1, &(&1 + stats.repairs + 1))

        case repair(eng, code, diags, round_opts) do
          {:ok, new_code, ms} ->
            stats = %{
              stats
              | repairs: stats.repairs + 1,
                ms_total: stats.ms_total + ms,
                history: stats.history ++ [{:repair, hd(diags).line, hd(diags).message}]
            }

            repair_until_ok(eng, new_code, opts, check, budget - 1, stats)

          {:error, reason} ->
            {:error, reason, stats}
        end
    end
  end

  @doc "Extract the first fenced code block, or return the trimmed text when unfenced."
  def extract_code(text) do
    case Regex.run(~r/```(?:elixir|ex)?\s*\n(.*?)```/s, text) do
      [_, code] -> String.trim_trailing(code)
      nil -> String.trim(text)
    end
  end
end
