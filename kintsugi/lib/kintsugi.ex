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

  alias Kintsugi.{Autofix, Engine, Masker, Verifier}

  @default_opts %{
    "steps" => 128,
    "conf_threshold" => 0.6,
    "temp" => 0.2,
    "top_k" => 40,
    "eps" => 0.001,
    # generation canvas: every denoising step pays for the full canvas, so size it to
    # the answer, not the server max; redraft attempts double it (see attempt_drafts)
    "n_gen" => 192
  }

  @doc """
  THE entry point: turn an instruction into VERIFIED Elixir code.

  Everything else in this module is plumbing for this call. Behind the scenes it drafts
  (fast confidence-threshold decode), compiles, repairs broken spans by masked infill,
  and - when a draft proves unhealable - quietly starts over with a fresh draft. The
  caller never sees any of that: just `{:ok, code}` where `code` compiles (and passes
  the optional `"check"`), or `{:error, reason}`.

      {:ok, code} = Kintsugi.generate(eng, "a function double/1 that doubles a number")

  Accepts the same opts as the plumbing (`"check"`, `"seed"`, `"max_repairs"`, sampling
  knobs) plus `"max_drafts"` (default 2).
  """
  def generate(%Engine{} = eng, instruction, opts \\ %{}) do
    case generate_with_stats(eng, instruction, opts) do
      {:ok, code, _stats} -> {:ok, code}
      {:error, reason, _stats} -> {:error, reason}
    end
  end

  @doc """
  `generate/3` with the accounting visible: returns `{:ok, code, stats}` where stats
  carries drafts, repairs, ms_wall/ms_total, history, and the throughput fields
  (`tokens` = tokenized FINAL answer only, `tokens_per_second` = tokens / wall seconds -
  discarded drafts and failed fills spend time but never count as tokens).
  """
  def generate_with_stats(%Engine{} = eng, instruction, opts \\ %{}) do
    t0 = System.monotonic_time(:millisecond)
    opts = Map.new(opts, fn {k, v} -> {to_string(k), v} end)
    max_drafts = Map.get(opts, "max_drafts", 3)

    attempt_drafts(eng, instruction, opts, max_drafts, 0, t0, nil)
  end

  defp attempt_drafts(eng, instruction, opts, budget, spent, t0, last_error) do
    if budget <= 0 do
      {reason, stats} = last_error || {:no_drafts_attempted, %{drafts: 0, repairs: 0, ms_total: 0, history: []}}
      {:error, reason, stats}
    else
      # each retry drafts from a different seed (identical canvases fail identically)
      # and on a DOUBLED canvas - a too-small canvas truncates the draft mid-expression,
      # which no amount of repair can fix
      retry_opts =
        opts
        |> Map.update("seed", spent * 1000, &(&1 + spent * 1000))
        |> Map.update("n_gen", 192 * round(:math.pow(2, spent)), &(&1 * round(:math.pow(2, spent))))

      case forge(eng, instruction, retry_opts) do
        {:ok, code, stats} ->
          {:ok, code, restamp(stats, eng, code, spent, t0)}

        {:error, reason, stats} ->
          attempt_drafts(eng, instruction, opts, budget - 1, spent + 1, t0, {reason, stats})
      end
    end
  end

  # fold the wall clock of FAILED attempts into the final accounting and recompute tok/s
  defp restamp(stats, eng, code, prior_attempts, t0) do
    ms_wall = System.monotonic_time(:millisecond) - t0

    tps =
      case Engine.tokenize(eng, code) do
        {:ok, tokens} -> Float.round(length(tokens) * 1000 / max(ms_wall, 1), 2)
        _ -> stats[:tokens_per_second]
      end

    stats
    |> Map.put(:ms_wall, ms_wall)
    |> Map.put(:tokens_per_second, tps)
    |> Map.update(:drafts, 1 + prior_attempts, &(&1 + prior_attempts))
  end

  @doc """
  Plumbing: one draft + its repair loop. Prefer `generate/3`.

  Returns `{:ok, code, stats}` or `{:error, reason, stats}` where stats counts drafts,
  repairs, total ms and the repair history.
  """
  def forge(%Engine{} = eng, instruction, opts \\ %{}) do
    t0 = System.monotonic_time(:millisecond)
    opts = Map.merge(@default_opts, Map.new(opts, fn {k, v} -> {to_string(k), v} end))
    max_repairs = Map.get(opts, "max_repairs", 4)
    check = Map.get(opts, "check")

    prompt =
      "Write Elixir code for the following task. Reply with ONLY a single ```elixir code block, no explanation.\n\nTask: " <>
        instruction

    case Engine.generate(eng, prompt, Map.drop(opts, ["max_repairs", "check"])) do
      {:ok, %{"text" => text, "ms_total" => ms}} ->
        code = text |> extract_code() |> normalize_draft() |> Autofix.run()
        stats = %{drafts: 1, repairs: 0, ms_total: ms, history: []}

        code = align_module_name(code, check)

        # an (almost) empty draft COMPILES - reject anything without a function
        # definition outright so the caller redrafts instead of "succeeding" with junk
        if code =~ ~r/^\s*defp?\s/m do
          eng |> repair_until_ok(code, opts, check, max_repairs, stats) |> finalize(eng, t0)
        else
          {:error, :empty_draft, Map.put(stats, :ms_wall, System.monotonic_time(:millisecond) - t0)}
        end

      {:error, reason} ->
        {:error, reason, %{drafts: 0, repairs: 0, ms_total: 0, history: []}}
    end
  end

  @doc """
  Plumbing: repair EXISTING code until it verifies (no drafting) - mask the broken
  statement, let diffusion fill the hole, recompile. Used by generate/3 via forge/3;
  call directly when you already have code (e.g. human-written) to fix.
  """
  def heal(%Engine{} = eng, code, opts \\ %{}) do
    t0 = System.monotonic_time(:millisecond)
    opts = Map.merge(@default_opts, Map.new(opts, fn {k, v} -> {to_string(k), v} end))
    max_repairs = Map.get(opts, "max_repairs", 4)
    check = Map.get(opts, "check")
    stats = %{drafts: 0, repairs: 0, ms_total: 0, history: []}

    eng |> repair_until_ok(Autofix.run(code), opts, check, max_repairs, stats) |> finalize(eng, t0)
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

    # parse errors often point at where the parser GAVE UP, not where the bug is
    # ("unexpected end ... may not have a matching do" points one line up). The caller
    # escalates "window" when the same error survives a repair: 0 = the line alone,
    # 1 = include the line above, 2+ = the line above through the line below.
    window = Map.get(opts, "window", 0)
    header? = match?("defmodule" <> _, String.trim_leading(Enum.at(lines, 0, "")))

    # a parse error AT the module header means the structure inside is broken, not the
    # header - and masking the header only destroys the skeleton. Same for an exhausted
    # escalation, and for FUNCTIONAL check failures (their line numbers point into the
    # appended check script, past the code - no single line is to blame): remask the
    # WHOLE module body (a guided redraft inside the skeleton).
    {from, to} =
      if header? and length(lines) >= 4 and (line == 1 or line > length(lines) or window >= 3) do
        {2, length(lines) - 1}
      else
        line = min(line, length(lines))
        {max(line - min(window, 1), 1), min(line + (if window >= 2, do: 1, else: 0), length(lines))}
      end

    replaced =
      lines |> Enum.slice((from - 1)..(to - 1)) |> Enum.join("\n") |> String.trim()

    n_tokens =
      case Engine.tokenize(eng, replaced) do
        {:ok, tokens} -> max(length(tokens), 4)
        _ -> Masker.hole_size(replaced)
      end

    infill_opts =
      Map.merge(
        %{"steps" => 16, "conf_threshold" => Map.get(opts, "conf_threshold", 0.6)},
        Map.take(opts, ["temp", "top_k", "eps", "seed"])
      )

    # the model cannot end a fill early, so undershooting truncates valid code; sweep
    # exact, +2 and +40% before giving the outer loop a (possibly broken) best effort
    variants = Enum.uniq([n_tokens, n_tokens + 2, round(n_tokens * 1.4)])

    try_hole_variants(eng, code, {from, to}, variants, infill_opts, 0, nil)
  end

  defp try_hole_variants(_eng, _code, _span, [], _opts, ms_acc, last) do
    case last do
      nil -> {:error, :no_fill}
      text -> {:ok, text, ms_acc}
    end
  end

  defp try_hole_variants(eng, code, {from, to} = span, [n | rest], opts, ms_acc, _last) do
    canvas = Masker.mask_lines(code, from, to, eng.mask_piece, n)

    case Engine.infill(eng, canvas, opts) do
      {:ok, %{"text" => raw, "ms_total" => ms}} ->
        text = Autofix.run(raw)

        case Verifier.compile(text) do
          :ok -> {:ok, text, ms_acc + ms}
          {:error, _} -> try_hole_variants(eng, code, span, rest, opts, ms_acc + ms, text)
        end

      {:error, reason} ->
        if rest == [], do: {:error, reason}, else: try_hole_variants(eng, code, span, rest, opts, ms_acc, nil)
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
        same_line_streak =
          stats.history
          |> Enum.reverse()
          |> Enum.take_while(fn {:repair, l, _} -> l == hd(diags).line end)
          |> length()

        round_opts =
          opts
          |> Map.update("seed", stats.repairs + 1, &(&1 + stats.repairs + 1))
          |> Map.put("window", same_line_streak)

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

  # Throughput accounting: `tokens` is the tokenized FINAL answer only - every
  # generation along the way (draft, discarded hole-size variants, replaced spans)
  # spends TIME but produces no counted tokens, so there is no double counting.
  # tokens_per_second = final-answer tokens / wall-clock seconds (HTTP + compile
  # checks included; ms_total remains the engine-side generation time alone).
  defp finalize({verdict, code_or_reason, stats}, eng, t0) do
    ms_wall = System.monotonic_time(:millisecond) - t0
    stats = Map.put(stats, :ms_wall, ms_wall)

    stats =
      with :ok <- verdict,
           {:ok, tokens} <- Engine.tokenize(eng, code_or_reason) do
        n = length(tokens)

        stats
        |> Map.put(:tokens, n)
        |> Map.put(:tokens_per_second, Float.round(n * 1000 / max(ms_wall, 1), 2))
      else
        _ -> Map.merge(stats, %{tokens: nil, tokens_per_second: nil})
      end

    {verdict, code_or_reason, stats}
  end

  # when the check expects a specific module (e.g. "4 = Doubler.double(2)") but the
  # model named its module something else, rename it deterministically - a free fix
  # that previously cost a full redraft cycle
  defp align_module_name(code, check) when is_binary(check) do
    with [_, wanted] <- Regex.run(~r/\b([A-Z]\w*(?:\.[A-Z]\w*)*)\./, check),
         [[_, actual]] <- Regex.scan(~r/defmodule\s+([A-Z][\w.]*)/, code),
         false <- actual == wanted do
      code
      |> String.replace(~r/defmodule\s+#{Regex.escape(actual)}\b/, "defmodule #{wanted}")
      |> String.replace(~r/\b#{Regex.escape(actual)}\./, wanted <> ".")
    else
      _ -> code
    end
  end

  defp align_module_name(code, _), do: code

  # models frequently draft bare `def ...` functions, which cannot compile outside a
  # module - wrap them so the verifier (and the repair loop) get a fair target
  defp normalize_draft(code) do
    if code =~ ~r/^\s*defmodule\s/m or not (code =~ ~r/^\s*defp?\s/m) do
      code
    else
      body = code |> String.split("\n") |> Enum.map_join("\n", &("  " <> &1))
      "defmodule KintsugiGen do\n" <> body <> "\nend"
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
