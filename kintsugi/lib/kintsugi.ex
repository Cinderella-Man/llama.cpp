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

      # failures must bill ALL attempts: without this restamp a 3-draft failure reported
      # only the last attempt's wall (measured 4-6x under-billing - bench grilling
      # finding 2 in docs/dllms/dllm-elixir-harness-measuring-updates.md)
      stats =
        stats
        |> Map.put(:ms_wall, System.monotonic_time(:millisecond) - t0)
        |> Map.put(:drafts, spent)

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

    prompt = forge_wrapper(opts) <> instruction

    case Engine.generate(eng, prompt, Map.drop(opts, ["max_repairs", "check", "slim_prompt", "multi_hole"])) do
      {:ok, %{"text" => text, "ms_total" => ms}} ->
        code =
          text |> extract_code() |> Autofix.run() |> normalize_draft() |> align_module_name(check)

        stats = %{drafts: 1, repairs: 0, ms_total: ms, credence_fixes: 0, history: []}

        # an (almost) empty draft COMPILES - reject anything without a function
        # definition outright so the caller redrafts instead of "succeeding" with junk
        if code =~ ~r/^\s*defp?\s/m do
          eng |> repair_until_ok(code, opts, check, max_repairs, stats) |> finalize(eng, t0)
        else
          {:error, :empty_draft, Map.put(stats, :ms_wall, System.monotonic_time(:millisecond) - t0)}
        end

      {:error, reason} ->
        {:error, reason, %{drafts: 0, repairs: 0, ms_total: 0, credence_fixes: 0, history: []}}
    end
  end

  # C5 prompt slimming: 13 fewer wrapper tokens (26 -> 13; full templated prompt
  # 58 -> 45 for a typical bench task) = ~5% fewer rows/step on a 192-token draft.
  # Opt-in per request; pass-rate gated by the bench (prompt changes shift drafts).
  defp forge_wrapper(%{"slim_prompt" => true}), do: "Reply with only an ```elixir code block.\nTask: "

  defp forge_wrapper(%{"slim_prompt" => "mid"}),
    do: "Write Elixir code for the task. Reply with ONLY a single ```elixir code block.\n\nTask: "

  defp forge_wrapper(_opts),
    do: "Write Elixir code for the following task. Reply with ONLY a single ```elixir code block, no explanation.\n\nTask: "

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
    code = Autofix.run(code)
    stats = %{drafts: 0, repairs: 0, ms_total: 0, credence_fixes: 0, history: []}

    eng |> repair_until_ok(code, opts, check, max_repairs, stats) |> finalize(eng, t0)
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
        %{
          "steps" => 16,
          "conf_threshold" => Map.get(opts, "conf_threshold", 0.6),
          # Prophet early-commit (Layer B2): repairs are short, hole-bounded and
          # verified afterwards - bench-measured -35% infill wall, quality-neutral
          # (02_layer_b.md round 2). Drafts keep it OFF (it regresses them).
          "early_commit" => 0.5
        },
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
        # variants are tried in quick succession - only the cheap regex-based syntax
        # phase here; the full (compiling) Credence pass runs once at draft/heal entry
        {text, _trace} = raw |> Autofix.run() |> Credence.Syntax.fix_with_trace([])

        case Verifier.compile(text) do
          :ok -> {:ok, text, ms_acc + ms}
          {:error, _} -> try_hole_variants(eng, code, span, rest, opts, ms_acc + ms, text)
        end

      {:error, reason} ->
        if rest == [], do: {:error, reason}, else: try_hole_variants(eng, code, span, rest, opts, ms_acc, nil)
    end
  end

  # C6: pick the lines for a multi-hole round. Semantic (compile) errors arrive as a
  # LIST of diagnostics; parse errors stop the parser at one, so they never qualify.
  # Adjacent lines stay one hole (interacting fills); the rescued generic CompileError
  # ("cannot compile module ...") duplicates a precise diagnostic - skip it.
  defp multi_hole_lines(code, diags, opts) do
    cap = Map.get(opts, "multi_hole", 0)

    if cap < 2 do
      []
    else
      n_lines = code |> String.split("\n") |> length()

      diags
      |> Enum.filter(&(&1.severity == :error and is_integer(&1.line)))
      |> Enum.reject(&String.starts_with?(&1.message, "cannot compile module"))
      |> Enum.map(& &1.line)
      |> Enum.filter(&(&1 >= 2 and &1 <= n_lines))
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.reduce([], fn ln, acc ->
        case acc do
          [prev | _] when ln - prev <= 1 -> acc
          _ -> [ln | acc]
        end
      end)
      |> Enum.reverse()
      |> Enum.take(cap)
    end
  end

  # C6: mask every target line in ONE canvas (the engine fills all mask runs in a
  # single infill), one verify. Single fill attempt - the outer loop re-verifies and
  # falls back to the single-hole ladder if holes remain broken.
  defp repair_multi(eng, code, lines_to_mask, opts) do
    lines = String.split(code, "\n")

    canvas =
      Enum.reduce(lines_to_mask, code, fn ln, acc ->
        replaced = lines |> Enum.at(ln - 1, "") |> String.trim()

        n =
          case Engine.tokenize(eng, replaced) do
            {:ok, tokens} -> max(length(tokens), 4) + 2
            _ -> Masker.hole_size(replaced)
          end

        Masker.mask_line(acc, ln, eng.mask_piece, n)
      end)

    infill_opts =
      Map.merge(
        %{
          "steps" => 16,
          "conf_threshold" => Map.get(opts, "conf_threshold", 0.6),
          "early_commit" => 0.5
        },
        Map.take(opts, ["temp", "top_k", "eps", "seed"])
      )

    case Engine.infill(eng, canvas, infill_opts) do
      {:ok, %{"text" => raw, "ms_total" => ms}} ->
        {text, _trace} = raw |> Autofix.run() |> Credence.Syntax.fix_with_trace([])
        {:ok, text, ms}

      {:error, reason} ->
        {:error, reason}
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
        # round 0: the deterministic fixer (Credence, 117 rules). Runs ONCE, on the
        # first failure, consuming no repair budget - everything a rule can fix is
        # free; only what remains costs forward passes.
        if Map.get(stats, :credence_fixes, 0) == 0 and not Map.get(stats, :credenced, false) do
          {fixed, n} = deterministic_fix(code)
          stats = stats |> Map.put(:credenced, true) |> Map.put(:credence_fixes, n)

          if fixed != code do
            throw({:credenced, fixed, stats})
          else
            repair_after_diags(eng, code, diags, opts, check, budget, stats)
          end
        else
          repair_after_diags(eng, code, diags, opts, check, budget, stats)
        end
    end
  catch
    {:credenced, fixed, stats} -> repair_until_ok(eng, fixed, opts, check, budget, stats)
  end

  defp repair_after_diags(eng, code, diags, opts, check, budget, stats) do
        same_line_streak =
          stats.history
          |> Enum.reverse()
          |> Enum.take_while(fn {:repair, l, _} -> l == hd(diags).line end)
          |> length()

        round_opts =
          opts
          |> Map.update("seed", stats.repairs + 1, &(&1 + stats.repairs + 1))
          |> Map.put("window", same_line_streak)

        # C6 multi-hole: independent SEMANTIC diagnostics (parse errors stop at one)
        # on distinct non-adjacent lines fix in ONE canvas/infill/verify round. Only
        # on a fresh error (streak 0) - escalation ladders stay single-hole.
        multi_lines =
          if same_line_streak == 0, do: multi_hole_lines(code, diags, opts), else: []

        if length(multi_lines) >= 2 do
          case repair_multi(eng, code, multi_lines, round_opts) do
            {:ok, new_code, ms} ->
              stats = %{
                stats
                | repairs: stats.repairs + 1,
                  ms_total: stats.ms_total + ms,
                  history:
                    stats.history ++
                      [{:repair, hd(multi_lines), "multihole:#{inspect(multi_lines)}"}]
              }

              repair_until_ok(eng, new_code, opts, check, budget - 1, stats)

            {:error, _reason} ->
              repair_single(eng, code, diags, round_opts, opts, check, budget, stats)
          end
        else
          repair_single(eng, code, diags, round_opts, opts, check, budget, stats)
        end
  end

  defp repair_single(eng, code, diags, round_opts, opts, check, budget, stats) do
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

  # THE deterministic stage (the point of the whole exercise): Credence's three-phase
  # rule engine fixes everything rules can fix; only what remains costs forward passes.
  # Credence.fix compiles internally (~hundreds of ms), so it runs ONLY when the code is
  # actually broken - healthy code skips straight through on a cheap compile probe.
  defp deterministic_fix(code) do
    case Verifier.compile(code) do
      :ok ->
        {code, 0}

      {:error, _} ->
        %{code: fixed, applied_rules: applied} = Credence.fix(code)
        {fixed, length(applied)}
    end
  rescue
    # never let the fixer kill the loop - fall back to the unfixed code
    _ -> {code, 0}
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
    # fences come in many broken shapes (DiffuCoder: SIX backticks + a truncated
    # language tag like "elix") - accept any run of 3+ backticks and any tag
    case Regex.run(~r/`{3,}[ \t]*\w*[ \t]*\n(.*?)`{3,}/s, text) do
      [_, code] -> String.trim_trailing(code)
      nil -> String.trim(text)
    end
  end
end
