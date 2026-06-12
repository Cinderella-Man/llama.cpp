defmodule Kintsugi.Verifier do
  @moduledoc """
  The deterministic half of the loop: compile (and optionally test) candidate Elixir code
  and return structured diagnostics the Masker can aim at.

  Compilation runs in this VM via `Code.with_diagnostics/2` + `Code.string_to_quoted` -
  cheap and line-accurate. Generated code is never executed here unless `run/2` is called
  explicitly (which uses a separate OS process for isolation).
  """

  @type diagnostic :: %{line: pos_integer(), message: String.t(), severity: :error | :warning}

  @doc """
  Check that `code` parses and compiles. Returns `:ok` or `{:error, [diagnostic]}`.
  Warnings are returned in the diagnostics list of `:ok_with_warnings` but do not fail.
  """
  def compile(code) do
    case Code.string_to_quoted(code, columns: true) do
      {:error, {meta, message, token}} ->
        {:error, [%{line: meta_line(meta), message: format_parse_error(message, token), severity: :error}]}

      {:ok, _ast} ->
        compile_quoted(code)
    end
  end

  defp compile_quoted(code) do
    case redefined_core_module(code) do
      nil -> do_compile_quoted(code)
      mod ->
        {:error,
         [%{line: 1, message: "candidate redefines already-loaded module #{mod} - rejected " <>
              "(a generated `defmodule List` once clobbered the stdlib and killed the VM)",
            severity: :error}]}
    end
  end

  # generated code occasionally redefines core modules (observed: defmodule List under
  # aggressive decoding) - compiling that into this VM destroys the runtime. Reject any
  # candidate that defines a module which is ALREADY LOADED and not a kintsugi candidate.
  defp redefined_core_module(code) do
    with {:ok, ast} <- Code.string_to_quoted(code) do
      {_, found} =
        Macro.prewalk(ast, nil, fn
          {:defmodule, _, [{:__aliases__, _, parts} | _]} = node, nil ->
            mod = Module.concat(parts)

            if Code.ensure_loaded?(mod) and not candidate_module?(mod) do
              {node, inspect(mod)}
            else
              {node, nil}
            end

          node, acc ->
            {node, acc}
        end)

      found
    else
      _ -> nil
    end
  end

  # in-memory modules report :code.which == [] - the compile-source path (which DOES
  # carry our marker filename) lives in module_info(:compile). Discovered the hard way:
  # the :code.which variant made the redefinition guard reject every RE-compile of a
  # candidate (killing all repair ladders, bench 33/45 -> 13/45) and silently broke
  # purge_candidate_modules since its beginning.
  defp candidate_module?(mod) do
    src = mod.module_info(:compile)[:source]
    is_list(src) and List.to_string(src) =~ "kintsugi_candidate"
  rescue
    _ -> false
  end

  defp do_compile_quoted(code) do
    {result, diagnostics} =
      Code.with_diagnostics(fn ->
        try do
          # compile, don't run: wrap the code so module bodies still compile but
          # top-level expressions are not evaluated
          Code.compile_string(code, "kintsugi_candidate.ex")
          :ok
        rescue
          e in CompileError ->
            {:error, [%{line: e.line || 1, message: Exception.message(e), severity: :error}]}

          e ->
            {:error, [%{line: 1, message: Exception.message(e), severity: :error}]}
        end
      end)

    errors =
      for d <- diagnostics, d.severity == :error do
        %{line: pos_line(d.position), message: d.message, severity: :error}
      end

    purge_candidate_modules()

    case {result, errors} do
      {:ok, []} -> :ok
      {:ok, errs} -> {:error, errs}
      # the logged diagnostics carry the precise message + line; the rescued
      # CompileError is just "cannot compile module ..." - put it last
      {{:error, errs}, more} -> {:error, more ++ errs}
    end
  end

  @doc """
  Compile `code` and run `check` (a string of assertions, e.g. ExUnit-less
  `true = Mod.f(1) == 2`) in a SEPARATE OS process with a timeout - generated code never
  executes inside the harness VM. Returns :ok | {:error, [diagnostic]}.
  """
  def run(code, check, timeout_ms \\ 10_000) do
    script = code <> "\n" <> check <> "\nIO.puts(\"KINTSUGI_OK\")\n"
    path = Path.join(System.tmp_dir!(), "kintsugi_run_#{System.unique_integer([:positive])}.exs")
    File.write!(path, script)

    task = Task.async(fn -> System.cmd("elixir", [path], stderr_to_stdout: true) end)

    try do
      case Task.await(task, timeout_ms) do
        {out, 0} ->
          if String.contains?(out, "KINTSUGI_OK"),
            do: :ok,
            else: {:error, [%{line: 1, message: "check produced no confirmation:\n" <> out, severity: :error}]}

        {out, _status} ->
          {:error, [%{line: extract_line(out), message: out, severity: :error}]}
      end
    catch
      :exit, _ ->
        Task.shutdown(task, :brutal_kill)
        {:error, [%{line: 1, message: "execution timed out after #{timeout_ms} ms", severity: :error}]}
    after
      File.rm(path)
    end
  end

  # -- helpers -------------------------------------------------------------------------

  defp meta_line(meta) when is_list(meta), do: Keyword.get(meta, :line, 1)
  defp meta_line(line) when is_integer(line), do: line
  defp meta_line(_), do: 1

  defp pos_line({line, _col}), do: line
  defp pos_line(line) when is_integer(line), do: line
  defp pos_line(_), do: 1

  defp format_parse_error(message, token) when is_binary(message) and is_binary(token),
    do: message <> token

  defp format_parse_error({opening, hint}, token), do: "#{opening}#{token}#{hint}"
  defp format_parse_error(message, _token), do: inspect(message)

  # elixirc error lines look like: "  kintsugi_run_1.exs:3: ..." or "** (CompileError) file:3:"
  defp extract_line(out) do
    case Regex.run(~r/\.exs?:(\d+)/, out) do
      [_, line] -> String.to_integer(line)
      _ -> 1
    end
  end

  defp purge_candidate_modules do
    for {mod, _file} <- :code.all_loaded(), candidate_module?(mod) do
      :code.purge(mod)
      :code.delete(mod)
    end
  end
end
