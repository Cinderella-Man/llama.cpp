# P0: raw-draft failure taxonomy for Dream-7B Elixir forge.
#
# Question: when a Dream draft does not pass, is it a PARSE error (a grammar / scaffold /
# logit-ban could prevent it), a COMPILE/semantic error (undefined symbols - grammar
# cannot help), or a CHECK/logic error (wrong answer - grammar cannot help)? And how much
# of the syntactic slice does the free Credence layer already take?
#
# Run from the kintsugi project root:
#   cd kintsugi && mix run ../docs/dllms/dllm-grammar-scaffold-research/p0_draft_taxonomy.exs
# Output: ../docs/dllms/dllm-grammar-scaffold-research/p0-samples.jsonl + stdout summary.

alias Kintsugi.{Engine, Verifier, Autofix}

Code.require_file("/home/car/projects/llama.cpp/kintsugi/bench/cases.exs")

url = System.get_env("KINTSUGI_ENGINE") || "http://127.0.0.1:8080"
{:ok, eng} = Engine.connect(url)

# draft config = @default_opts (kintsugi.ex:19-28) - faithful to the real draft path
draft_opts = %{"steps" => 128, "conf_threshold" => 0.6, "temp" => 0.2, "top_k" => 40, "eps" => 0.001, "n_gen" => 192}
wrapper = "Write Elixir code for the following task. Reply with ONLY a single ```elixir code block, no explanation.\n\nTask: "

# --- private helpers copied verbatim from kintsugi.ex (normalize_draft:607, align_module_name:591)
normalize_draft = fn code ->
  if code =~ ~r/^\s*defmodule\s/m or not (code =~ ~r/^\s*defp?\s/m) do
    code
  else
    body = code |> String.split("\n") |> Enum.map_join("\n", &("  " <> &1))
    "defmodule KintsugiGen do\n" <> body <> "\nend"
  end
end

align_module_name = fn
  code, check when is_binary(check) ->
    with [_, wanted] <- Regex.run(~r/\b([A-Z]\w*(?:\.[A-Z]\w*)*)\./, check),
         [[_, actual]] <- Regex.scan(~r/defmodule\s+([A-Z][\w.]*)/, code),
         false <- actual == wanted do
      code
      |> String.replace(~r/defmodule\s+#{Regex.escape(actual)}\b/, "defmodule #{wanted}")
      |> String.replace(~r/\b#{Regex.escape(actual)}\./, wanted <> ".")
    else
      _ -> code
    end

  code, _ ->
    code
end

parse_ok? = fn code -> match?({:ok, _}, Code.string_to_quoted(code)) end

parse_msg = fn code ->
  case Code.string_to_quoted(code, columns: true) do
    {:error, {_meta, msg, tok}} -> (is_binary(msg) && msg || inspect(msg)) <> " | tok=" <> inspect(tok)
    _ -> nil
  end
end

# parse -> compile(semantic) -> check(logic) -> pass
classify = fn code, check ->
  cond do
    not parse_ok?.(code) -> :parse
    Verifier.compile(code) != :ok -> :compile
    is_binary(check) and Verifier.run(code, check) != :ok -> :check
    true -> :pass
  end
end

forge_cases = Enum.filter(Kintsugi.Bench.Cases.all(), &(&1.kind == :forge))

out = "/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/p0-samples.jsonl"
io = File.open!(out, [:write, :utf8])

samples =
  for c <- forge_cases, seed <- c.seeds do
    {:ok, %{"text" => raw, "ms_total" => ms}} =
      Engine.generate(eng, wrapper <> c.instruction, Map.put(draft_opts, "seed", seed))

    a = Kintsugi.extract_code(raw)
    b = a |> Autofix.run() |> normalize_draft.() |> align_module_name.(c.check)
    {c_fixed, _trace} = b |> Credence.Syntax.fix_with_trace([])
    cc = c_fixed |> align_module_name.(c.check)
    dd = if Verifier.compile(cc) == :ok, do: cc, else: Map.get(Credence.fix(cc), :code, cc) |> align_module_name.(c.check)

    classB = classify.(b, c.check)
    classC = classify.(cc, c.check)
    classD = classify.(dd, c.check)

    row = %{
      id: c.id, tier: c.tier, seed: seed, ms_total: ms,
      classB: classB, classC: classC, classD: classD,
      parse_msg_B: (classB == :parse && parse_msg.(b)) || nil,
      draftB: b
    }

    IO.write(io, JSON.encode!(row) <> "\n")
    IO.puts("#{String.pad_trailing(c.id, 10)} seed=#{String.pad_trailing(to_string(seed), 4)} B=#{classB}  C=#{classC}  D=#{classD}")
    row
  end

File.close(io)

# ---- summary ----
tally = fn key ->
  samples |> Enum.frequencies_by(&Map.get(&1, key))
end

n = length(samples)
IO.puts("\n==== P0 summary (#{n} forge drafts, Dream-7B Q4_K_M) ====")
IO.puts("stage B (draft entering repair, = extract+autofix+normalize+align):")
IO.inspect(tally.(:classB), label: "  B")
IO.puts("stage C (+ Credence.Syntax, the free regex layer):")
IO.inspect(tally.(:classC), label: "  C")
IO.puts("stage D (+ full Credence.fix, 117 rules):")
IO.inspect(tally.(:classD), label: "  D")

passB = Enum.count(samples, &(&1.classB == :pass))
passC = Enum.count(samples, &(&1.classC == :pass))
passD = Enum.count(samples, &(&1.classD == :pass))
parseB = Enum.count(samples, &(&1.classB == :parse))
parseD = Enum.count(samples, &(&1.classD == :parse))

IO.puts("\nfirst-shot pass: B=#{passB}/#{n}  C=#{passC}/#{n}  D=#{passD}/#{n}")
IO.puts("PARSE failures: B=#{parseB}  -> after full Credence D=#{parseD}  (residual parse errors a grammar could still target)")
IO.puts("\nPARSE diagnostics at stage B (what the syntactic failures actually are):")
for s <- samples, s.classB == :parse, do: IO.puts("  #{s.id}/#{s.seed}: #{s.parse_msg_B}")
IO.puts("\nsamples written to #{out}")
