# DiffuCoder cross-check: re-run P0 (raw-draft taxonomy) + P1 (module-shell scaffold) + P1b
# (signature scaffold) on DiffuCoder-7B-cpGRPO, same bench/params/harness as the Dream runs.
# Question: is the "syntax is not the bottleneck; scaffold doesn't raise pass" finding
# Dream-specific, or does a CODE-tuned diffusion model show the same?
#
# Run (DiffuCoder server on :8080): cd kintsugi && mix run ../docs/.../dc_crosscheck.exs

alias Kintsugi.{Engine, Verifier, Autofix}
Code.require_file("/home/car/projects/llama.cpp/kintsugi/bench/cases.exs")

{:ok, eng} = Engine.connect(System.get_env("KINTSUGI_ENGINE") || "http://127.0.0.1:8080")
mask = eng.mask_piece
draft_opts = %{"steps" => 128, "conf_threshold" => 0.6, "temp" => 0.2, "top_k" => 40, "eps" => 0.001, "n_gen" => 192}
infill_params = %{"steps" => 48, "conf_threshold" => 0.6, "temp" => 0.2, "top_k" => 40, "eps" => 0.001}
wrapper = "Write Elixir code for the following task. Reply with ONLY a single ```elixir code block, no explanation.\n\nTask: "

# private helpers copied from kintsugi.ex (normalize_draft:607, align_module_name:591)
normalize_draft = fn code ->
  if code =~ ~r/^\s*defmodule\s/m or not (code =~ ~r/^\s*defp?\s/m) do
    code
  else
    body = code |> String.split("\n") |> Enum.map_join("\n", &("  " <> &1))
    "defmodule KintsugiGen do\n" <> body <> "\nend"
  end
end
align = fn
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
  code, _ -> code
end

top_args = fn s ->
  {n, any} =
    s |> String.to_charlist() |> Enum.reduce_while({0, 1, false}, fn ch, {n, d, any} ->
      cond do
        ch in [?(, ?[, ?{] -> {:cont, {n, d + 1, any}}
        ch in [?), ?], ?}] -> if d - 1 == 0, do: {:halt, {n, 0, any}}, else: {:cont, {n, d - 1, any}}
        ch == ?, and d == 1 -> {:cont, {n + 1, d, true}}
        ch in [?\s, ?\t] -> {:cont, {n, d, any}}
        true -> {:cont, {n, d, true}}
      end
    end)
    |> case do {n, _d, any} -> {n, any} end
  if any, do: n + 1, else: 0
end

mod_of = fn check -> case Regex.run(~r/\b([A-Z]\w*)\./, check) do [_, m] -> m; _ -> "Solution" end end
sig = fn check ->
  case Regex.run(~r/\b([A-Z]\w*)\.(\w+[?!]?)\((.*)/, check) do
    [_, mod, fname, rest] ->
      {mod, fname, Enum.take(~w(a b c d e), max(top_args.(rest), 1)) |> Enum.join(", ")}
    _ -> nil
  end
end

classify = fn code, check ->
  cond do
    String.contains?(code, mask) -> :incomplete
    not match?({:ok, _}, Code.string_to_quoted(code)) -> :parse
    Verifier.compile(code) != :ok -> :compile
    is_binary(check) and Verifier.run(code, check) != :ok -> :check
    true -> :pass
  end
end
rank = fn :pass -> 4; :check -> 3; :compile -> 2; :parse -> 1; _ -> 0 end
best = fn canvases, check, seed ->
  canvases
  |> Enum.map(fn canvas ->
    case Engine.infill(eng, canvas, Map.put(infill_params, "seed", seed)) do
      {:ok, %{"text" => t}} -> classify.(t, check)
      _ -> :error
    end
  end)
  |> Enum.max_by(rank)
end

forge_cases = Enum.filter(Kintsugi.Bench.Cases.all(), &(&1.kind == :forge))
out = "/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/dc-samples.jsonl"
io = File.open!(out, [:write, :utf8])

samples =
  for c <- forge_cases, seed <- c.seeds do
    # P0: raw draft
    {:ok, %{"text" => raw}} = Engine.generate(eng, wrapper <> c.instruction, Map.put(draft_opts, "seed", seed))
    draft = raw |> Kintsugi.extract_code() |> Autofix.run() |> normalize_draft.() |> align.(c.check)
    raw_class = classify.(draft, c.check)

    # P1: module-shell scaffold (sweep), seed fixed per case via infill_params? infill uses no seed -> add
    name = mod_of.(c.check)
    p1 = best.(Enum.map([16, 24, 32, 48], fn n -> "defmodule #{name} do\n  " <> String.duplicate(mask, n) <> "\nend" end), c.check, seed)

    # P1b: signature scaffold (sweep)
    p1b =
      case sig.(c.check) do
        {m, f, a} -> best.(Enum.map([12, 16, 24], fn n -> "defmodule #{m} do\n  def #{f}(#{a}) do\n    " <> String.duplicate(mask, n) <> "\n  end\nend" end), c.check, seed)
        _ -> p1
      end

    row = %{id: c.id, seed: seed, raw: raw_class, scaffold: p1, sig: p1b}
    IO.write(io, JSON.encode!(row) <> "\n")
    IO.puts("#{String.pad_trailing(c.id,10)} seed=#{String.pad_trailing(to_string(seed),4)} raw=#{String.pad_trailing(to_string(raw_class),9)} scaffold=#{String.pad_trailing(to_string(p1),9)} sig=#{p1b}")
    row
  end

File.close(io)
n = length(samples)
IO.puts("\n==== DiffuCoder-7B cross-check (#{n} forge drafts) ====")
IO.inspect(Enum.frequencies_by(samples, & &1.raw), label: "raw draft (P0)")
IO.inspect(Enum.frequencies_by(samples, & &1.scaffold), label: "module scaffold (P1)")
IO.inspect(Enum.frequencies_by(samples, & &1.sig), label: "signature scaffold (P1b)")
IO.puts("first-shot PASS: raw=#{Enum.count(samples,&(&1.raw==:pass))}/#{n}  scaffold=#{Enum.count(samples,&(&1.scaffold==:pass))}/#{n}  sig=#{Enum.count(samples,&(&1.sig==:pass))}/#{n}")
IO.puts("PARSE-fail: raw=#{Enum.count(samples,&(&1.raw==:parse))}  scaffold=#{Enum.count(samples,&(&1.scaffold==:parse))}  sig=#{Enum.count(samples,&(&1.sig==:parse))}")
IO.puts("-> #{out}")
