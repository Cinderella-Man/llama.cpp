# P4b: grammar-constrained diffusion DECODE (tractable frontier realization), A/B.
# Same scaffold canvas (module + signature + body masks), infilled twice: grammar OFF vs
# grammar ON (server applies elixir-subset.gbnf at the committed-prefix frontier each step).
# Measures whether constraining decode to grammar-valid tokens changes parse/compile/pass
# and at what latency cost. Isolates the grammar's effect (identical canvas/seed).
#
# Run: cd kintsugi && mix run ../docs/dllms/dllm-grammar-scaffold-research/p4b_grammar_infill.exs

alias Kintsugi.{Engine, Verifier}
Code.require_file("/home/car/projects/llama.cpp/kintsugi/bench/cases.exs")

url = System.get_env("KINTSUGI_ENGINE") || "http://127.0.0.1:8080"
{:ok, eng} = Engine.connect(url)
mask = eng.mask_piece
grammar = File.read!("/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/elixir-subset.gbnf")
# grammar engages ONLY on the CPU sampling path (it is not wired into the GPU backend
# sampler), so force backend_sampling:false on BOTH arms to isolate the grammar's effect.
infill_params = %{"steps" => 48, "conf_threshold" => 0.6, "temp" => 0.2, "top_k" => 40, "eps" => 0.001, "backend_sampling" => false}

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

scaffold = fn check ->
  case Regex.run(~r/\b([A-Z]\w*)\.(\w+[?!]?)\((.*)/, check) do
    [_, mod, fname, rest] ->
      args = Enum.take(~w(a b c d e), max(top_args.(rest), 1)) |> Enum.join(", ")
      "defmodule #{mod} do\n  def #{fname}(#{args}) do\n    " <> String.duplicate(mask, 24) <> "\n  end\nend"
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

forge_cases = Enum.filter(Kintsugi.Bench.Cases.all(), &(&1.kind == :forge))
out = "/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/p4b-samples.jsonl"
io = File.open!(out, [:write, :utf8])

samples =
  for c <- forge_cases, seed <- c.seeds, canvas = scaffold.(c.check), canvas != nil do
    run = fn extra ->
      case Engine.infill(eng, canvas, Map.merge(Map.put(infill_params, "seed", seed), extra)) do
        {:ok, %{"text" => t, "ms_total" => ms}} -> {classify.(t, c.check), ms, t}
        _ -> {:error, 0, ""}
      end
    end

    {off_c, off_ms, _} = run.(%{})
    {on_c, on_ms, on_t} = run.(%{"grammar" => grammar})

    row = %{id: c.id, seed: seed, off: off_c, on: on_c, off_ms: off_ms, on_ms: on_ms, on_text: on_t}
    IO.write(io, JSON.encode!(row) <> "\n")
    IO.puts("#{String.pad_trailing(c.id,10)} seed=#{String.pad_trailing(to_string(seed),4)} OFF=#{String.pad_trailing(to_string(off_c),10)} ON=#{String.pad_trailing(to_string(on_c),10)} (#{round(off_ms)}->#{round(on_ms)}ms)")
    row
  end

File.close(io)
n = length(samples)
IO.puts("\n==== P4b grammar-constrained infill A/B (#{n} cases) ====")
IO.inspect(Enum.frequencies_by(samples, & &1.off), label: "grammar OFF")
IO.inspect(Enum.frequencies_by(samples, & &1.on),  label: "grammar ON ")
IO.puts("pass: OFF=#{Enum.count(samples,&(&1.off==:pass))}/#{n}  ON=#{Enum.count(samples,&(&1.on==:pass))}/#{n}")
IO.puts("parse-fail: OFF=#{Enum.count(samples,&(&1.off==:parse))}  ON=#{Enum.count(samples,&(&1.on==:parse))}")
changed = Enum.count(samples, &(&1.off != &1.on))
off_ms = samples |> Enum.map(& &1.off_ms) |> Enum.sum()
on_ms = samples |> Enum.map(& &1.on_ms) |> Enum.sum()
IO.puts("outcome changed by grammar: #{changed}/#{n} cases")
IO.puts("latency: OFF sum=#{round(off_ms)}ms  ON sum=#{round(on_ms)}ms  (grammar overhead #{Float.round(on_ms/max(off_ms,1),2)}x)")
IO.puts("samples -> #{out}")
