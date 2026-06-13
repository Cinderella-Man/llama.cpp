# P1b: signature-seeded scaffold. P1 showed module-shell seeding kills parse errors but
# residual failures are wrong-function-name / wrong-API (model renames the function). Seed
# the function signature too, derived from the check:
#   defmodule <Name> do
#     def <fname>(<a,b,..>) do
#       <MASK*n>
#     end
#   end
# Compare pass vs P1 (module-only) and baseline draft.
#
# Run: cd kintsugi && mix run ../docs/dllms/dllm-grammar-scaffold-research/p1b_signature_seed.exs

alias Kintsugi.{Engine, Verifier}
Code.require_file("/home/car/projects/llama.cpp/kintsugi/bench/cases.exs")

url = System.get_env("KINTSUGI_ENGINE") || "http://127.0.0.1:8080"
{:ok, eng} = Engine.connect(url)
mask = eng.mask_piece
infill_params = %{"steps" => 48, "conf_threshold" => 0.6, "temp" => 0.2, "top_k" => 40, "eps" => 0.001}
hole_sweep = [12, 16, 24]

# count comma-separated args at depth 1 until the function's closing paren
top_args = fn s ->
  {n, depth, started, any} =
    s |> String.to_charlist() |> Enum.reduce_while({0, 1, false, false}, fn ch, {n, d, st, any} ->
      cond do
        ch in [?(, ?[, ?{] -> {:cont, {n, d + 1, st, any}}
        ch in [?), ?], ?}] ->
          d2 = d - 1
          if d2 == 0, do: {:halt, {n, d2, st, any}}, else: {:cont, {n, d2, st, any}}
        ch == ?, and d == 1 -> {:cont, {n + 1, d, st, true}}
        ch in [?\s, ?\t] -> {:cont, {n, d, st, any}}
        true -> {:cont, {n, d, st, true}}
      end
    end)
  _ = {depth, started}
  if any, do: n + 1, else: 0
end

# top_args is referenced inside sig via closure-order; rebind sig now that top_args exists
sig = fn check ->
  case Regex.run(~r/\b([A-Z]\w*)\.(\w+[?!]?)\((.*)/, check) do
    [_, mod, fname, rest] ->
      arity = top_args.(rest)
      args = Enum.take(~w(a b c d e), max(arity, 1)) |> Enum.join(", ")
      {mod, fname, args}
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

base = "/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/p0-samples.jsonl"
       |> File.stream!() |> Enum.map(&JSON.decode!/1)
       |> Map.new(fn r -> {{r["id"], r["seed"]}, r["classB"]} end)
p1 = "/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/p1-samples.jsonl"
     |> File.stream!() |> Enum.map(&JSON.decode!/1)
     |> Map.new(fn r -> {{r["id"], r["seed"]}, r["best_class"]} end)

forge_cases = Enum.filter(Kintsugi.Bench.Cases.all(), &(&1.kind == :forge))
out = "/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/p1b-samples.jsonl"
io = File.open!(out, [:write, :utf8])

samples =
  for c <- forge_cases, seed <- c.seeds do
    {name, fname, args} =
      case sig.(c.check) do
        {m, f, a} -> {m, f, a}
        _ -> {"Solution", nil, nil}
      end

    attempts =
      for n <- hole_sweep do
        canvas =
          if fname do
            "defmodule #{name} do\n  def #{fname}(#{args}) do\n    " <> String.duplicate(mask, n) <> "\n  end\nend"
          else
            "defmodule #{name} do\n  " <> String.duplicate(mask, n) <> "\nend"
          end

        case Engine.infill(eng, canvas, Map.put(infill_params, "seed", seed)) do
          {:ok, %{"text" => filled, "ms_total" => ms}} -> %{n: n, class: classify.(filled, c.check), ms: ms, filled: filled}
          _ -> %{n: n, class: :error, ms: 0, filled: ""}
        end
      end

    best = Enum.max_by(attempts, &rank.(&1.class))
    row = %{id: c.id, seed: seed, sig: "#{fname}/#{args}", best_class: best.class, best_n: best.n,
            baseline: Map.get(base, {c.id, seed}), p1_module_only: Map.get(p1, {c.id, seed}), best_filled: best.filled}
    IO.write(io, JSON.encode!(row) <> "\n")
    IO.puts("#{String.pad_trailing(c.id,10)} seed=#{String.pad_trailing(to_string(seed),4)} sig-seed=#{String.pad_trailing(to_string(best.class),9)} | p1(module)=#{String.pad_trailing(to_string(Map.get(p1,{c.id,seed})),8)} baseline=#{Map.get(base,{c.id,seed})}")
    row
  end

File.close(io)
n = length(samples)
IO.puts("\n==== P1b signature-seed summary (#{n} cases, Dream-7B) ====")
IO.inspect(Enum.frequencies_by(samples, & &1.best_class), label: "best_class")
IO.puts("first-shot PASS: sig-seed=#{Enum.count(samples, &(&1.best_class==:pass))}/#{n}  " <>
        "module-only=#{Enum.count(samples, &(&1.p1_module_only=="pass"))}/#{n}  " <>
        "baseline=#{Enum.count(samples, &(&1.baseline=="pass"))}/#{n}")
IO.puts("samples -> #{out}")
