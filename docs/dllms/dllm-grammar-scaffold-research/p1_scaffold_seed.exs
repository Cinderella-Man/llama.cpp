# P1: scaffold-seeding via existing infill (no engine change).
#
# Instead of free-form drafting, seed the module shell and let diffusion fill only the
# body: canvas = "defmodule <Name> do\n  <MASK*n>\nend". <Name> comes from the check.
# Sweep body-hole size n; classify the filled result. Compare first-shot pass vs the
# free-form draft baseline (P0 stage B = 3/30).
#
# Run: cd kintsugi && mix run ../docs/dllms/dllm-grammar-scaffold-research/p1_scaffold_seed.exs

alias Kintsugi.{Engine, Verifier}

Code.require_file("/home/car/projects/llama.cpp/kintsugi/bench/cases.exs")

url = System.get_env("KINTSUGI_ENGINE") || "http://127.0.0.1:8080"
{:ok, eng} = Engine.connect(url)
mask = eng.mask_piece

# from-scratch body fill: draft-like threshold so all masks commit, modest steps (the
# canvas is tiny vs a 192-token draft, so steps are cheap)
infill_params = %{"steps" => 48, "conf_threshold" => 0.6, "temp" => 0.2, "top_k" => 40, "eps" => 0.001}
hole_sweep = [16, 24, 32, 48]

module_name = fn check ->
  case Regex.run(~r/\b([A-Z]\w*)\./, check) do
    [_, name] -> name
    _ -> "Solution"
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

rank = fn :pass -> 4; :check -> 3; :compile -> 2; :parse -> 1; :incomplete -> 0; _ -> 0 end

# baseline (P0 stage B) per {id,seed}
base =
  "/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/p0-samples.jsonl"
  |> File.stream!()
  |> Enum.map(&JSON.decode!/1)
  |> Map.new(fn r -> {{r["id"], r["seed"]}, r["classB"]} end)

forge_cases = Enum.filter(Kintsugi.Bench.Cases.all(), &(&1.kind == :forge))

out = "/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/p1-samples.jsonl"
io = File.open!(out, [:write, :utf8])

samples =
  for c <- forge_cases, seed <- c.seeds do
    name = module_name.(c.check)

    attempts =
      for n <- hole_sweep do
        canvas = "defmodule #{name} do\n  " <> String.duplicate(mask, n) <> "\nend"
        case Engine.infill(eng, canvas, Map.put(infill_params, "seed", seed)) do
          {:ok, %{"text" => filled, "ms_total" => ms}} -> %{n: n, class: classify.(filled, c.check), ms: ms, filled: filled}
          _ -> %{n: n, class: :error, ms: 0, filled: ""}
        end
      end

    best = Enum.max_by(attempts, &rank.(&1.class))
    base_class = Map.get(base, {c.id, seed}, "?")

    row = %{id: c.id, tier: c.tier, seed: seed, name: name,
            best_class: best.class, best_n: best.n, best_ms: best.ms,
            baseline_classB: base_class,
            per_n: Enum.map(attempts, &%{n: &1.n, class: &1.class}),
            best_filled: best.filled}

    IO.write(io, JSON.encode!(row) <> "\n")
    IO.puts("#{String.pad_trailing(c.id, 10)} seed=#{String.pad_trailing(to_string(seed),4)} scaffold=#{String.pad_trailing(to_string(best.class),10)} (n=#{best.n}) | baseline=#{base_class}")
    row
  end

File.close(io)

n = length(samples)
freq = Enum.frequencies_by(samples, & &1.best_class)
scaffold_pass = Enum.count(samples, &(&1.best_class == :pass))
base_pass = Enum.count(samples, &(&1.baseline_classB == "pass"))
# improvement = baseline-failed cases that scaffold passes
fixed = Enum.count(samples, &(&1.baseline_classB != "pass" and &1.best_class == :pass))
broke = Enum.count(samples, &(&1.baseline_classB == "pass" and &1.best_class != :pass))

IO.puts("\n==== P1 scaffold-seed summary (#{n} cases, Dream-7B) ====")
IO.inspect(freq, label: "best_class")
IO.puts("first-shot PASS: scaffold=#{scaffold_pass}/#{n}  vs  baseline draft=#{base_pass}/#{n}")
IO.puts("scaffold fixes #{fixed} baseline-failures; breaks #{broke} baseline-passes")
IO.puts("samples written to #{out}")
