# G13 validity probe: capture what the model drafts when SHOWN the check, to detect
# "teaching to the test" (hardcoding the example values instead of writing general code).
# Drafts each forge case once with check-first, prints the code + a crude hardcode flag.
# Run AFTER the benches (GPU is serial): cd kintsugi && mix run ../docs/.../g13_capture.exs

alias Kintsugi.{Engine, Verifier, Autofix}
Code.require_file("/home/car/projects/llama.cpp/kintsugi/bench/cases.exs")
{:ok, eng} = Engine.connect(System.get_env("KINTSUGI_ENGINE") || "http://127.0.0.1:8080")
draft_opts = %{"steps" => 128, "conf_threshold" => 0.6, "temp" => 0.2, "top_k" => 40, "eps" => 0.001, "n_gen" => 192}
wrapper = "Write Elixir code for the following task. Reply with ONLY a single ```elixir code block, no explanation.\n\nTask: "

forge_cases = Enum.filter(Kintsugi.Bench.Cases.all(), &(&1.kind == :forge))

for c <- forge_cases do
  seed = hd(c.seeds)
  prompt = wrapper <> c.instruction <> "\n\nThe code must pass these tests:\n" <> c.check
  {:ok, %{"text" => raw}} = Engine.generate(eng, prompt, Map.put(draft_opts, "seed", seed))
  code = raw |> Kintsugi.extract_code() |> Autofix.run()
  cls =
    cond do
      not match?({:ok, _}, Code.string_to_quoted(code)) -> :parse
      Verifier.compile(code) != :ok -> :compile
      Verifier.run(code, c.check) != :ok -> :check
      true -> :pass
    end
  IO.puts("===== #{c.id} (#{cls}) | check: #{c.check} =====")
  IO.puts(code)
  IO.puts("")
end
