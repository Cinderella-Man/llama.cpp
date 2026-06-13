# Dump the bench reference solutions (known-good Elixir) to references.json for the P3
# grammar-acceptor study. Run: cd kintsugi && mix run ../docs/.../dump_refs.exs
Code.require_file("/home/car/projects/llama.cpp/kintsugi/bench/cases.exs")

refs =
  Kintsugi.Bench.Cases.all()
  |> Enum.map(&Map.get(&1, :reference))
  |> Enum.reject(&is_nil/1)

path = "/home/car/projects/llama.cpp/docs/dllms/dllm-grammar-scaffold-research/references.json"
File.write!(path, JSON.encode!(refs))
IO.puts("wrote #{length(refs)} references -> #{path}")
