# Fixed-seed end-to-end benchmark: instruction -> VERIFIED code, tokens/s of the FINAL
# answer over wall time. Run:  mix run bench/bench.exs [engine_url]
url = List.first(System.argv()) || "http://127.0.0.1:8080"
{:ok, eng} = Kintsugi.Engine.connect(url)

broken_fib =
  "defmodule Fib do\n  def fib(0), do: 0\n  def fib(1), do: 1\n  def fib(n), do fib(n - 1) + fib(n - 2)\nend\n"

cases = [
  {:forge_short,
   fn ->
     Kintsugi.generate_with_stats(eng, "a module Doubler with a function double/1 that doubles a number",
       %{"seed" => 42, "check" => "4 = Doubler.double(2)"})
   end},
  {:forge_medium,
   fn ->
     Kintsugi.generate_with_stats(eng, "a module Sums with a function sum_list/1 that sums a list of numbers recursively",
       %{"seed" => 5, "check" => "6 = Sums.sum_list([1, 2, 3])"})
   end},
  {:forge_long,
   fn ->
     Kintsugi.generate_with_stats(eng,
       "a module Stack with push/2, pop/1 returning {value, rest}, and size/1, implemented on a list",
       %{"seed" => 9, "check" => "{2, [1]} = Stack.pop(Stack.push(Stack.push([], 1), 2)); 1 = Stack.size([:a])"})
   end},
  {:heal_fib, fn -> Kintsugi.heal(eng, broken_fib, %{"check" => "2 = Fib.fib(3)", "seed" => 7}) end}
]

results =
  for {name, run} <- cases do
    case run.() do
      {:ok, _code, stats} ->
        {name, :ok, stats}

      {:error, _reason, stats} ->
        {name, :FAIL, stats}
    end
  end

IO.puts("\ncase          ok    wall_ms  tokens  tok/s   drafts  repairs")

total_ms = total_tok = 0

{total_ms, total_tok, fails} =
  Enum.reduce(results, {0, 0, 0}, fn {name, ok, st}, {ms, tk, fl} ->
    :io.format("~-13s ~-5s ~7w  ~6w  ~-6w ~6w ~8w~n", [
      name, ok, st[:ms_wall], st[:tokens], st[:tokens_per_second], st[:drafts], st[:repairs]
    ])

    {ms + (st[:ms_wall] || 0), tk + (st[:tokens] || 0), fl + if(ok == :ok, do: 0, else: 1)}
  end)

IO.puts("\nTOTAL: #{total_tok} tokens / #{total_ms} ms = #{Float.round(total_tok * 1000 / max(total_ms, 1), 2)} tok/s, #{fails} failures")
