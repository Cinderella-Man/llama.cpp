# Bench v2 case definitions - empirically graded 2026-06-12 (see
# ../../docs/dllms/dllm-elixir-harness-measuring-updates.md, appendix C-F).
#
# Tiers (Dream-7B grading; expectations are PER-MODEL - DiffuCoder differs):
#   p_*  pass tier        - reliable passes; measures draft speed + cascade variance
#   c_*  ceiling tier     - stable fails probing two capability holes (strings,
#                           multi-function modules); every pass gained = real movement
#   h_*  heal tier        - Credence-deterministic repairs (~320 ms, 0 GPU expected);
#                           doubles as a Credence regression test
#   a_*  aspirational     - unhealable today; the G3/G4 progress meter
#   i_*  infill micro     - raw engine repair latency, no harness logic
#
# Every case carries a REFERENCE solution; `mix run bench/cases.exs` self-tests each
# check against its reference (a broken check must never poison the bench - one was
# already caught this way: %Point{} literals cannot be used top-level in the same file).

defmodule Kintsugi.Bench.Cases do
  @mask_placeholder "{{MASK}}"

  def all do
    [
      # ---- P: pass tier --------------------------------------------------------------
      %{
        id: "p_double",
        tier: :p,
        kind: :forge,
        seeds: [42, 142, 242],
        instruction: "a module Doubler with a function double/1 that doubles a number",
        check: "4 = Doubler.double(2)",
        reference: "defmodule Doubler do\n  def double(n), do: n * 2\nend"
      },
      %{
        id: "p_even",
        tier: :p,
        kind: :forge,
        seeds: [81, 181, 281],
        instruction: "a module Parity with even?/1 returning true for even integers",
        check: "true = Parity.even?(4); false = Parity.even?(3)",
        reference: "defmodule Parity do\n  def even?(n), do: rem(n, 2) == 0\nend"
      },
      %{
        id: "p_sum",
        tier: :p,
        kind: :forge,
        seeds: [5, 105, 205],
        instruction: "a module Sums with a function sum_list/1 that sums a list of numbers recursively",
        check: "6 = Sums.sum_list([1, 2, 3])",
        reference: "defmodule Sums do\n  def sum_list([]), do: 0\n  def sum_list([h | t]), do: h + sum_list(t)\nend"
      },
      %{
        id: "p_reverse",
        tier: :p,
        kind: :forge,
        seeds: [11, 111, 211],
        instruction: "a module Rev with reverse/1 that reverses a list without using Enum.reverse",
        check: "[3, 2, 1] = Rev.reverse([1, 2, 3])",
        reference:
          "defmodule Rev do\n  def reverse(l), do: do_rev(l, [])\n  defp do_rev([], acc), do: acc\n  defp do_rev([h | t], acc), do: do_rev(t, [h | acc])\nend"
      },
      %{
        id: "p_max",
        tier: :p,
        kind: :forge,
        seeds: [41, 141, 241],
        instruction: "a module Lists with max_of/1 returning the largest number in a non-empty list",
        check: "9 = Lists.max_of([3, 9, 1])",
        reference: "defmodule Lists do\n  def max_of(l), do: Enum.max(l)\nend"
      },
      %{
        id: "p_swap",
        tier: :p,
        kind: :forge,
        seeds: [71, 171, 271],
        instruction: "a module Pairs with swap/1 turning {a, b} into {b, a}",
        check: "{2, 1} = Pairs.swap({1, 2})",
        reference: "defmodule Pairs do\n  def swap({a, b}), do: {b, a}\nend"
      },

      # ---- M: boundary tier (added r3: the long-form gap probe found exactly one
      # candidate Dream can SOMETIMES pass - 58-token single-function-with-docs; 2/3
      # at probe time; this is the flip-prone middle that detects quality movement) ----
      %{
        id: "m_sumdoc",
        tier: :m,
        kind: :forge,
        seeds: [44, 144, 244],
        instruction:
          "a module Acc with sum_to/1 that computes the sum 1+2+...+n recursively, including a @doc comment and a @spec",
        check: "6 = Acc.sum_to(3); 0 = Acc.sum_to(0)",
        reference:
          "defmodule Acc do\n  @doc \"Sums 1..n.\"\n  @spec sum_to(non_neg_integer) :: non_neg_integer\n  def sum_to(0), do: 0\n  def sum_to(n), do: n + sum_to(n - 1)\nend"
      },

      # ---- C: ceiling tier -----------------------------------------------------------
      %{
        id: "c_vowels",
        tier: :c,
        kind: :forge,
        seeds: [21, 121, 221],
        instruction: "a module Vowels with count/1 counting vowels in a string",
        check: "3 = Vowels.count(\"banana\")",
        reference:
          "defmodule Vowels do\n  def count(s) do\n    s |> String.graphemes() |> Enum.count(&(&1 in ~w(a e i o u)))\n  end\nend"
      },
      %{
        id: "c_shout",
        tier: :c,
        kind: :forge,
        seeds: [61, 161, 261],
        instruction: "a module Str with shout/1 that upcases a string and appends an exclamation mark",
        check: "\"HI!\" = Str.shout(\"hi\")",
        reference: "defmodule Str do\n  def shout(s), do: String.upcase(s) <> \"!\"\nend"
      },
      %{
        id: "c_stack",
        tier: :c,
        kind: :forge,
        seeds: [9, 109, 209],
        instruction:
          "a module Stack with push/2, pop/1 returning {value, rest}, and size/1, implemented on a list",
        check: "{2, [1]} = Stack.pop(Stack.push(Stack.push([], 1), 2)); 1 = Stack.size([:a])",
        reference:
          "defmodule Stack do\n  def push(stack, value), do: [value | stack]\n  def pop([h | t]), do: {h, t}\n  def size(stack), do: length(stack)\nend"
      },

      # ---- H: heal tier (Credence-deterministic do-fusion classes) --------------------
      %{
        id: "h_fib",
        tier: :h,
        kind: :heal,
        seeds: [7, 107, 207],
        code:
          "defmodule Fib do\n  def fib(0), do: 0\n  def fib(1), do: 1\n  def fib(n), do fib(n - 1) + fib(n - 2)\nend",
        check: "2 = Fib.fib(3)",
        reference:
          "defmodule Fib do\n  def fib(0), do: 0\n  def fib(1), do: 1\n  def fib(n), do: fib(n - 1) + fib(n - 2)\nend"
      },
      %{
        id: "h_dofusion",
        tier: :h,
        kind: :heal,
        seeds: [7, 107, 207],
        code: "defmodule Calc do\n  def square(n) do: n * n end\nend",
        check: "9 = Calc.square(3)",
        reference: "defmodule Calc do\n  def square(n), do: n * n\nend"
      },
      %{
        id: "h_commado",
        tier: :h,
        kind: :heal,
        seeds: [7, 107, 207],
        code: "defmodule Tax do\n  def add_vat(price), do\n    price * 1.2\n  end\nend",
        check: "true = abs(Tax.add_vat(10) - 12.0) < 0.001",
        reference: "defmodule Tax do\n  def add_vat(price) do\n    price * 1.2\n  end\nend"
      },

      # ---- A: aspirational heal (G3/G4 progress meter) ---------------------------------
      %{
        id: "a_undef",
        tier: :a,
        kind: :heal,
        seeds: [7, 107, 207],
        code: "defmodule Area do\n  def circle(r), do: 3.14159 * radius * radius\nend",
        check: "true = abs(Area.circle(1) - 3.14159) < 0.001",
        reference: "defmodule Area do\n  def circle(r), do: 3.14159 * r * r\nend"
      },

      # ---- I: infill micro (raw engine latency; full params, no server defaults) ------
      %{
        id: "i_small",
        tier: :i,
        kind: :infill,
        seeds: [1, 101, 201],
        canvas: "def add(a, b), do: " <> String.duplicate(@mask_placeholder, 3),
        # contract: fixed text stays byte-identical and the fill compiles in a module
        check_prefix: "def add(a, b), do: ",
        reference: nil
      },
      %{
        id: "i_body",
        tier: :i,
        kind: :infill,
        seeds: [1, 101, 201],
        canvas:
          "defmodule M do\n  def cube(n) do\n    " <>
            String.duplicate(@mask_placeholder, 12) <> "\n  end\nend",
        check_prefix: "defmodule M do",
        reference: nil
      }
    ]
  end

  @doc "Replace the mask placeholder with the engine's actual mask piece."
  def materialize_canvas(canvas, mask_piece), do: String.replace(canvas, @mask_placeholder, mask_piece)

  @doc "Self-test: every check must pass against its hand-written reference solution."
  def self_test do
    results =
      for %{reference: ref} = c when not is_nil(ref) <- all() do
        case Kintsugi.Verifier.run(c.reference, c.check) do
          :ok ->
            {c.id, :ok}

          {:error, diags} ->
            {c.id, {:broken_check, diags}}
        end
      end

    bad = Enum.reject(results, fn {_, r} -> r == :ok end)

    IO.puts("self-test: #{length(results) - length(bad)}/#{length(results)} checks valid")

    for {id, err} <- bad do
      IO.puts("  BROKEN CHECK #{id}: #{inspect(err) |> String.slice(0, 160)}")
    end

    if bad == [], do: :ok, else: :error
  end
end

# `mix run bench/cases.exs` = the self-test
if Kintsugi.Bench.Cases.self_test() == :error do
  System.halt(1)
end
