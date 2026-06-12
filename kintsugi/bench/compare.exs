# Bench v2 paired comparison. Usage:
#
#   mix run bench/compare.exs old.jsonl new.jsonl
#
# Pairs rows by (id, seed). Gates (calibrated empirically - measuring-updates doc,
# findings H/L): same-seed outcomes are DETERMINISTIC, so every pass transition is a
# real behavioral change. REGRESSION verdict (exit 1) when:
#   - any pass LOST in the expected-stable tiers (p, h, i), or
#   - a tier's median wall (passing pairs) worsens by > 10% [tight only for
#     same-process runs; cross-restart single-case tails reach +138%]
# c/a-tier transitions are reported as quality movement, not gated.

defmodule Kintsugi.Bench.Compare do
  def main([old_path, new_path]) do
    old = load(old_path)
    new = load(new_path)

    keys = Map.keys(old.rows) |> Enum.filter(&Map.has_key?(new.rows, &1)) |> Enum.sort()

    if keys == [] do
      IO.puts("no paired rows - are these the same suite?")
      System.halt(2)
    end

    IO.puts("comparing #{old.label} -> #{new.label}  (#{length(keys)} paired runs)")

    transitions =
      for k <- keys,
          {o, n} = {old.rows[k], new.rows[k]},
          o.ok != n.ok,
          do: {k, o.ok, n.ok, tier_of(o)}

    {gains, losses} = Enum.split_with(transitions, fn {_, _, n_ok, _} -> n_ok end)

    for {{id, seed}, _, _, tier} <- gains, do: IO.puts("  PASS GAINED  #{id} seed=#{seed} (tier #{tier})")
    for {{id, seed}, _, _, tier} <- losses, do: IO.puts("  PASS LOST    #{id} seed=#{seed} (tier #{tier})")
    if transitions == [], do: IO.puts("  pass outcomes: identical")

    stable_losses = Enum.filter(losses, fn {_, _, _, tier} -> tier in [:p, :h, :i] end)

    wall_regressions =
      for tier <- [:p, :c, :h, :a, :i],
          {om, nm} = {tier_median(old.rows, new.rows, keys, tier, :old),
                      tier_median(old.rows, new.rows, keys, tier, :new)},
          om > 0 do
        delta = (nm - om) / om
        IO.puts("  tier #{tier}: median wall #{om} -> #{nm} ms (#{pct(delta)})")
        {tier, delta}
      end
      |> Enum.filter(fn {_, d} -> d > 0.10 end)

    cond do
      stable_losses != [] ->
        IO.puts("\nVERDICT: REGRESSION - pass lost in stable tier(s): #{inspect(Enum.map(stable_losses, &elem(&1, 0)))}")
        System.halt(1)

      wall_regressions != [] ->
        IO.puts("\nVERDICT: REGRESSION - tier median wall worsened >10%: #{inspect(Enum.map(wall_regressions, &elem(&1, 0)))}")
        System.halt(1)

      true ->
        quality = if gains != [], do: " (quality movement: +#{length(gains)} passes)", else: ""
        IO.puts("\nVERDICT: OK#{quality}")
    end
  end

  def main(_), do: IO.puts("usage: mix run bench/compare.exs old.jsonl new.jsonl")

  defp load(path) do
    [header | rows] = File.stream!(path) |> Enum.map(&JSON.decode!/1)

    %{
      label: header["label"],
      rows:
        Map.new(rows, fn r ->
          {{r["id"], r["seed"]},
           %{ok: r["ok"], wall: r["wall_ms"], tier: String.to_atom(r["tier"])}}
        end)
    }
  end

  defp tier_of(row), do: row.tier

  defp tier_median(old_rows, new_rows, keys, tier, which) do
    walls =
      for k <- keys,
          o = old_rows[k],
          n = new_rows[k],
          o.tier == tier,
          o.ok and n.ok do
        if which == :old, do: o.wall, else: n.wall
      end

    case Enum.sort(walls) do
      [] -> 0
      sorted -> Enum.at(sorted, div(length(sorted), 2))
    end
  end

  defp pct(d), do: "#{if d >= 0, do: "+", else: ""}#{Float.round(d * 100, 1)}%"
end

Kintsugi.Bench.Compare.main(System.argv())
