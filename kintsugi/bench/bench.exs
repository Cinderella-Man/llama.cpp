# Bench v2 runner. Usage:
#
#   mix run bench/bench.exs [url] [label] [profile] [--allow-battery]
#
#   url      engine endpoint            (default http://127.0.0.1:8080)
#   label    free-form run name         (default the profile name)
#   profile  baseline | kvpfx32         (request-parameter overlays; one server
#                                        process serves ALL profiles - same-process
#                                        A/B is required for tight wall comparisons,
#                                        see the measuring-updates doc, finding L/M)
#
# Output: bench/results/<utc-timestamp>-<label>.jsonl
#   line 1: {"type":"header", ...environment + profile...}
#   lines : {"type":"case","id":..,"seed":..,"ok":..,"wall_ms":..(EXTERNAL),..}
# plus a per-tier summary on stdout. Walls are measured AROUND each call - reported
# stats are recorded but never used as a denominator (failures under-billed 4-6x
# before the lib restamp fix; trust nothing).

Code.require_file("cases.exs", __DIR__)

defmodule Kintsugi.Bench.Runner do
  @profiles %{
    "baseline" => %{},
    "kvpfx32" => %{"kv_prefix" => 32}
  }

  # full request params for infill cases - NO reliance on server defaults
  @infill_params %{"steps" => 16, "conf_threshold" => 0.9, "temp" => 0.2, "top_k" => 40, "eps" => 0.001}

  def main(argv) do
    {flags, args} = Enum.split_with(argv, &String.starts_with?(&1, "--"))
    [url, label, profile_name] =
      case args do
        [] -> ["http://127.0.0.1:8080", "baseline", "baseline"]
        [u] -> [u, "baseline", "baseline"]
        [u, l] -> [u, l, "baseline"]
        [u, l, p] -> [u, l, p]
      end

    profile = Map.fetch!(@profiles, profile_name)

    battery_guard!(flags)

    {:ok, eng} = Kintsugi.Engine.connect(url)

    # warmup: first request pays sampler attach + graph capture (~85 ms measured)
    Kintsugi.Engine.generate(eng, "hi", %{"steps" => 4, "n_gen" => 32, "seed" => 1})

    out = results_path(label)
    File.mkdir_p!(Path.dirname(out))
    io = File.open!(out, [:write, :utf8])

    IO.write(io, JSON.encode!(header(url, eng, label, profile_name, profile)) <> "\n")

    rows =
      for c <- Kintsugi.Bench.Cases.all(), seed <- c.seeds do
        row = run_case(eng, c, seed, profile)
        IO.write(io, JSON.encode!(row) <> "\n")
        row
      end

    File.close(io)
    summarize(rows, label, out)
  end

  defp battery_guard!(flags) do
    ac =
      case Path.wildcard("/sys/class/power_supply/A*/online") do
        [p | _] -> String.trim(File.read!(p)) == "1"
        [] -> true
      end

    unless ac or "--allow-battery" in flags do
      IO.puts(:stderr, "REFUSING to run on battery power - measured up to 5x timing skew.")
      IO.puts(:stderr, "Plug in, or pass --allow-battery to record a knowingly-skewed run.")
      System.halt(2)
    end
  end

  defp header(url, eng, label, profile_name, profile) do
    git = fn args ->
      case System.cmd("git", args, stderr_to_stdout: true) do
        {out, 0} -> String.trim(out)
        _ -> nil
      end
    end

    gpu =
      case System.cmd("nvidia-smi", ~w(--query-gpu=name,power.draw,clocks.sm --format=csv,noheader),
             stderr_to_stdout: true
           ) do
        {out, 0} -> String.trim(out)
        _ -> nil
      end

    %{
      type: "header",
      label: label,
      profile: profile_name,
      profile_params: profile,
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      url: url,
      engine: Map.from_struct(eng),
      git_rev: git.(["rev-parse", "--short", "HEAD"]),
      power_ac: Path.wildcard("/sys/class/power_supply/A*/online")
                |> Enum.map(&String.trim(File.read!(&1)))
                |> List.first(),
      gpu: gpu,
      elixir: System.version(),
      otp: System.otp_release()
    }
  end

  defp run_case(eng, c, seed, profile) do
    t0 = System.monotonic_time(:millisecond)
    {ok, reported} = dispatch(eng, c, seed, profile)
    wall = System.monotonic_time(:millisecond) - t0

    %{
      type: "case",
      id: c.id,
      tier: c.tier,
      kind: c.kind,
      seed: seed,
      ok: ok,
      wall_ms: wall,
      reported: reported
    }
  end

  defp dispatch(eng, %{kind: :forge} = c, seed, profile) do
    opts = Map.merge(%{"seed" => seed, "check" => c.check}, profile)

    case Kintsugi.generate_with_stats(eng, c.instruction, opts) do
      {:ok, _, st} -> {true, slim(st)}
      {:error, _, st} -> {false, slim(st)}
    end
  end

  defp dispatch(eng, %{kind: :heal} = c, seed, profile) do
    opts = Map.merge(%{"seed" => seed, "check" => c.check}, profile)

    case Kintsugi.heal(eng, c.code, opts) do
      {:ok, _, st} -> {true, slim(st)}
      {:error, _, st} -> {false, slim(st)}
    end
  end

  defp dispatch(eng, %{kind: :infill} = c, seed, profile) do
    canvas = Kintsugi.Bench.Cases.materialize_canvas(c.canvas, eng.mask_piece)
    params = @infill_params |> Map.merge(profile) |> Map.put("seed", seed)

    case Kintsugi.Engine.infill(eng, canvas, params) do
      {:ok, %{"text" => text, "ms_total" => ms}} ->
        contract = String.starts_with?(text, c.check_prefix) and
                     not String.contains?(text, eng.mask_piece)
        compiles = Kintsugi.Verifier.compile(wrap_for_compile(text)) == :ok
        {contract and compiles, %{ms_total: ms}}

      _ ->
        {false, %{}}
    end
  end

  defp wrap_for_compile("defmodule" <> _ = text), do: text
  defp wrap_for_compile(text), do: "defmodule InfillCheck do\n#{text}\nend"

  defp slim(st) do
    Map.take(st, [:ms_wall, :ms_total, :drafts, :repairs, :credence_fixes, :tokens, :tokens_per_second])
  end

  defp results_path(label) do
    stamp = DateTime.utc_now() |> Calendar.strftime("%Y%m%dT%H%M%SZ")
    Path.join([__DIR__, "results", "#{stamp}-#{label}.jsonl"])
  end

  defp summarize(rows, label, out) do
    IO.puts("\n#{label}  (#{out})")
    IO.puts("tier  pass   median_wall  worst_wall  notes")

    for tier <- [:p, :c, :h, :a, :i] do
      runs = Enum.filter(rows, &(&1.tier == tier))
      npass = Enum.count(runs, & &1.ok)
      walls = Enum.map(runs, & &1.wall_ms) |> Enum.sort()
      med = median(walls)
      worst = List.last(walls)
      IO.puts(
        "#{tier}     #{npass}/#{length(runs)}   #{med}ms        #{worst}ms"
      )
    end

    total_pass = Enum.count(rows, & &1.ok)
    tokens = rows |> Enum.map(&(&1.reported[:tokens] || 0)) |> Enum.sum()
    wall_all = rows |> Enum.map(& &1.wall_ms) |> Enum.sum()

    IO.puts(
      "TOTAL #{total_pass}/#{length(rows)} | deliverable #{tokens} tok / #{wall_all} ms = " <>
        "#{Float.round(tokens * 1000 / max(wall_all, 1), 2)} tok/s (honest denominator: ALL runs)"
    )
  end

  defp median([]), do: 0
  defp median(sorted) do
    n = length(sorted)
    Enum.at(sorted, div(n, 2))
  end
end

Kintsugi.Bench.Runner.main(System.argv())
