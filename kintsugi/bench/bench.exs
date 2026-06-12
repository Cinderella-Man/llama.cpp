# Bench v2 runner. Usage:
#
#   mix run bench/bench.exs [url] [label] [profile] [draft_url] [--allow-battery]
#
#   url       engine endpoint           (default http://127.0.0.1:8080)
#   label     free-form run name        (default the profile name)
#   profile   baseline | kvpfx32        (request-parameter overlays; one server
#                                        process serves ALL profiles - same-process
#                                        A/B is required for tight wall comparisons,
#                                        see the measuring-updates doc, finding L/M)
#   draft_url D4 hybrid (06_d4_hybrid.md): a second engine that DRAFTS; `url`
#             becomes the repair/escalation engine. Forge cases route through
#             Kintsugi.generate_hybrid; heal/infill stay on the repair engine
#             (block-AR drafters cannot infill).
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
    "kvpfx32" => %{"kv_prefix" => 32},
    "tau03" => %{"tau_alpha" => 0.3},
    "taub06" => %{"tau_alpha" => 0.6},
    "ec05" => %{"early_commit" => 0.5},
    "remask03" => %{"remask_margin" => 0.3},
    "win64" => %{"window" => 64},
    # C5 slim/mid REMOVED from the runnable set (results kept in results/*c5-*):
    # both REJECTED (slim 28/48, mid 8/18 p-tier), and their repair-cascade load
    # crashed the machine twice - see 03_layer_c.md crash post-mortem. Re-adding
    # them requires reading that section first.
    "grow" => %{"n_gen" => 384, "gen_initial" => 96},
    "big384" => %{"n_gen" => 384},
    "mh2" => %{"multi_hole" => 2},
    "winroute" => %{"win_route" => true},
    # Fast-dLLM v2 (block-AR): the commit threshold is the model's own scale -
    # 0.6 (Dream-tuned) floods adjacent commits into duplicate-token corruption
    "fastdllm" => %{"conf_threshold" => 0.9},
    # E4 same-process A/B (05_layer_e.md): block-kv via request param instead of
    # the --diffusion-block-kv server flag; e3kv = E3 configuration reproduced,
    # e4bs = + backend (GPU) sampling for the block-AR loop
    "e3kv" => %{"conf_threshold" => 0.9, "block_kv" => true, "backend_sampling" => false},
    "e4bs" => %{"conf_threshold" => 0.9, "block_kv" => true, "backend_sampling" => true},
    # D4 hybrid: per-engine thresholds live INSIDE generate_hybrid (draft 0.9,
    # repair engine on its own defaults) - the profile stays empty on purpose
    "d4" => %{}
  }

  # full request params for infill cases - NO reliance on server defaults
  @infill_params %{"steps" => 16, "conf_threshold" => 0.9, "temp" => 0.2, "top_k" => 40, "eps" => 0.001}

  def main(argv) do
    {flags, args} = Enum.split_with(argv, &String.starts_with?(&1, "--"))
    [url, label, profile_name, draft_url] =
      case args do
        [] -> ["http://127.0.0.1:8080", "baseline", "baseline", nil]
        [u] -> [u, "baseline", "baseline", nil]
        [u, l] -> [u, l, "baseline", nil]
        [u, l, p] -> [u, l, p, nil]
        [u, l, p, d] -> [u, l, p, d]
      end

    profile = Map.fetch!(@profiles, profile_name)

    battery_guard!(flags)

    {:ok, eng} = Kintsugi.Engine.connect(url)

    # warmup: first request pays sampler attach + graph capture (~85 ms measured)
    Kintsugi.Engine.generate(eng, "hi", %{"steps" => 4, "n_gen" => 32, "seed" => 1})

    draft_eng =
      case draft_url do
        nil ->
          nil

        d ->
          {:ok, de} = Kintsugi.Engine.connect(d)
          Kintsugi.Engine.generate(de, "hi", %{"n_gen" => 32, "seed" => 1})
          de
      end

    out = results_path(label)
    File.mkdir_p!(Path.dirname(out))
    io = File.open!(out, [:write, :utf8])

    IO.write(io, JSON.encode!(header(url, eng, label, profile_name, profile, draft_url, draft_eng)) <> "\n")

    rows =
      for c <- Kintsugi.Bench.Cases.all(), seed <- c.seeds do
        ram_guard!(out)
        row = run_case({eng, draft_eng}, c, seed, profile)
        IO.write(io, JSON.encode!(row) <> "\n")
        row
      end

    File.close(io)
    summarize(rows, label, out)
  end

  # a runaway generated-code runner that escapes the Verifier's kill (or any other
  # leak) must abort the bench before it takes the whole desktop session down -
  # two machine OOMs on 2026-06-12 happened exactly this way; cases are sequential,
  # so no legitimate kintsugi_run_ process exists between them
  @min_avail_kb 4_000_000

  defp ram_guard!(out) do
    System.cmd("pkill", ["-9", "-f", "kintsugi_run_"], stderr_to_stdout: true)

    avail_kb =
      case Regex.run(~r/MemAvailable:\s+(\d+) kB/, File.read!("/proc/meminfo")) do
        [_, kb] -> String.to_integer(kb)
        _ -> nil
      end

    if avail_kb && avail_kb < @min_avail_kb do
      IO.puts(:stderr, "ABORT: MemAvailable #{div(avail_kb, 1024)} MB below #{div(@min_avail_kb, 1024)} MB floor - partial results preserved in #{out}")
      System.halt(75)
    end
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

  defp header(url, eng, label, profile_name, profile, draft_url \\ nil, draft_eng \\ nil) do
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
      draft_url: draft_url,
      draft_engine: draft_eng && Map.from_struct(draft_eng),
      git_rev: git.(["rev-parse", "--short", "HEAD"]),
      # the deterministic-fix layer is part of the measured system - a Credence change
      # invalidated an afternoon of verdicts before this field existed
      credence_rev:
        (case System.cmd("git", ["-C", Path.expand("../../credence", __DIR__), "rev-parse", "--short", "HEAD"],
               stderr_to_stdout: true) do
           {out, 0} -> String.trim(out)
           _ -> nil
         end),
      credence_dirty:
        (case System.cmd("git", ["-C", Path.expand("../../credence", __DIR__), "status", "--porcelain"],
               stderr_to_stdout: true) do
           {out, 0} -> out != ""
           _ -> nil
         end),
      power_ac: Path.wildcard("/sys/class/power_supply/A*/online")
                |> Enum.map(&String.trim(File.read!(&1)))
                |> List.first(),
      gpu: gpu,
      elixir: System.version(),
      otp: System.otp_release()
    }
  end

  defp run_case(engs, c, seed, profile) do
    t0 = System.monotonic_time(:millisecond)
    {ok, reported} = dispatch(engs, c, seed, profile)
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

  defp dispatch({eng, nil}, %{kind: :forge} = c, seed, profile) do
    opts = Map.merge(%{"seed" => seed, "check" => c.check}, profile)

    case Kintsugi.generate_with_stats(eng, c.instruction, opts) do
      {:ok, _, st} -> {true, slim(st)}
      {:error, _, st} -> {false, slim(st)}
    end
  end

  # D4 hybrid: forge cases run the two-engine ladder; everything else stays on
  # the repair engine via the heads below
  defp dispatch({eng, draft_eng}, %{kind: :forge} = c, seed, profile) do
    opts = Map.merge(%{"seed" => seed, "check" => c.check}, profile)

    case Kintsugi.generate_hybrid(draft_eng, eng, c.instruction, opts) do
      {:ok, _, st} -> {true, slim(st)}
      {:error, _, st} -> {false, slim(st)}
    end
  end

  defp dispatch({eng, _draft_eng}, %{kind: :heal} = c, seed, profile) do
    opts = Map.merge(%{"seed" => seed, "check" => c.check}, profile)

    case Kintsugi.heal(eng, c.code, opts) do
      {:ok, _, st} -> {true, slim(st)}
      {:error, _, st} -> {false, slim(st)}
    end
  end

  defp dispatch({eng, _draft_eng}, %{kind: :infill} = c, seed, profile) do
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
    Map.take(st, [:ms_wall, :ms_total, :drafts, :repairs, :credence_fixes, :tokens, :tokens_per_second, :rung])
  end

  defp results_path(label) do
    stamp = DateTime.utc_now() |> Calendar.strftime("%Y%m%dT%H%M%SZ")
    Path.join([__DIR__, "results", "#{stamp}-#{label}.jsonl"])
  end

  defp summarize(rows, label, out) do
    IO.puts("\n#{label}  (#{out})")
    IO.puts("tier  pass   median_wall  worst_wall  notes")

    for tier <- [:p, :m, :c, :h, :a, :i] do
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
