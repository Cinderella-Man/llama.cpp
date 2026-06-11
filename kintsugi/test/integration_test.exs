defmodule KintsugiIntegrationTest do
  @moduledoc """
  Live tests against a running llama-diffusion-server (masked family). Skipped unless
  KINTSUGI_ENGINE is set, e.g.:

      KINTSUGI_ENGINE=http://127.0.0.1:8080 mix test --include engine
  """
  use ExUnit.Case

  alias Kintsugi.Engine

  @moduletag :engine
  @engine_url System.get_env("KINTSUGI_ENGINE", "http://127.0.0.1:8080")

  setup_all do
    case Engine.connect(@engine_url) do
      {:ok, %Engine{family: "masked"} = eng} -> {:ok, eng: eng}
      _ -> :ok
    end
  end

  setup ctx do
    if ctx[:eng], do: :ok, else: {:skip, "no masked-family engine at #{@engine_url}"}
  end

  test "health carries the mask piece", %{eng: eng} do
    assert eng.mask_piece != nil and eng.mask_piece != ""
  end

  test "infill keeps fixed text byte-identical", %{eng: eng} do
    canvas = "def add(a, b), do: " <> String.duplicate(eng.mask_piece, 3)
    {:ok, %{"text" => text}} = Engine.infill(eng, canvas, %{"steps" => 8, "conf_threshold" => 0.9, "seed" => 1})
    assert String.starts_with?(text, "def add(a, b), do: ")
    refute text =~ eng.mask_piece
  end

  test "forge produces compiling code", %{eng: eng} do
    {:ok, code, stats} =
      Kintsugi.forge(eng, "a function double/1 that doubles a number", %{"seed" => 42})

    assert :ok = Kintsugi.verify(code, nil)
    IO.puts("\nforge: #{stats.drafts} draft + #{stats.repairs} repairs, #{round(stats.ms_total)} ms\n#{code}")
  end
end
