defmodule KintsugiTest do
  use ExUnit.Case

  alias Kintsugi.{Masker, Verifier}

  describe "Masker" do
    test "mask_line replaces the line, keeps indentation, leaves the rest byte-identical" do
      code = "defmodule A do\n  def f(x) do\n    x +\n  end\nend"
      canvas = Masker.mask_line(code, 3, "<|mask|>", 3)

      assert canvas ==
               "defmodule A do\n  def f(x) do\n    <|mask|><|mask|><|mask|>\n  end\nend"
    end

    test "mask_lines collapses a range into one hole" do
      code = "a\nb\nc\nd"
      assert Masker.mask_lines(code, 2, 3, "_M_", 2) == "a\n_M__M_\nd"
    end

    test "hole_size scales with replaced text and clamps" do
      assert Masker.hole_size("x", 4, 24) == 4
      assert Masker.hole_size(String.duplicate("a", 300), 4, 24) == 24
    end
  end

  describe "Verifier.compile" do
    test "accepts valid code" do
      assert :ok = Verifier.compile("defmodule KintsugiOk do\n  def double(n), do: n * 2\nend")
    end

    test "reports parse errors with a line" do
      assert {:error, [%{line: line} | _]} =
               Verifier.compile("defmodule KintsugiBad do\n  def f(x) do\n    x +\n  end\nend")

      assert line in 3..5
    end

    test "reports undefined function errors" do
      assert {:error, [%{message: msg} | _]} =
               Verifier.compile("defmodule KintsugiUndef do\n  def f(x), do: not_a_fn(x)\nend")

      assert msg =~ "undefined"
    end
  end

  describe "Verifier.run" do
    test "runs a check in an isolated process" do
      code = "defmodule KintsugiRun do\n  def double(n), do: n * 2\nend"
      assert :ok = Verifier.run(code, "4 = KintsugiRun.double(2)")
    end

    test "fails a wrong check" do
      code = "defmodule KintsugiRun2 do\n  def double(n), do: n + 2\nend"
      assert {:error, _} = Verifier.run(code, "4 = KintsugiRun2.double(1)")
    end
  end

  describe "extract_code" do
    test "pulls the first fenced block" do
      assert Kintsugi.extract_code("hi\n```elixir\ndef f, do: 1\n```\nbye") == "def f, do: 1"
    end

    test "falls back to the whole text" do
      assert Kintsugi.extract_code("  def f, do: 1  ") == "def f, do: 1"
    end
  end
end
