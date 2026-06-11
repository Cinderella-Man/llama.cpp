defmodule Kintsugi.Masker do
  @moduledoc """
  Turns broken code + a diagnostic location into an infill canvas: the offending span is
  replaced by a run of the model's mask pieces, everything else stays byte-identical
  (that is the infill contract of `llama-diffusion-server`).
  """

  @doc """
  Replace `line_no` (1-based) of `code` with `n` mask pieces, preserving the line's
  leading indentation. Returns the canvas string.
  """
  def mask_line(code, line_no, mask_piece, n \\ 8) do
    lines = String.split(code, "\n")

    masked =
      List.update_at(lines, line_no - 1, fn line ->
        indent = String.slice(line, 0, String.length(line) - String.length(String.trim_leading(line)))
        indent <> String.duplicate(mask_piece, n)
      end)

    Enum.join(masked, "\n")
  end

  @doc """
  Replace lines `from..to` (1-based, inclusive) with a single run of `n` mask pieces,
  indented like the first masked line.
  """
  def mask_lines(code, from, to, mask_piece, n \\ 12) do
    lines = String.split(code, "\n")
    {head, rest} = Enum.split(lines, from - 1)
    {masked, tail} = Enum.split(rest, to - from + 1)

    indent =
      case masked do
        [first | _] ->
          String.slice(first, 0, String.length(first) - String.length(String.trim_leading(first)))

        [] ->
          ""
      end

    Enum.join(head ++ [indent <> String.duplicate(mask_piece, n)] ++ tail, "\n")
  end

  @doc """
  Size the mask run from what it replaces: roughly one mask per 3 characters of replaced
  text, clamped to [min, max]. Diffusion fills exactly the masked positions, so the hole
  has to be big enough for plausible code but small enough to keep the step fast.
  """
  def hole_size(replaced_text, min \\ 4, max \\ 24) do
    (String.length(replaced_text) |> div(3)) |> clamp(min, max)
  end

  defp clamp(v, lo, _hi) when v < lo, do: lo
  defp clamp(v, _lo, hi) when v > hi, do: hi
  defp clamp(v, _lo, _hi), do: v
end
