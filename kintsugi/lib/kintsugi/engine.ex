defmodule Kintsugi.Engine do
  @moduledoc """
  Zero-dependency HTTP client for `llama-diffusion-server` (this repo's general diffusion
  daemon). One Engine = one server endpoint; the server itself dispatches across its GPU
  replicas (`--diffusion-replicas`), so one endpoint can already be a whole rig.

  Uses `:httpc` and the built-in `JSON` module (Elixir >= 1.18) - no deps to fetch, which
  matters on offline mining-rig hosts.
  """

  defstruct base_url: "http://127.0.0.1:8080",
            family: nil,
            mask_piece: nil,
            canvas_length: 0,
            n_ubatch: nil,
            replicas: 1

  @type t :: %__MODULE__{}

  @generate_timeout_ms 600_000

  @doc "Probe /health and return a connected engine struct, or {:error, reason}."
  def connect(base_url \\ "http://127.0.0.1:8080") do
    with {:ok, body} <- get(base_url <> "/health") do
      {:ok,
       %__MODULE__{
         base_url: base_url,
         family: body["family"],
         mask_piece: body["mask_piece"],
         canvas_length: body["canvas_length"] || 0,
         n_ubatch: body["n_ubatch"],
         replicas: body["replicas"] || 1
       }}
    end
  end

  @doc """
  POST /generate. `opts` map goes through to the server verbatim (steps, conf_threshold,
  seed, temp, top_k, infill, raw, return_confidences, ...). Returns the decoded response
  map: %{"text" => ..., "ms_total" => ..., "degenerate" => ..., "confidences" => ...}.
  """
  def generate(%__MODULE__{} = eng, prompt, opts \\ %{}) do
    post(eng.base_url <> "/generate", Map.put(opts, "prompt", prompt))
  end

  @doc "Infill: prompt is a literal canvas containing mask pieces; only those are generated."
  def infill(%__MODULE__{} = eng, canvas, opts \\ %{}) do
    generate(eng, canvas, Map.merge(%{"infill" => true}, opts))
  end

  def tokenize(%__MODULE__{} = eng, content) do
    case post(eng.base_url <> "/tokenize", %{"content" => content}) do
      {:ok, %{"tokens" => tokens}} -> {:ok, tokens}
      other -> other
    end
  end

  # -- bare-bones HTTP ----------------------------------------------------------------

  defp get(url) do
    :inets.start()

    case :httpc.request(:get, {String.to_charlist(url), []}, [timeout: 10_000], body_format: :binary) do
      {:ok, {{_, 200, _}, _, body}} -> {:ok, JSON.decode!(body)}
      {:ok, {{_, status, _}, _, body}} -> {:error, {status, body}}
      {:error, reason} -> {:error, reason}
    end
  end

  defp post(url, payload) do
    :inets.start()

    req = {String.to_charlist(url), [], ~c"application/json", JSON.encode!(payload)}

    case :httpc.request(:post, req, [timeout: @generate_timeout_ms], body_format: :binary) do
      {:ok, {{_, 200, _}, _, body}} -> {:ok, JSON.decode!(body)}
      {:ok, {{_, status, _}, _, body}} -> {:error, {status, safe_error(body)}}
      {:error, reason} -> {:error, reason}
    end
  end

  defp safe_error(body) do
    case JSON.decode(body) do
      {:ok, %{"error" => msg}} -> msg
      _ -> body
    end
  end
end
