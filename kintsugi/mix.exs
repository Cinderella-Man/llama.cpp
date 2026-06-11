defmodule Kintsugi.MixProject do
  use Mix.Project

  def project do
    [
      app: :kintsugi,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger, :inets, :ssl],
      mod: {Kintsugi.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # the deterministic auto-corrector (117 rules, three phases, compile-gated):
      # Credence fixes what rules can fix; the diffusion engine only fills the rest
      {:credence, path: "../../credence"},
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
    ]
  end
end
