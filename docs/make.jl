using Documenter
using RiskLabAI
using RiskLabAI.CausalFactorAnalysis

makedocs(
    sitename = "RiskLabAI",
    format = Documenter.HTML(),
    modules = [RiskLabAI, RiskLabAI.CausalFactorAnalysis],
    source = ".",
    pages = [
        "Home" => "src/index.md",
        "Causal factor analysis" => "causal_factor_analysis.md",
        "Compatibility" => "compatibility.md",
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
