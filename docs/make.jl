using Documenter
using RiskLabAI

makedocs(
    sitename = "RiskLabAI",
    format = Documenter.HTML(),
    modules = [RiskLabAI]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
