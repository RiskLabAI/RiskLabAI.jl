module RiskLabAIDeepBSDEExt

using RiskLabAI

include(joinpath(@__DIR__, "..", "src", "Pde", "DeepBSDESolver.jl"))

end # module RiskLabAIDeepBSDEExt
