"""Small, deterministic examples for the frozen causal-factor API."""

using RiskLabAI
using RiskLabAI.CausalFactorAnalysis
using LinearAlgebra: Diagonal

@assert Base.pkgversion(RiskLabAI) == v"1.0.0"

covariance = Matrix(Diagonal([1.0, 2.0, 4.0]))
factor_exposures = [1.0 0.0; 0.0 1.0; 1.0 1.0]
target_exposures = [0.0, 1.0]
weights = minimum_variance_factor_weights(
    covariance,
    factor_exposures,
    target_exposures,
)
@assert isapprox(weights, [-2.0 / 7.0, 5.0 / 7.0, 2.0 / 7.0])

effect = average_treatment_effect(3.5, 1.25)
@assert effect == 2.25

dag = CausalDAG(
    ("T", "U", "Y"),
    (("U", "T"), ("U", "Y"), ("T", "Y")),
    ("T", "U", "Y"),
)
@assert !check_backdoor_adjustment_set(dag, "T", "Y").admissible
@assert check_backdoor_adjustment_set(dag, "T", "Y", ("U",)).admissible

println("minimum-variance weights: ", weights)
println("average treatment effect: ", effect)
println("back-door adjustment set: (\"U\",)")
