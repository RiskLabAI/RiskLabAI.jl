"""Small, deterministic examples for the frozen causal-factor API."""

using RiskLabAI
using RiskLabAI.CausalFactorAnalysis
using LinearAlgebra: Diagonal

@assert Base.pkgversion(RiskLabAI) == v"1.1.0"

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
role = classify_treatment_outcome_role(dag, "T", "Y", "U")
@assert role.role == TreatmentOutcomeRole(:confounder)

generalized_coefficient = generalized_confounder_undercontrolled_coefficient(
    2.0,
    3.0,
    4.0;
    confounder_variance = 5.0,
    exposure_noise_variance = 6.0,
)
@assert generalized_coefficient == 116.0 / 43.0

family_errors = max_selection_family_errors(
    0.024997895148220373,
    0.9515427737332771,
    0.95,
    10,
)
@assert isapprox(family_errors.family_type_i_error, 0.22365361940347483)

println("minimum-variance weights: ", weights)
println("average treatment effect: ", effect)
println("back-door adjustment set: (\"U\",)")
println("one-hop role for U: ", role.role.value)
println("general-variance confounder coefficient: ", generalized_coefficient)
println("ten-trial family Type-I error: ", family_errors.family_type_i_error)
