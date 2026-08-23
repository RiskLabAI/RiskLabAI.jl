module CausalFactorAnalysis

using LinearAlgebra
using Random
using Statistics

include("TreatmentEffects.jl")
include("SpecificationDiagnostics.jl")
include("FactorMirage.jl")
include("Optimizer.jl")
include("GraphIdentification.jl")
include("Protocol.jl")

export
    BackdoorAdjustmentEvidence,
    BacktestStage,
    CausalDAG,
    CausalAdjustmentSetStage,
    CausalDiscoveryStage,
    CausalExplanatoryAndPredictivePowerStage,
    CausalFactorProtocolReport,
    CausalPortfolioConstructionStage,
    ColliderCoefficients,
    ColliderDiagnostics,
    DSeparationEvidence,
    DifferenceInDifferencesEstimate,
    EventHorizon,
    FoldEvidence,
    FrontdoorAdjustmentEvidence,
    InstrumentEvidence,
    MultipleTestingAdjustmentsStage,
    NodeRoleEvidence,
    PathEvidence,
    PopulationRegressionDiagnostics,
    RegressionDiagnostics,
    SpecificationExperimentResult,
    SpecificationPopulationResult,
    StrategyPerformance,
    TreatmentEffectDecomposition,
    ValidationEvidence,
    VariableSelectionStage,
    average_treatment_effect,
    backdoor_adjusted_average_treatment_effect,
    backdoor_adjusted_expectation,
    causal_role_evidence,
    check_backdoor_adjustment_set,
    check_frontdoor_adjustment_set,
    check_instrument,
    collider_factor_return,
    collider_forecast_return,
    collider_model_diagnostics,
    collider_overcontrolled_coefficients,
    collider_population_diagnostics,
    collider_specification_experiment,
    confounded_mediator_population_diagnostics,
    confounded_mediator_specification_experiment,
    confounder_factor_return,
    confounder_forecast_return,
    confounder_undercontrolled_coefficient,
    d_separation,
    difference_in_differences,
    fork_population_diagnostics,
    fork_specification_experiment,
    frontdoor_adjusted_probability,
    linear_instrumental_variable_effect,
    minimal_backdoor_adjustment_sets,
    minimal_frontdoor_adjustment_sets,
    minimum_variance_factor_weights,
    randomized_mean_difference,
    treatment_effect_decomposition,
    validate_causal_factor_protocol

end
