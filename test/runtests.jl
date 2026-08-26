include("preserved_runtests.jl")

using Test
using LinearAlgebra

using RiskLabAI.CausalFactorAnalysis

const EXPECTED_CAUSAL_FACTOR_EXPORTS = Set((
    :BackdoorAdjustmentEvidence,
    :BacktestStage,
    :CausalDAG,
    :CausalAdjustmentSetStage,
    :ColliderCoefficients,
    :ColliderDiagnostics,
    :CausalDiscoveryStage,
    :CausalExplanatoryAndPredictivePowerStage,
    :CausalFactorProtocolReport,
    :CausalPortfolioConstructionStage,
    :DSeparationEvidence,
    :DifferenceInDifferencesEstimate,
    :EventHorizon,
    :FoldEvidence,
    :FrontdoorAdjustmentEvidence,
    :InstrumentEvidence,
    :MultipleTestingAdjustmentsStage,
    :NodeRoleEvidence,
    :PathEvidence,
    :PopulationRegressionDiagnostics,
    :RegressionDiagnostics,
    :SpecificationExperimentResult,
    :SpecificationPopulationResult,
    :StrategyPerformance,
    :TreatmentEffectDecomposition,
    :ValidationEvidence,
    :VariableSelectionStage,
    :average_treatment_effect,
    :backdoor_adjusted_average_treatment_effect,
    :backdoor_adjusted_expectation,
    :causal_role_evidence,
    :check_backdoor_adjustment_set,
    :check_frontdoor_adjustment_set,
    :check_instrument,
    :collider_factor_return,
    :collider_forecast_return,
    :collider_model_diagnostics,
    :collider_overcontrolled_coefficients,
    :collider_population_diagnostics,
    :collider_specification_experiment,
    :confounded_mediator_population_diagnostics,
    :confounded_mediator_specification_experiment,
    :confounder_factor_return,
    :confounder_forecast_return,
    :confounder_undercontrolled_coefficient,
    :d_separation,
    :difference_in_differences,
    :fork_population_diagnostics,
    :fork_specification_experiment,
    :frontdoor_adjusted_probability,
    :linear_instrumental_variable_effect,
    :minimal_backdoor_adjustment_sets,
    :minimal_frontdoor_adjustment_sets,
    :minimum_variance_factor_weights,
    :randomized_mean_difference,
    :treatment_effect_decomposition,
    :validate_causal_factor_protocol,
    :AllocationMisspecificationDiagnostics,
    :FDRComparisonEvidence,
    :FDRNonIdentificationWitness,
    :FactorControlRoles,
    :GaussianSearchAdjustedFDR,
    :GaussianTrialMixture,
    :MaxSelectionFamilyErrors,
    :SelectionLevelProbabilityEvidence,
    :StructuralCausalModelResult,
    :TreatmentOutcomeRole,
    :TreatmentOutcomeRoleEvidence,
    :allocation_misspecification_diagnostics,
    :classify_treatment_outcome_role,
    :compare_single_and_family_fdr,
    :conditional_upper_tail_probability,
    :evaluate_structural_causal_model,
    :factor_control_roles,
    :family_level_false_discovery_rate,
    :fdr_nonidentification_witness,
    :gaussian_max_selection_cdf,
    :gaussian_max_selection_log_density,
    :gaussian_max_selection_log_likelihood,
    :gaussian_search_adjusted_false_discovery_rate,
    :gaussian_trial_mixture_cdf,
    :generalized_collider_overcontrolled_coefficients,
    :generalized_confounder_undercontrolled_coefficient,
    :max_selection_family_errors,
    :max_selection_null_probability,
    :maximum_mixture_cdf,
    :single_trial_false_discovery_rate,
))

@testset "CausalFactorAnalysis" begin
    @test Set(names(CausalFactorAnalysis)) ==
          union(EXPECTED_CAUSAL_FACTOR_EXPORTS, Set((:CausalFactorAnalysis,)))
    include("causal_factor_analysis/test_treatment_effects.jl")
    include("causal_factor_analysis/test_specification_diagnostics.jl")
    include("causal_factor_analysis/test_factor_mirage.jl")
    include("causal_factor_analysis/test_optimizer.jl")
    include("causal_factor_analysis/test_graph_identification.jl")
    include("causal_factor_analysis/test_protocol.jl")
    include("causal_factor_analysis/test_generalized_factor_mirage.jl")
    include("causal_factor_analysis/test_allocation_diagnostics.jl")
    include("causal_factor_analysis/test_graph_roles.jl")
    include("causal_factor_analysis/test_structural_models.jl")
    include("causal_factor_analysis/test_false_discovery_rates.jl")
    include("causal_factor_analysis/test_additive_numeric_parity.jl")
end
