using Test
using TOML
using Random
using RiskLabAI

const FactorDiagnostics = RiskLabAI.CausalFactorAnalysis
const factor_diagnostic_fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "factor_diagnostics_python_320.toml"))

function check_factor_fields(actual, expected; atol = 1e-12)
    for name in propertynames(actual)
        result, reference = getproperty(actual, name), expected[string(name)]
        if result isa Bool || result isa String || result isa Integer
            @test result == reference
        elseif reference isa AbstractDict
            check_factor_fields(result, reference; atol)
        elseif result isa Tuple
            if all(value -> value isa String, result)
                @test collect(result) == reference
            else
                @test collect(result) ≈ reference rtol = 1e-10 atol = atol
            end
        else
            @test result ≈ reference rtol = 1e-10 atol = atol
        end
    end
end

mutable struct SpecificationFixtureRNG <: AbstractRNG
    draws::Vector{Vector{Float64}}
    position::Int
end

function Random.randn(rng::SpecificationFixtureRNG, count::Int)
    result = copy(rng.draws[rng.position])
    rng.position += 1
    length(result) == count || error("incorrect fixture draw count")
    return result
end

@testset "Closed-form public factor diagnostics against Python 3.2.0" begin
    for row in factor_diagnostic_fixtures["factor"]
        beta, gamma, delta = row["beta"], row["gamma"], row["delta"]
        @test FactorDiagnostics.confounder_undercontrolled_coefficient(beta, gamma, delta) ≈ row["confounder_coefficient"] rtol = 1e-10 atol = 1e-12
        coefficients = FactorDiagnostics.collider_overcontrolled_coefficients(beta, gamma, delta)
        @test coefficients isa FactorDiagnostics.ColliderCoefficients
        check_factor_fields(coefficients, row["collider_coefficients"])
        @test FactorDiagnostics.generalized_confounder_undercontrolled_coefficient(beta, gamma, delta; confounder_variance = row["confounder_variance"], exposure_noise_variance = row["exposure_variance"]) ≈ row["generalized_confounder"] rtol = 1e-10 atol = 1e-12
        check_factor_fields(FactorDiagnostics.generalized_collider_overcontrolled_coefficients(beta, gamma, delta; outcome_noise_variance = row["outcome_variance"], collider_noise_variance = row["collider_variance"]), row["generalized_collider"])
        factor_return = FactorDiagnostics.confounder_factor_return(row["x"], row["z"], beta, gamma, delta; delta_realized = row["realized"])
        @test factor_return isa FactorDiagnostics.StrategyPerformance
        check_factor_fields(factor_return, row["confounder_factor"])
        check_factor_fields(FactorDiagnostics.confounder_forecast_return(beta, gamma, delta; delta_realized = row["realized"]), row["confounder_forecast"])
        check_factor_fields(FactorDiagnostics.collider_factor_return(row["x"], row["z"], beta, gamma, delta), row["collider_factor"])
        check_factor_fields(FactorDiagnostics.collider_forecast_return(beta, gamma, delta), row["collider_forecast"])
        check_factor_fields(FactorDiagnostics.confounder_factor_return(row["x"], row["z"], beta, gamma, delta), row["confounder_factor_default"])
        check_factor_fields(FactorDiagnostics.confounder_forecast_return(beta, gamma, delta), row["confounder_forecast_default"])
        @test FactorDiagnostics.generalized_confounder_undercontrolled_coefficient(beta, gamma, delta; confounder_variance = 1, exposure_noise_variance = 1) == FactorDiagnostics.confounder_undercontrolled_coefficient(beta, gamma, delta)
        standardized = FactorDiagnostics.generalized_collider_overcontrolled_coefficients(beta, gamma, delta; outcome_noise_variance = 1, collider_noise_variance = 1)
        @test standardized.beta_hat == coefficients.beta_hat && standardized.theta_hat == coefficients.theta_hat
    end
    for row in factor_diagnostic_fixtures["diagnostics"]
        result = FactorDiagnostics.collider_model_diagnostics(row["beta"], row["gamma"], row["delta"], row["n"])
        @test result isa FactorDiagnostics.ColliderDiagnostics
        check_factor_fields(result, row["expected"])
    end
    @test !FactorDiagnostics.collider_model_diagnostics(0, 1, 0, 4).adjusted_r_squared_prefers_overcontrolled
    @test FactorDiagnostics.collider_model_diagnostics(0, nextfloat(1.0), 0, 4).adjusted_r_squared_prefers_overcontrolled
    @test !FactorDiagnostics.collider_model_diagnostics(0, prevfloat(1.0), 0, 4).adjusted_r_squared_prefers_overcontrolled
    @test !FactorDiagnostics.collider_model_diagnostics(0, 0, 0, 10).absolute_beta_t_prefers_overcontrolled
end

@testset "Factor diagnostic numerical and domain boundaries" begin
    for row in factor_diagnostic_fixtures["extreme"]
        parameters = (row["beta"], row["gamma"], row["delta"])
        check_factor_fields(FactorDiagnostics.collider_overcontrolled_coefficients(parameters...), row["collider"]; atol = 0.0)
        @test FactorDiagnostics.confounder_undercontrolled_coefficient(parameters...) ≈ row["confounder"] rtol = 1e-10 atol = 0.0
        check_factor_fields(FactorDiagnostics.collider_model_diagnostics(parameters..., 8), row["diagnostics"]; atol = 0.0)
    end
    zero_result = FactorDiagnostics.collider_model_diagnostics(0, 0, 0, big(10)^600)
    @test zero_result.correct_beta_variance == zero_result.correct_beta_t_statistic == zero_result.collider_t_statistic == 0
    @test FactorDiagnostics.collider_overcontrolled_coefficients(1e308, 1e308, 1e308).beta_hat == -1.0
    @test FactorDiagnostics.confounder_factor_return(1e200, -5e199, 1e200, 2e200, -1).correct == 0.0
    for function_name in (:confounder_undercontrolled_coefficient, :collider_overcontrolled_coefficients, :collider_forecast_return)
        method = getproperty(FactorDiagnostics, function_name)
        for invalid in (true, NaN, Inf, -Inf, "0", 1im)
            @test_throws ArgumentError method(invalid, 1, 1)
            @test_throws ArgumentError method(1, invalid, 1)
            @test_throws ArgumentError method(1, 1, invalid)
        end
    end
    for count in (3, 0, -1, true, 4.0, Inf)
        @test_throws ArgumentError FactorDiagnostics.collider_model_diagnostics(1, 1, 1, count)
    end
    for invalid in (0, -1, true, NaN, Inf)
        @test_throws ArgumentError FactorDiagnostics.generalized_confounder_undercontrolled_coefficient(1, 1, 1; confounder_variance = invalid, exposure_noise_variance = 1)
        @test_throws ArgumentError FactorDiagnostics.generalized_confounder_undercontrolled_coefficient(1, 1, 1; confounder_variance = 1, exposure_noise_variance = invalid)
        @test_throws ArgumentError FactorDiagnostics.generalized_collider_overcontrolled_coefficients(1, 1, 1; outcome_noise_variance = invalid, collider_noise_variance = 1)
        @test_throws ArgumentError FactorDiagnostics.generalized_collider_overcontrolled_coefficients(1, 1, 1; outcome_noise_variance = 1, collider_noise_variance = invalid)
    end
    @test_throws ArgumentError FactorDiagnostics.confounder_factor_return(1, 1, 1, 1, 1; delta_realized = NaN)
    @test_throws ArgumentError FactorDiagnostics.confounder_forecast_return(1, 1, 1; delta_realized = true)
    @test_throws ArgumentError FactorDiagnostics.collider_factor_return(1, NaN, 1, 1, 1)
    @test_throws ArgumentError FactorDiagnostics.collider_forecast_return(1e308, 0, 0)
    @test_throws ArgumentError FactorDiagnostics.collider_model_diagnostics(1e308, 0, 0, 5)
end

@testset "Public structural specification experiments with shared draws" begin
    for row in factor_diagnostic_fixtures["population"]
        result = getproperty(FactorDiagnostics, Symbol(row["structure"] * "_population_diagnostics"))()
        @test result isa FactorDiagnostics.SpecificationPopulationResult
        @test result.unconditioned isa FactorDiagnostics.PopulationRegressionDiagnostics
        check_factor_fields(result, row["expected"])
    end
    for row in factor_diagnostic_fixtures["experiment"]
        rng = SpecificationFixtureRNG([Vector{Float64}(values) for values in row["draws"]], 1)
        result = getproperty(FactorDiagnostics, Symbol(row["structure"] * "_specification_experiment"))(row["count"], 17; rng)
        @test result isa FactorDiagnostics.SpecificationExperimentResult
        @test result.unconditioned isa FactorDiagnostics.RegressionDiagnostics
        @test rng.position == length(row["draws"]) + 1
        check_factor_fields(result, row["expected"])
        @test result.unconditioned.coefficients isa Tuple && result.conditioned.coefficients isa Tuple
    end
    for method in (FactorDiagnostics.fork_specification_experiment, FactorDiagnostics.collider_specification_experiment, FactorDiagnostics.confounded_mediator_specification_experiment)
        first_result = method(100, 71)
        second_result = method(100, 71)
        @test first_result.unconditioned == second_result.unconditioned
        @test first_result.conditioned == second_result.conditioned
        @test first_result.conditioned.coefficients == method(100, 0; rng = MersenneTwister(71)).conditioned.coefficients
        @test first_result.conditioned.n_observations == 100
        @test all(isfinite, first_result.conditioned.standard_errors)
        for count in (3, 0, -1, true, 4.0, 1_000_001)
            @test_throws ArgumentError method(count)
        end
        for seed in (-1, 2^32, true, 0.0)
            @test_throws ArgumentError method(10, seed)
        end
        @test_throws ArgumentError method(10, 0; rng = 5)
    end
    constant_rng = SpecificationFixtureRNG([ones(5), ones(5), ones(5)], 1)
    @test_throws ArgumentError FactorDiagnostics.fork_specification_experiment(5; rng = constant_rng)
    @test_throws ArgumentError FactorDiagnostics._specification_ols(ones(5), (("X", collect(1.0:5.0)),))
    @test_throws ArgumentError FactorDiagnostics._specification_ols([1.0, NaN, 2.0, 3.0], (("X", collect(1.0:4.0)),))
end
