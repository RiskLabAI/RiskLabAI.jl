using Test
using TOML
using LinearAlgebra
using RiskLabAI

const PortfolioUnderTest = RiskLabAI.Optimization
portfolio_fixture_matrix(values) = reduce(vcat, permutedims.(values))

@testset "Constrained portfolios against Python 3.2.0 and scalar calculus" begin
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "portfolio_python_320.toml"))
    for row in fixtures["cases"]
        covariance = portfolio_fixture_matrix(row["covariance"])
        kwargs = Dict{Symbol, Any}(Symbol(name) => row[name] for name in ("caps", "return_floor") if haskey(row, name))
        result = if row["method"] == "constrained_minimum_variance"
            haskey(row, "expected_returns") && (kwargs[:expected_returns] = row["expected_returns"])
            PortfolioUnderTest.constrained_minimum_variance(covariance; kwargs...)
        else
            PortfolioUnderTest.robust_mean_variance(row["expected_returns"], covariance,
                portfolio_fixture_matrix(row["uncertainty_loading"]);
                radius = row["radius"], risk_aversion = row["risk_aversion"], kwargs...)
        end
        covariance_scale = maximum(abs, covariance)
        objective_scale = row["method"] == "constrained_minimum_variance" ? covariance_scale :
            max(covariance_scale * row["risk_aversion"], maximum(abs, row["expected_returns"]), row["radius"] * norm(portfolio_fixture_matrix(row["uncertainty_loading"])))
        @test result.solver_status == row["expected_solver_status"] == "optimal"
        @test all(abs.(result.weights - row["expected_weights"]) .<= 1e-6 .+ 1e-6 .* abs.(row["expected_weights"]))
        @test all(abs.(result.weights - row["independent_weights"]) .<= 1e-6 .+ 1e-6 .* abs.(row["independent_weights"]))
        mean = get(row, "expected_returns", zeros(length(result.weights)))
        loading = haskey(row, "uncertainty_loading") ? portfolio_fixture_matrix(row["uncertainty_loading"]) : zeros(length(result.weights), 1)
        radius = get(row, "radius", 0.0)
        weight_error_bound = 1e-6 .+ 1e-6 .* abs.(row["expected_weights"])
        variance_error_bound = opnorm(covariance, 2) * norm(weight_error_bound) * norm(result.weights + row["expected_weights"])
        mean_error_bound = dot(abs.(mean), weight_error_bound)
        worst_mean_error_bound = mean_error_bound + radius * opnorm(loading, 2) * norm(weight_error_bound)
        @test abs(result.variance - row["expected_variance"]) <= variance_error_bound + 1e-10 * covariance_scale
        @test abs(result.expected_return - row["expected_expected_return"]) <= mean_error_bound + 1e-10
        @test abs(result.worst_case_expected_return - row["expected_worst_case_expected_return"]) <= worst_mean_error_bound + 1e-10
        @test result.variance ≈ dot(result.weights, covariance * result.weights) rtol = 1e-12 atol = 1e-14 * covariance_scale
        @test result.expected_return ≈ dot(mean, result.weights) rtol = 1e-12 atol = 1e-14
        @test result.worst_case_expected_return ≈ dot(mean, result.weights) - radius * norm(loading' * result.weights) rtol = 1e-12 atol = 1e-14
        @test result.objective ≈ row["expected_objective"] rtol = 1e-8 atol = 1e-10 * objective_scale
        @test result.objective ≈ row["independent_objective"] rtol = 1e-8 atol = 1e-10 * objective_scale
        @test abs(sum(result.weights) - 1) <= 1e-7
        @test all(result.weights .>= -1e-7)
        @test all(result.weights .<= get(row, "caps", ones(length(result.weights))) .+ 1e-7)
        if haskey(row, "return_floor")
            @test dot(row["expected_returns"], result.weights) >= row["return_floor"] - 1e-7
        end
    end
end

@testset "Public-book robust portfolio conic formulation" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "portfolio_conic_book_cases.toml"))
    covariance = portfolio_fixture_matrix(fixture["covariance"])
    expected = fixture["expected_returns"]
    base_loading = portfolio_fixture_matrix(fixture["uncertainty_loading"])
    caps = fixture["caps"]
    for row in fixture["cases"]
        loading = row["zero_loading"] ? zero(base_loading) : base_loading
        floor = get(row, "return_floor", nothing)
        result = PortfolioUnderTest.robust_mean_variance(expected, covariance, loading;
            radius = row["radius"], risk_aversion = row["risk_aversion"], caps, return_floor = floor)
        @test result.solver_status == row["solver_status"]
        @test all(isapprox.(result.weights, row["expected_weights"]; rtol = 1e-9, atol = 1e-11))
        @test result.variance ≈ row["variance"] rtol = 1e-9 atol = 1e-11
        @test result.expected_return ≈ row["expected_return"] rtol = 1e-9 atol = 1e-11
        @test result.worst_case_expected_return ≈ row["worst_case_expected_return"] rtol = 1e-9 atol = 1e-11
        @test result.objective ≈ row["objective"] rtol = 1e-9 atol = 1e-11
        @test abs(sum(result.weights) - 1) <= 1e-7
        @test all(result.weights .>= -1e-7)
        @test all(result.weights .<= caps .+ 1e-7)
        @test isnothing(floor) || result.expected_return >= floor - 1e-7
    end
end

@testset "Constrained portfolio domains and nonunique optima" begin
    for covariance in (zeros(3, 3), ones(3, 3), [1.0 -1.0 0.0; -1.0 1.0 0.0; 0.0 0.0 0.0])
        result = PortfolioUnderTest.constrained_minimum_variance(covariance; caps = [0.4, 0.4, 0.4])
        @test result.solver_status == "optimal"
        @test abs(sum(result.weights) - 1) <= 1e-7
        @test all(-1e-7 .<= result.weights .<= 0.4 + 1e-7)
        @test result.variance ≈ (all(==(1.0), covariance) ? 1.0 : 0.0) atol = 1e-10
    end
    result = PortfolioUnderTest.constrained_minimum_variance(reshape([0.04], 1, 1); expected_returns = [0.08], return_floor = 0.07)
    @test result.weights ≈ [1.0] atol = 1e-8
    @test result.variance ≈ 0.04 atol = 1e-10
    diagonal = Diagonal([1.0, 2.0, 4.0])
    result = PortfolioUnderTest.constrained_minimum_variance(diagonal)
    @test result.weights ≈ [4 / 7, 2 / 7, 1 / 7] atol = 1e-7
    @test result.variance ≈ 4 / 7 atol = 1e-10
    identity_covariance = Matrix{Float64}(I, 2, 2)
    for covariance in (zeros(0, 0), ones(2, 3), [1.0 0.1; 0.2 1.0], [1.0 0.0; 0.0 -1e-8], [1.0 NaN; NaN 1.0], fill(true, 2, 2), ones(ComplexF64, 2, 2))
        @test_throws ArgumentError PortfolioUnderTest.constrained_minimum_variance(covariance)
    end
    @test_throws ArgumentError PortfolioUnderTest.constrained_minimum_variance(identity_covariance; caps = [0.4, 0.4])
    @test_throws ArgumentError PortfolioUnderTest.constrained_minimum_variance(identity_covariance; caps = [-0.1, 1.1])
    @test_throws ArgumentError PortfolioUnderTest.constrained_minimum_variance(identity_covariance; expected_returns = [0.1])
    @test_throws ArgumentError PortfolioUnderTest.constrained_minimum_variance(identity_covariance; return_floor = 0.1)
    @test_throws ArgumentError PortfolioUnderTest.constrained_minimum_variance(identity_covariance; expected_returns = [0.1, 0.15], return_floor = 0.2)
    @test_throws ArgumentError PortfolioUnderTest.robust_mean_variance([0.1, 0.2], identity_covariance, ones(1, 2); radius = 0.1, risk_aversion = 1.0)
    @test_throws ArgumentError PortfolioUnderTest.robust_mean_variance([0.1, 0.2], identity_covariance, zeros(2, 0); radius = 0.1, risk_aversion = 1.0)
    @test_throws ArgumentError PortfolioUnderTest.robust_mean_variance([0.1, 0.2], identity_covariance, identity_covariance; radius = -0.1, risk_aversion = 1.0)
    @test_throws ArgumentError PortfolioUnderTest.robust_mean_variance([0.1, 0.2], identity_covariance, identity_covariance; radius = 0.1, risk_aversion = -1.0)
    @test_throws ArgumentError PortfolioUnderTest.robust_mean_variance([0.1, 0.2], identity_covariance, identity_covariance; radius = true, risk_aversion = 1.0)
end
