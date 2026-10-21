using Test
using TOML
using Random
using Statistics
using LinearAlgebra
using RiskLabAI

algebra_equal(left::Number, right::Number; rtol = 2e-9, atol = 2e-11) = isnan(right) ? isnan(left) : isinf(right) ? left == right : isapprox(left, right; rtol, atol)
algebra_equal(left::AbstractArray, right::AbstractArray; kwargs...) = size(left) == size(right) && all(algebra_equal(a, b; kwargs...) for (a, b) in zip(left, right))
algebra_matrix(rows, columns) = isempty(rows) ? zeros(0, columns) : reduce(vcat, permutedims.([Float64[row...] for row in rows]))

@testset "Shared Marcenko Pastur grid and covariance conversions" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "covariance_algebra_python_320.toml"))
    for case in fixture["mp"]
        grid, density = D.marcenko_pastur_pdf(case["variance"], case["q"]; num_points = case["count"])
        @test algebra_equal(grid, Float64[case["grid"]...])
        @test algebra_equal(density, Float64[case["pdf"]...]; atol = 1e-7)
        @test all(>=(0), density)
    end
    for case in fixture["correlation"]
        covariance = algebra_matrix(case["covariance"], length(case["standard"]))
        original = copy(covariance)
        expected = algebra_matrix(case["correlation"], size(covariance, 2))
        @test algebra_equal(D.cov_to_corr(covariance), expected)
        @test algebra_equal(RiskLabAI.Cluster.covariance_to_correlation(covariance), expected)
        @test algebra_equal(D.corr_to_cov(expected, case["standard"]), algebra_matrix(case["recovered"], size(covariance, 2)))
        @test covariance == original
    end
end

@testset "Shared spectral reconstruction and noise eigenvalues" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "covariance_algebra_python_320.toml"))
    for case in fixture["pca"]
        matrix = algebra_matrix(case["matrix"], length(case["eigenvalues"]))
        original = copy(matrix)
        values, vectors = D.pca(matrix)
        @test algebra_equal(values, case["eigenvalues"])
        @test isapprox(vectors * Diagonal(values) * vectors', matrix; atol = 1e-10)
        @test isapprox(vectors' * vectors, I; atol = 1e-12)
        @test matrix == original
        @test issorted(values; rev = true)
    end
    for case in fixture["spectral"]
        values, vectors = case["eigenvalues"], algebra_matrix(case["eigenvectors"], length(case["eigenvalues"]))
        original = deepcopy((values, vectors))
        actual = D.denoised_corr(values, vectors, case["factors"])
        @test algebra_equal(actual, algebra_matrix(case["output"], length(values)))
        @test diag(actual) == ones(length(values))
        @test isapprox(actual, actual'; atol = 1e-12)
        @test minimum(eigvals(Symmetric(actual))) >= -1e-10
        @test isequal((values, vectors), original)
    end
end

@testset "Shared normalized covariance weights and PCA risk allocation" begin
    D, O = RiskLabAI.Data, RiskLabAI.Optimization
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "covariance_algebra_python_320.toml"))
    for case in fixture["weights"]
        covariance = algebra_matrix(case["covariance"], length(case["mu"]))
        target = case["with_mean"] ? case["mu"] : nothing
        weight_data = D.optimal_portfolio(covariance; mu = target)
        weight_nco = O.get_optimal_portfolio_weights(covariance; mu = target)
        @test algebra_equal(weight_data, case["output"])
        @test algebra_equal(weight_nco, case["output"])
        @test isapprox(sum(weight_data), 1; atol = 1e-11)
        @test isapprox(sum(weight_nco), 1; atol = 1e-11)
        @test algebra_equal(covariance * weight_data, case["mu"] .* (covariance * weight_data)[1] ./ case["mu"][1])
    end
    for case in fixture["hedging"]
        count = length(case["eigenvalues"])
        covariance = algebra_matrix(case["covariance"], count)
        expected_vectors = algebra_matrix(case["eigenvectors"], count)
        distribution = case["with_distribution"] ? case["distribution"] : nothing
        original = deepcopy((covariance, distribution))
        weights = O.pca_weights(covariance; risk_distribution = distribution, risk_target = case["target"])
        actual_risks = (expected_vectors' * weights) .^ 2 .* case["eigenvalues"]
        remaining = collect(1:count)
        while !isempty(remaining)
            reference = case["eigenvalues"][first(remaining)]
            group = [index for index in remaining if isapprox(case["eigenvalues"][index], reference; rtol = 1e-10)]
            @test isapprox(sum(actual_risks[group]), case["target"]^2 * sum(case["distribution"][group]); atol = 2e-10, rtol = 2e-9)
            remaining = setdiff(remaining, group)
        end
        @test isapprox(weights' * covariance * weights, case["target"]^2 * sum(case["distribution"]); atol = 2e-10)
        @test isequal((covariance, distribution), original)
    end
end

@testset "Shared NERCOME permutations and held out covariance" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "covariance_algebra_python_320.toml"))
    for case in fixture["nercome"]
        values = algebra_matrix(case["values"], length(first(case["values"])))
        permutations = isempty(case["permutations"]) ? zeros(Int, 0, size(values, 1)) : Int.(algebra_matrix(case["permutations"], size(values, 1)))
        original = deepcopy((values, permutations))
        actual = D.nercome_denoised_covariance(values; n_splits = case["splits"], split_fraction = case["fraction"], permutations = permutations)
        @test algebra_equal(actual, algebra_matrix(case["output"], size(values, 2)); rtol = 2e-8, atol = 2e-9)
        @test isapprox(actual, actual'; atol = 1e-10)
        @test isposdef(actual)
        @test isequal((values, permutations), original)
    end
    @test_throws ArgumentError D.nercome_denoised_covariance(ones(3, 2))
    @test_throws ArgumentError D.nercome_denoised_covariance(ones(6, 2); n_splits = 1, permutations = ones(Int, 1, 6))
    @test_throws ArgumentError D.nercome_denoised_covariance(ones(6, 2); n_splits = 2, permutations = permutedims(1:6))
end
