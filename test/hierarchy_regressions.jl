using Test
using TOML
using Random
using LinearAlgebra
using RiskLabAI

hierarchy_matrix(rows) = reduce(vcat, permutedims.([Float64[row...] for row in rows]))
hierarchy_equal(left, right) = isapprox(left, right; rtol = 1e-10, atol = 1e-12)

@testset "Shared inverse variance, cluster variance and tree leaves" begin
    O = RiskLabAI.Optimization
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "hierarchy_python_320.toml"))
    for case in fixture["ties"]
        distance = hierarchy_matrix(case["distance"])
        original = copy(distance)
        @test O._single_linkage_order(distance) == case["order"]
        @test distance == original
    end
    for case in fixture["hrp"]
        covariance, correlation, tree = hierarchy_matrix(case["covariance"]), hierarchy_matrix(case["correlation"]), hierarchy_matrix(case["linkage"])
        original = deepcopy((covariance, correlation, tree))
        @test hierarchy_equal(O.inverse_variance_weights(covariance), case["inverse"])
        @test hierarchy_equal(O.cluster_variance(covariance, case["selected"]), case["variance"])
        @test hierarchy_equal(O.distance_corr(correlation), hierarchy_matrix(case["distance"]))
        @test O.quasi_diagonal(tree) == case["order"]
        @test isequal((covariance, correlation, tree), original)
    end
end

@testset "Shared recursive allocation and complete single linkage portfolios" begin
    O = RiskLabAI.Optimization
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "hierarchy_python_320.toml"))
    for case in fixture["bisection"]
        covariance, order = hierarchy_matrix(case["covariance"]), case["order"]
        original = deepcopy((covariance, order))
        actual = O.recursive_bisection(covariance, order)
        @test hierarchy_equal(actual, case["output"])
        @test isapprox(sum(actual), 1; atol = 1e-12)
        @test all(>(0), actual)
        @test isequal((covariance, order), original)
    end
    for case in fixture["hrp"]
        covariance, correlation = hierarchy_matrix(case["covariance"]), hierarchy_matrix(case["correlation"])
        original = deepcopy((covariance, correlation))
        actual = O.hrp(covariance, correlation)
        @test hierarchy_equal(actual, case["output"])
        @test isapprox(sum(actual), 1; atol = 1e-12)
        @test all(>(0), actual)
        @test isequal((covariance, correlation), original)
    end
end

@testset "Shared supplied cluster NCO allocation" begin
    O = RiskLabAI.Optimization
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "hierarchy_python_320.toml"))
    for case in fixture["nco"]
        covariance = hierarchy_matrix(case["covariance"])
        clusters = Dict(index => Int[items...] for (index, items) in enumerate(case["clusters"]))
        target = case["with_mean"] ? case["mu"] : nothing
        original = deepcopy((covariance, clusters, target))
        actual = O.get_optimal_portfolio_weights_nco(covariance; mu = target, clusters = clusters)
        @test hierarchy_equal(actual, case["output"])
        @test isapprox(sum(actual), 1; atol = 1e-12)
        @test isequal((covariance, clusters, target), original)
        @test_throws ArgumentError O.get_optimal_portfolio_weights_nco(covariance; clusters = Dict(1 => [1, 1]))
    end
end
