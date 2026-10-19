using Test
using TOML
using Random
using LinearAlgebra
using RiskLabAI

clustering_matrix(rows) = isempty(rows) ? zeros(0, 0) : reduce(vcat, [permutedims(Float64.(row)) for row in rows])

@testset "Clustering: supplied candidate selection and recursive refinement" begin
    C = RiskLabAI.Cluster
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "clustering_python_320.toml"))
    for mode in ("base", "top"), case in fixtures[mode]
        correlation = clustering_matrix(case["correlation"])
        preserved = copy(correlation)
        position = Ref(0)
        records = case["records"]
        provider = function (features, count, rng)
            position[] += 1
            row = records[position[]]
            @test count == row["k"]
            @test features ≈ clustering_matrix(row["features"]) atol = 2e-14 rtol = 2e-13
            return Int.(row["labels"])
        end
        method = mode == "base" ? C.cluster_k_means_base : C.cluster_k_means_top
        sorted_correlation, clusters, scores = method(correlation; max_clusters = case["maximum"], iterations = case["iterations"], label_provider = provider, random_state = 7)
        expected_clusters = Dict(row["key"] => Int.(row["items"]) for row in case["clusters"])
        @test clusters == expected_clusters
        @test scores ≈ Vector{Float64}(case["silhouette"]) atol = 2e-10 rtol = 2e-9
        order = reduce(vcat, (clusters[key] for key in sort(collect(keys(clusters)))); init = Int[])
        @test isequal(sorted_correlation, correlation[order, order])
        @test sort(order) == collect(1:size(correlation, 1))
        @test position[] == length(records)
        @test isequal(correlation, preserved)
    end
end

@testset "Clustering: recorded block and market primitives" begin
    C = RiskLabAI.Cluster
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "clustering_python_320.toml"))
    for case in fixtures["random"]
        blocks = [(common = Vector{Float64}(row["common"]), noise = clustering_matrix(row["noise"])) for row in case["draws"]]
        market = (common = Vector{Float64}(case["market"]["common"]), noise = clustering_matrix(case["market"]["noise"]))
        plan = (cuts = Vector{Int}(case["cuts"]), blocks = blocks, market = market)
        actual = C.random_block_correlation(case["columns"], case["blocks"]; block_size_min = case["minimum"], draws = plan)
        @test actual ≈ clustering_matrix(case["output"]) atol = 2e-12 rtol = 2e-11
        @test actual ≈ transpose(actual) atol = 1e-14
        @test diag(actual) == ones(case["columns"])
        @test minimum(eigvals(Symmetric(actual))) >= -1e-12
    end
end

@testset "Clustering: native seed isolation and failure boundaries" begin
    C = RiskLabAI.Cluster
    correlation = [1.0 0.8 0.1 0.1; 0.8 1.0 0.1 0.1; 0.1 0.1 1.0 0.6; 0.1 0.1 0.6 1.0]
    for method in (C.cluster_k_means_base, C.cluster_k_means_top)
        Random.seed!(193)
        expected = rand(5)
        Random.seed!(193)
        first = method(correlation; max_clusters = 3, iterations = 2, random_state = 15)
        @test rand(5) == expected
        second = method(correlation; max_clusters = 3, iterations = 2, random_state = 15)
        @test first == second
        @test sort(reduce(vcat, collect(values(first[2])))) == collect(1:4)
        @test all(isfinite, first[3])
    end
    @test_throws ArgumentError C.cluster_k_means_base(correlation; iterations = 0)
    @test_throws ArgumentError C.cluster_k_means_base(correlation; max_clusters = 1)
    @test_throws ArgumentError C.cluster_k_means_base(correlation; max_clusters = 4)
    @test_throws ArgumentError C.cluster_k_means_base(correlation; max_clusters = 3, label_provider = (x, k, rng) -> ones(Int, 4))
    @test_throws DimensionMismatch C.cluster_k_means_base(correlation; max_clusters = 3, label_provider = (x, k, rng) -> [0, 1])
    @test_throws ArgumentError C.cluster_k_means_base(ones(4, 4); max_clusters = 3, label_provider = (x, k, rng) -> [0, 0, 1, 1])
    @test_throws ArgumentError C.random_block_covariance(5, 3; block_size_min = 2)
    @test_throws ArgumentError C.random_block_covariance(5, 3; cuts = [1, 1])
    @test_throws DimensionMismatch C.random_covariance_sub(5, 2, .5; common_draws = zeros(4), idiosyncratic_draws = zeros(5, 2))
    @test_throws DimensionMismatch C.random_covariance_sub(5, 2, .5; common_draws = zeros(5), idiosyncratic_draws = zeros(2, 5))
end
