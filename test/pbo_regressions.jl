using Test
using TOML
using Random
using RiskLabAI

pbo_matrix(rows) = reduce(vcat, [permutedims(Vector{Float64}(row)) for row in rows])
pbo_mean(values, risk_free) = sum(values) / length(values) - risk_free
pbo_same(left, right) = length(left) == length(right) && all(isequal(a, b) || isapprox(a, b; rtol = 1e-10, atol = 2e-12, nans = true) for (a, b) in zip(left, right))

function pbo_rank_provider(records)
    position = Ref(0)
    provider = function (scores)
        position[] += 1
        record = records[position[]]
        @test pbo_same(scores, record["testing"])
        return Vector{Int}(record["order"])
    end
    return provider, position
end

@testset "PBO: recorded ordinal ranks and combinatorial paths" begin
    B = RiskLabAI.Backtest
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "pbo_python_320.toml"))
    for case in fixtures["evaluation"]
        train, test = pbo_matrix(case["train"]), pbo_matrix(case["test"])
        provider, calls = pbo_rank_provider([case["record"]])
        actual = B.performance_evaluation(train, test, size(train, 2), (values, rf) -> first(values)-rf, 0.; rank_order = provider)
        @test actual[1] == case["record"]["is_overfit"]
        @test actual[2] ≈ case["record"]["output"] atol = 1e-13
        @test calls[] == 1
    end
    for case in fixtures["plain"]
        values = pbo_matrix(case["values"])
        provider, calls = pbo_rank_provider(case["records"])
        result = B.probability_of_backtest_overfitting(values; n_partitions = case["partitions"], risk_free_return = .001, metric = pbo_mean, rank_order = provider)
        @test result[1] ≈ case["pbo"] atol = 1e-14
        @test result[2] ≈ Vector{Float64}(case["logits"]) atol = 1e-13
        @test calls[] == length(case["records"])
        if all(length(unique(record["testing"])) == length(record["testing"]) for record in case["records"])
            native = B.probability_of_backtest_overfitting(values; n_partitions = case["partitions"], risk_free_return = .001, metric = pbo_mean)
            @test native[1] == result[1]
            @test native[2] == result[2]
        end
    end
    counterexample = first(case for case in fixtures["evaluation"] if case["record"]["is_overfit"] != case["record"]["stable_overfit"])
    actual = B.performance_evaluation(pbo_matrix(counterexample["train"]), pbo_matrix(counterexample["test"]), 10, (values, rf) -> first(values), 0.)
    @test actual[1] == counterexample["record"]["stable_overfit"]
    @test actual[1] != counterexample["record"]["is_overfit"]
end

@testset "PBO: observed-sample regimes and adaptive weights" begin
    V = RiskLabAI.Validation
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "pbo_python_320.toml"))
    for case in fixtures["regimes"]
        window = case["window"] == -1 ? nothing : case["window"]
        actual = V.estimate_volatility_regimes(pbo_matrix(case["values"]); n_regimes = case["n_regimes"], window = window)
        @test actual == Vector{Int}(case["output"])
    end
    for case in fixtures["adaptive"]
        provider, calls = pbo_rank_provider(case["records"])
        actual = V.adaptive_probability_of_backtest_overfitting(pbo_matrix(case["values"]); n_partitions = case["partitions"], target_fraction = case["fraction"], n_regimes = case["n_regimes"], risk_free_return = .001, metric = pbo_mean, rank_order = provider)
        @test actual[1] ≈ case["output"] atol = 2e-13
        @test actual[2] == case["selected"]
        @test calls[] == length(case["records"])
    end
end

@testset "PBO: supplied circular blocks and rank replay" begin
    V = RiskLabAI.Validation
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "pbo_python_320.toml"))
    for case in fixtures["bagged"]
        provider, calls = pbo_rank_provider(case["records"])
        size = case["block_size"] == -1 ? nothing : case["block_size"]
        rows = [Vector{Int}(row) for row in case["rows"]]
        actual = V.bagged_probability_of_backtest_overfitting(pbo_matrix(case["values"]); n_partitions = case["partitions"], n_bag = case["repetitions"], block_size = size, risk_free_return = .001, metric = pbo_mean, bootstrap_rows = rows, rank_order = provider)
        @test isapprox(actual[1], case["output"]; atol = 1e-14, nans = true)
        @test actual[2] ≈ Vector{Float64}(case["pbos"]) atol = 1e-14
        @test calls[] == length(case["records"])
    end
    for case in fixtures["blocks"]
        actual = V.moving_block_bootstrap_indices(case["observations"], case["block_size"], MersenneTwister(1); starts = Vector{Int}(case["starts"]))
        @test actual == Vector{Int}(case["output"])
    end
end

@testset "PBO: rank and bootstrap contract boundaries" begin
    B, V = RiskLabAI.Backtest, RiskLabAI.Validation
    x = [1. 2. 3.; 2. 1. 4.]
    @test_throws ArgumentError B.performance_evaluation(x, x, 3, pbo_mean, 0.; rank_order = values -> [1, 1, 3])
    @test_throws ArgumentError B.performance_evaluation(x, x, 3, pbo_mean, 0.; rank_order = values -> [3, 2, 1])
    @test_throws ArgumentError B.probability_of_backtest_overfitting(x; n_partitions = 3)
    @test_throws ArgumentError V.moving_block_bootstrap_indices(5, 2, MersenneTwister(1); starts = [0, 2, 3])
    @test_throws DimensionMismatch V.bagged_probability_of_backtest_overfitting(x; n_bag = 2, bootstrap_rows = [[1, 2]])
    @test_throws ArgumentError V.bagged_probability_of_backtest_overfitting(x; n_bag = 1, bootstrap_rows = [[0, 2]])
end
