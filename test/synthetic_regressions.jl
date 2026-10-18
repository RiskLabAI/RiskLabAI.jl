using Test
using TOML
using Random
using Statistics
using LinearAlgebra
using RiskLabAI

synthetic_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isinf(right) ? left == right : isapprox(left, right; rtol = 1e-11, atol = 1e-12)
synthetic_equal(left::AbstractArray, right::AbstractArray) = size(left) == size(right) && all(synthetic_equal(a, b) for (a, b) in zip(left, right))
synthetic_matrix(rows, columns) = isempty(rows) ? zeros(0, columns) : reduce(vcat, permutedims.([Float64[row...] for row in rows]))

@testset "Shared block covariances and supplied covariance draws" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "synthetic_python_320.toml"))
    for case in fixture["blocks"]
        actual = D.form_block_matrix(case["blocks"], case["size"], case["correlation"])
        @test synthetic_equal(actual, synthetic_matrix(case["output"], size(actual, 2)))
        @test issymmetric(actual)
        @test diag(actual) == ones(size(actual, 1))
    end
    for case in fixture["covariance"]
        normal, uniform = synthetic_matrix(case["normal"], case["factors"]), Float64.(case["uniform"])
        original = deepcopy((normal, uniform))
        actual = D.random_cov(case["columns"], case["factors"]; normal_draws = normal, uniform_draws = uniform)
        @test synthetic_equal(actual, synthetic_matrix(case["output"], case["columns"]))
        @test isposdef(actual)
        @test isequal((normal, uniform), original)
    end
    for case in fixture["true_matrix"]
        order, deviation, normal = case["order"], case["deviation"], case["normal"]
        original = deepcopy((order, deviation, normal))
        mu, covariance = D.form_true_matrix(case["blocks"], case["size"], case["correlation"]; permutation = order, standard_deviation_draws = deviation, normal_draws = normal)
        @test synthetic_equal(mu, case["mean"])
        @test synthetic_equal(covariance, synthetic_matrix(case["covariance"], length(mu)))
        @test isequal((order, deviation, normal), original)
    end
    @test_throws ArgumentError D.random_cov(2, 1; normal_draws = zeros(1, 2))
    @test_throws ArgumentError D.random_cov(2, 1; uniform_draws = [0.1, -1.])
    @test_throws ArgumentError D.form_true_matrix(2, 2, .5; permutation = [1, 1, 3, 4])
    @test_throws ArgumentError D.form_true_matrix(1, 2, .5; standard_deviation_draws = [.1, .4])
end

@testset "Shared drift profiles and parameter padding" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "synthetic_python_320.toml"))
    for case in fixture["burst"]
        drift, volatility = D.drift_volatility_burst(case["length"], .02, -.03, .1, .15, case["alpha"], case["beta"]; explosion_filter_width = case["width"])
        @test synthetic_equal(drift, Float64[case["drift"]...])
        @test synthetic_equal(volatility, Float64[case["volatility"]...])
    end
    for case in fixture["alignment"]
        parameters = deepcopy(case["parameters"])
        aligned, length = D.align_params_length(parameters)
        @test length == case["length"]
        @test Set(keys(aligned)) == Set(keys(case["output"]))
        for key in keys(aligned)
            @test synthetic_equal(aligned[key], Float64[case["output"][key]...])
        end
        @test isequal(parameters, case["parameters"])
    end
end

@testset "Shared Heston Merton draws and semidefinite Gaussian law" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "synthetic_python_320.toml"))
    negative_variances = 0
    for case in fixture["heston"]
        p = case["parameters"]
        gaussian, poisson, changes = synthetic_matrix(case["gaussian"], 3), case["poisson"], Bool.(case["changes"])
        original = deepcopy((p, gaussian, poisson, changes))
        args = (case["total_time"], case["length"], (p[key] for key in ("mu", "kappa", "theta", "xi", "rho", "lam", "m", "v"))..., changes)
        actual = D.heston_merton_log_returns(args...; gaussian_draws = gaussian, poisson_draws = poisson)
        @test synthetic_equal(actual, case["output"])
        direct = D.compute_log_returns(case["length"], p["mu"], p["kappa"], p["theta"], p["xi"], gaussian[:, 1], gaussian[:, 2], gaussian[:, 3], poisson, case["total_time"] / case["length"], sqrt(case["total_time"] / case["length"]), p["lam"], p["m"], p["v"], changes)
        @test synthetic_equal(direct, case["output"])
        native = D.heston_merton_log_returns(args...; random_state = 82)
        @test all(isfinite, native)
        @test isequal(native, D.heston_merton_log_returns(args...; random_state = 82))
        @test isequal((p, gaussian, poisson, changes), original)
        @test_throws ArgumentError D.heston_merton_log_returns(args...; gaussian_draws = zeros(case["length"], 2))
        @test_throws ArgumentError D.heston_merton_log_returns(args...; poisson_draws = fill(.5, case["length"]))
        negative_variances += count(<(0), case["variance"])
    end
    @test negative_variances > 0
    primitives = sqrt(3.) .* vcat(Matrix{Float64}(I, 3, 3), -Matrix{Float64}(I, 3, 3))
    for rho in (-1., -.4, 0., .8, 1.), dispersion in (0., -.2, .3)
        outcomes = reduce(vcat, [permutedims(collect(D._heston_merton_gaussian(rho, .07, dispersion, row))) for row in eachrow(primitives)])
        @test isapprox(vec(mean(outcomes; dims = 1)), [0., 0., .07]; atol = 1e-14)
        @test isapprox(cov(outcomes; corrected = false), [1. rho 0.; rho 1. 0.; 0. 0. dispersion^2]; atol = 1e-14)
    end
end

@testset "Shared Markov boundaries, regime expansion and missing prices" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "synthetic_python_320.toml"))
    for case in fixture["categorical"]
        @test D._sample_categorical(MersenneTwister(1), case["probabilities"]; uniform = case["uniform"]) == case["output"]
    end
    missing_prices, missing_drifts = 0, 0
    for case in fixture["regimes"]
        transition = synthetic_matrix(case["transition"], 3)
        gaussian = synthetic_matrix(case["gaussian"], 3)
        original = deepcopy(case)
        states = D._markov_simulate(transition, case["length"], 3; rng = MersenneTwister(1), initial_state = case["initial"], transition_uniforms = case["uniforms"])
        @test states == case["states"]
        prices, path = D.generate_prices_from_regimes(case["regimes"], transition, .8, case["length"]; regime_order = case["order"], initial_state = case["initial"], transition_uniforms = case["uniforms"], gaussian_draws = gaussian, poisson_draws = case["poisson"])
        @test path == case["path"]
        @test synthetic_equal(prices, case["prices"])
        @test isequal(case, original)
        @test_throws ArgumentError D.generate_prices_from_regimes(case["regimes"], transition, .8, case["length"]; regime_order = ["growth", "stress", "growth"])
        missing_prices += count(isnan, prices)
        missing_drifts += count(isnan, case["parameters"]["mu"])
    end
    @test missing_prices > 0
    @test missing_drifts > missing_prices
    @test_throws ArgumentError D._markov_simulate([.2 .2; .5 .5], 3, 2; rng = MersenneTwister(1))
    @test_throws ArgumentError D._markov_simulate([.5 .5; .5 .5], 3, 2; rng = MersenneTwister(1), initial_state = 0)
    @test_throws ArgumentError D._markov_simulate([.5 .5; .5 .5], 3, 2; rng = MersenneTwister(1), transition_uniforms = [1., .2])
end
