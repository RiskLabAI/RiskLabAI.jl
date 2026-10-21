using Test
using TOML
using RiskLabAI

micro_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isinf(right) ? left == right : isapprox(left, right; rtol = 1e-9, atol = 2e-12)
micro_equal(left::AbstractVector, right::AbstractVector) = length(left) == length(right) && all(micro_equal(a, b) for (a, b) in zip(left, right))

@testset "Shared rolling OHLC spread and volatility conventions" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "microstructure_python_320.toml"))
    for case in fixture["rolling"]
        high, low = Float64[case["high"]...], Float64[case["low"]...]
        original = deepcopy((high, low))
        beta, gamma = F.beta_estimates(high, low, case["width"]), F.gamma_estimates(high, low)
        @test micro_equal(beta, case["beta"])
        @test micro_equal(gamma, case["gamma"])
        @test micro_equal(F.alpha_estimates(beta, gamma), case["alpha"])
        @test micro_equal(F.sigma_estimates(beta, gamma), case["sigma"])
        @test micro_equal(F.corwin_schultz_estimator(high, low, case["width"]), case["spread"])
        @test micro_equal(F.bekker_parkinson_volatility_estimates(high, low, case["width"]), case["volatility"])
        @test isequal((high, low), original)
    end
    for case in fixture["moments"]
        @test micro_equal(only(F.alpha_estimates([case["beta"]], [case["gamma"]])), case["alpha"])
        @test micro_equal(only(F.sigma_estimates([case["beta"]], [case["gamma"]])), case["sigma"])
    end
end

@testset "Shared EDGE trade indicators, signed estimates and missing observations" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "microstructure_python_320.toml"))
    for case in fixture["edge"]
        inputs = Tuple(Float64[case[key]...] for key in ("open", "high", "low", "close"))
        original = deepcopy(inputs)
        @test micro_equal(F.edge_estimator(inputs...; sign = true), case["signed"])
        @test micro_equal(F.edge_estimator(inputs...), case["unsigned"])
        @test micro_equal(F.edge_estimator((values .* 8 for values in inputs)...; sign = true), case["signed"])
        @test isequal(inputs, original)
    end
    @test_throws ArgumentError F.edge_estimator([1., 2., 3.], [1., 2.], [1., 2., 3.], [1., 2., 3.])
end
