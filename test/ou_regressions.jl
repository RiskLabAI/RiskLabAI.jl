using Test
using TOML
using RiskLabAI

ou_equal(left, right) = isnan(right) ? isnan(left) : isinf(right) ? left == right : isapprox(left, right; rtol = 1e-9, atol = 2e-11)

@testset "Shared OU scale, first-passage and gain moments" begin
    B = RiskLabAI.Backtest
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "ou_python_320.toml"))
    for case in fixture["scales"]
        @test ou_equal(B.theta_from_half_life(case["half_life"]), case["theta"])
        @test ou_equal(B.stationary_std(case["theta"], case["sigma"]), case["stationary_std"])
    end
    for case in fixture["metrics"]
        theta, sigma, gap, pt, sl = (case[key] for key in ("theta", "sigma", "gap", "pt", "sl"))
        actual = B.ou_rule_metrics(pt, sl, theta, sigma, gap, case["cost"])
        for (key, value) in case["output"]
            @test ou_equal(getproperty(actual, Symbol(key)), value)
        end
        probability = B.hit_upper_probability(gap, pt, sl, theta, sigma)
        @test ou_equal(probability, case["output"]["hit_probability"])
        @test ou_equal(B.mean_exit_time(gap, pt, sl, theta, sigma), case["output"]["expected_holding_time"])
        @test ou_equal(probability+B.hit_upper_probability(-gap, sl, pt, theta, sigma), 1.)
        @test ou_equal(B.mean_exit_time(-gap, sl, pt, theta, sigma), actual.expected_holding_time)
    end
    for case in fixture["boundary"]
        actual = B.mean_exit_time(case["gap"], case["pt"], case["sl"], case["theta"], case["sigma"])
        @test isapprox(actual, case["time"]; rtol = 1e-4)
    end
    for case in fixture["invalid"]
        actual = B.ou_rule_metrics(case["pt"], case["sl"], .1, 1., .5)
        for (key, value) in case["output"]
            @test ou_equal(getproperty(actual, Symbol(key)), value)
        end
    end
    @test isnan(B.hit_upper_probability(1., -1., -1., .1, 1.))
    @test isnan(B.mean_exit_time(1., -1., -1., .1, 1.))
end

@testset "Shared OU least squares, finite filtering and persistence clipping" begin
    B = RiskLabAI.Backtest
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "ou_python_320.toml"))
    for case in fixture["fit"]
        values = Float64.(case["values"])
        original = copy(values)
        actual = B.fit_ornstein_uhlenbeck(values; dt = case["dt"])
        for (key, value) in case["output"]
            @test isapprox(getproperty(actual, Symbol(key)), value; rtol = 1e-8, atol = 2e-9)
        end
        @test isequal(values, original)
        @test 1e-6 <= actual.rho <= .999999
        @test isapprox(exp(-actual.theta*case["dt"]), actual.rho; rtol = 1e-12)
    end
end

@testset "OU optimizer objective and bound compatibility" begin
    B = RiskLabAI.Backtest
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "ou_python_320.toml"))
    for case in fixture["optimum"]
        lower, upper = case["bounds"]
        actual = B.optimal_ou_trading_rule(case["theta"], case["sigma"], case["gap"]; cost = case["cost"], bounds = (lower, upper))
        @test lower <= actual.profit_take <= upper
        @test lower <= actual.stop_loss <= upper
        recomputed = B.ou_rule_metrics(actual.profit_take, actual.stop_loss, case["theta"], case["sigma"], case["gap"], case["cost"])
        @test all(ou_equal(getproperty(actual, key), getproperty(recomputed, key)) for key in keys(recomputed))
        reference_rate = case["output"]["return_rate"]
        @test actual.return_rate >= reference_rate-1e-3*max(abs(reference_rate), 1e-3)
        println("OU numerical comparison: ", (theta = case["theta"], gap = case["gap"], cost = case["cost"], julia = (actual.profit_take, actual.stop_loss, actual.return_rate), python = (case["output"]["profit_take"], case["output"]["stop_loss"], reference_rate)))
    end
end
