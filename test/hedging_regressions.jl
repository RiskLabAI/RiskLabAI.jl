using Test
using TOML
using Random
using Zygote
using RiskLabAI
using RiskLabAI.Backtest: gbm_hedging_paths, hedging_pnl, entropic_risk,
    black_scholes_call_delta, DeepHedge, train_deep_hedge

hedging_fixture_matrix(rows, type = Float64) = type.(permutedims(hcat(rows...)))

function hedging_fixture_parameters(encoded)
    names = (:weight1, :bias1, :weight2, :bias2, :weight3, :bias3)
    return NamedTuple{names}(Tuple(startswith(string(name), "weight") ? hedging_fixture_matrix(encoded[string(name)]) : Float64.(encoded[string(name)]) for name in names))
end

@testset "Supplied-draw GBM and zero-rate call deltas" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "hedging_python_320.toml"))
    for case in fixture["gbm"]
        output = gbm_hedging_paths(; (Symbol(key) => case[key] for key in ("spot", "maturity", "volatility", "drift", "steps", "paths", "seed"))...,
            normal_draws = hedging_fixture_matrix(case["draws"]))
        @test isapprox(output, hedging_fixture_matrix(case["expected"]); rtol = 1e-10, atol = 1e-12)
        @test output[:, 1] == fill(case["spot"], case["paths"])
        @test eltype(output) == Float64
    end
    for case in fixture["delta"]
        prices = hedging_fixture_matrix(case["prices"])
        settings = (strike = case["strike"], maturity = case["maturity"], volatility = case["volatility"])
        delta = black_scholes_call_delta(prices; settings...)
        @test isapprox(delta, hedging_fixture_matrix(case["expected"]); rtol = 1e-10, atol = 1e-12)
        @test all(value -> 0 <= value <= 1, delta)
        delta32 = black_scholes_call_delta(Float32.(prices); settings...)
        @test eltype(delta32) == Float32
        @test maximum(abs.(delta32 .- delta)) <= 8eps(Float32)
    end
end

@testset "Self-financing hedge cash flow and entropic value gradients" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "hedging_python_320.toml"))
    for case in fixture["pnl"]
        type = case["dtype"] == "float32" ? Float32 : Float64
        prices, positions = hedging_fixture_matrix(case["prices"], type), hedging_fixture_matrix(case["positions"], type)
        settings = (strike = case["strike"], cost_rate = case["cost_rate"], premium = case["premium"])
        pnl = hedging_pnl(prices, positions; settings...)
        gradient = only(Zygote.gradient(value -> sum(hedging_pnl(prices, value; settings...)), positions))
        absolute = type == Float64 ? 1e-12 : case["roundoff_bound"]
        relative = type == Float64 ? 1e-10 : 0.0
        @test all(isapprox.(pnl, case["expected"]; rtol = relative, atol = absolute))
        @test all(isapprox.(gradient, hedging_fixture_matrix(case["gradient"]); rtol = relative, atol = absolute))
        @test eltype(pnl) == type
        @test eltype(gradient) == type
    end
    for case in fixture["risk"]
        pnl = Float64.(case["pnl"])
        risk = entropic_risk(pnl, case["risk_aversion"])
        gradient = only(Zygote.gradient(value -> entropic_risk(value, case["risk_aversion"]), pnl))
        bound = max(1e-12, case["roundoff_bound"])
        @test abs(risk - case["expected"]) <= bound
        @test abs(risk - case["high_precision"]) <= bound
        @test isapprox(gradient, case["gradient"]; rtol = 1e-10, atol = 1e-12)
        @test isapprox(sum(gradient), -1; rtol = 1e-10, atol = 1e-12)
        @test abs(entropic_risk(pnl .+ 7, case["risk_aversion"]) - (risk - 7)) <= 2bound
    end
    @test isapprox(hedging_pnl([100.0 120.0], reshape([0.5], 1, 1); strike = 110, cost_rate = 0.01, premium = 5), [3.9]; rtol = 1e-10, atol = 1e-12)
    @test hedging_pnl([100.0 120.0], zeros(1, 1); strike = 110, cost_rate = 0.01) == [-10.0]
    @test only(Zygote.gradient(value -> sum(hedging_pnl([100.0 120.0], value; strike = 110, cost_rate = 0.01)), zeros(1, 1))) == reshape([20.0], 1, 1)
end

@testset "Shared-weight adapted neural hedge, gradients and Adam training" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "hedging_python_320.toml"))
    for case in fixture["networks"]
        parameters = hedging_fixture_parameters(case["parameters"])
        model = DeepHedge(case["hidden_size"]; parameters)
        prices = hedging_fixture_matrix(case["prices"])
        settings = (strike = case["strike"], maturity = case["maturity"])
        positions = model(prices; settings...)
        @test isapprox(positions, hedging_fixture_matrix(case["positions"]); rtol = 1e-10, atol = 1e-12)
        objective(values) = entropic_risk(hedging_pnl(prices,
            RiskLabAI.Backtest._deep_hedge_positions(values, prices, settings.strike, settings.maturity);
            strike = settings.strike, cost_rate = case["cost_rate"]), case["risk_aversion"])
        @test isapprox(objective(model.parameters), case["risk"]; rtol = 1e-10, atol = 1e-12)
        gradients = only(Zygote.gradient(objective, model.parameters))
        expected_gradients = hedging_fixture_parameters(case["gradients"])
        for name in keys(parameters)
            @test isapprox(getproperty(gradients, name), getproperty(expected_gradients, name); rtol = 1e-10, atol = 1e-12)
        end
        for time in 1:size(positions, 2)
            modified = copy(prices)
            modified[:, time+1:end] .*= 1.7
            @test model(modified; settings...)[:, 1:time] == positions[:, 1:time]
        end
        history = train_deep_hedge(model, prices; settings..., cost_rate = case["cost_rate"],
            risk_aversion = case["risk_aversion"], iterations = case["iterations"], learning_rate = case["learning_rate"])
        @test isapprox(history, case["history"]; rtol = 1e-10, atol = 1e-12)
        trained = hedging_fixture_parameters(case["trained_parameters"])
        for name in keys(parameters)
            @test isapprox(getproperty(model.parameters, name), getproperty(trained, name); rtol = 1e-10, atol = 1e-12)
        end
        evaluation = hedging_fixture_matrix(case["evaluation_prices"])
        before = map(copy, model.parameters)
        @test isapprox(model(evaluation; settings...), hedging_fixture_matrix(case["evaluation_positions"]); rtol = 1e-10, atol = 1e-12)
        @test model.parameters == before
        @test parameters == hedging_fixture_parameters(case["parameters"])
    end
end

@testset "Hedging domains, snapshots, local RNGs and numerical range" begin
    Random.seed!(751)
    expected = rand(5)
    Random.seed!(751)
    model = DeepHedge(3; seed = 15)
    paths = gbm_hedging_paths(; spot = 100, maturity = 1, volatility = 0.2, drift = 0, steps = 4, paths = 5, seed = 16)
    @test rand(5) == expected
    @test model.parameters == DeepHedge(3; seed = 15).parameters
    @test paths == gbm_hedging_paths(; spot = 100, maturity = 1, volatility = 0.2, drift = 0, steps = 4, paths = 5, seed = 16)
    @test all(value -> value >= 0, black_scholes_call_delta(paths; strike = 100, maturity = 1, volatility = 0.2))
    supplied = map(copy, model.parameters)
    snapshot = DeepHedge(3; parameters = supplied)
    supplied.weight1[1] = 100
    @test snapshot.parameters.weight1[1] != 100
    @test all(abs.(snapshot.parameters.weight1) .<= Float32(1 / sqrt(3)))
    @test all(abs.(snapshot.parameters.weight2) .<= Float32(1 / sqrt(3)))
    @test Float64.(Float32.(snapshot.parameters.bias3)) == snapshot.parameters.bias3
    for bad in (0, -1, true, 1.5)
        @test_throws ArgumentError DeepHedge(bad)
    end
    for bad in (true, 1.5, big(2)^65)
        @test_throws ArgumentError DeepHedge(3; seed = bad)
    end
    @test DeepHedge(1; seed = -1).hidden_size == 1
    @test_throws ArgumentError DeepHedge(3; parameters = merge(model.parameters, (weight1 = ones(2, 3),)))
    @test_throws ArgumentError DeepHedge(3; parameters = map(value -> Float32.(value), model.parameters))
    settings = (spot = 100., maturity = 1., volatility = 0.2, drift = 0., steps = 4, paths = 5, seed = 0)
    for bad in (0, -1, NaN, Inf, true)
        @test_throws ArgumentError gbm_hedging_paths(; settings..., spot = bad)
        @test_throws ArgumentError gbm_hedging_paths(; settings..., maturity = bad)
    end
    @test_throws ArgumentError gbm_hedging_paths(; settings..., volatility = -1)
    @test_throws ArgumentError gbm_hedging_paths(; settings..., drift = Inf)
    @test_throws ArgumentError gbm_hedging_paths(; settings..., normal_draws = ones(5, 3))
    @test_throws ArgumentError gbm_hedging_paths(; settings..., drift = 1000)
    @test_throws ArgumentError gbm_hedging_paths(; settings..., drift = -1000)
    @test_throws ArgumentError hedging_pnl(paths, ones(5, 3); strike = 100, cost_rate = 0)
    @test_throws ArgumentError hedging_pnl(paths, ones(Float32, 5, 4); strike = 100, cost_rate = 0)
    @test_throws ArgumentError hedging_pnl(zeros(5, 5), ones(5, 4); strike = 100, cost_rate = 0)
    @test_throws ArgumentError hedging_pnl(paths, ones(5, 4); strike = 100, cost_rate = -0.1)
    @test_throws ArgumentError hedging_pnl(paths, ones(5, 4); strike = 100, cost_rate = 0, premium = Inf)
    @test_throws ArgumentError entropic_risk(Int[1, 2], 1.0)
    @test_throws ArgumentError entropic_risk(Float64[], 1.0)
    for bad in (0, -1, true, NaN, Inf)
        @test_throws ArgumentError entropic_risk([1., 2.], bad)
    end
    @test_throws ArgumentError black_scholes_call_delta(paths; strike = 100, maturity = 1, volatility = 0)
    @test_throws ArgumentError model(Float32.(paths); strike = 100, maturity = 1)
    @test_throws ArgumentError model(paths[:, 1:1]; strike = 100, maturity = 1)
    @test_throws ArgumentError train_deep_hedge(model, paths; strike = 100, maturity = 1, cost_rate = 0, risk_aversion = 1, iterations = 0)
    @test_throws ArgumentError train_deep_hedge(model, paths; strike = 100, maturity = 1, cost_rate = 0, risk_aversion = 1, iterations = 1, learning_rate = 0)
end
