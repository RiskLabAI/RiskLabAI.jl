using Test
using TOML
using RiskLabAI

signal_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isinf(right) ? left == right : isapprox(left, right; rtol = 1e-11, atol = 1e-13)
signal_equal(left::AbstractArray, right::AbstractArray) = size(left) == size(right) && all(signal_equal(a, b) for (a, b) in zip(left, right))
signal_ends(values) = Union{Int,Missing}[value == -1 ? missing : Int(value) for value in values]

@testset "Shared probability sizes and inclusive bet intervals" begin
    B = RiskLabAI.Backtest
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "signal_sizing_python_320.toml"))
    for case in fixture["probability"]
        @test signal_equal(B.probability_bet_size(Float64[case["probabilities"]...], Float64[case["sides"]...]), Float64[case["output"]...])
    end
    for case in fixture["inclusive"]
        prices, starts, ends = Int.(case["prices"]), Int.(case["starts"]), Int.(case["ends"])
        signals, sides, probabilities = Float64.(case["signals"]), Float64.(case["sides"]), Float64.(case["probabilities"])
        original = deepcopy((prices, starts, ends, signals, sides, probabilities))
        @test signal_equal(B.average_bet_sizes(prices, starts, ends, signals), Float64.(case["output"]))
        @test signal_equal(B.strategy_bet_sizing(prices, starts, ends, sides, probabilities), Float64.(case["strategy"]))
        @test (prices, starts, ends, signals, sides, probabilities) == original
    end
    @test B.average_bet_sizes(0:3, Int[], Int[], Float64[]) == zeros(4)
end

@testset "Shared half-open signal averaging and discretized meta-signals" begin
    B = RiskLabAI.Backtest
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "signal_sizing_python_320.toml"))
    for case in fixture["active"]
        starts, ends, values = Int.(case["starts"]), signal_ends(case["ends"]), Float64.(case["signals"])
        original = deepcopy((starts, ends, values))
        points, actual = B.avg_active_signals(starts, ends, values)
        @test points == case["points"]
        @test signal_equal(actual, Float64.(case["output"]))
        @test signal_equal(B.mp_avg_active_signals(starts, ends, values, reverse(points)), reverse(Float64.(case["output"])))
        @test isequal((starts, ends, values), original)
    end
    for case in fixture["generated"]
        starts, ends = Int.(case["starts"]), signal_ends(case["ends"])
        sides = case["meta"] ? Float64.(case["sides"]) : nothing
        probability, prediction = Float64.(case["probabilities"]), Float64.(case["predictions"])
        original = deepcopy((starts, ends, sides, probability, prediction))
        points, actual = B.generate_signal(starts, ends, sides, probability, prediction, case["classes"], case["step"])
        @test points == case["points"]
        @test signal_equal(actual, Float64.(case["output"]))
        @test all(-1 .<= actual .<= 1)
        @test isequal((starts, ends, sides, probability, prediction), original)
    end
    for case in fixture["discrete"]
        @test signal_equal(B.discrete_signal(Float64.(case["values"]), case["step"]), Float64.(case["output"]))
    end
    @test B.generate_signal(Int[], Union{Int,Missing}[], nothing, Float64[], Float64[], 2, .1) == (Int[], Float64[])
    @test B.avg_active_signals(Int[], Union{Int,Missing}[], Float64[]) == (Int[], Float64[])
end

@testset "Shared sigmoid positions, inverses and public limit-price ranges" begin
    B = RiskLabAI.Backtest
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "signal_sizing_python_320.toml"))
    for case in fixture["sigmoid"]
        @test signal_equal(B.bet_size_sigmoid(case["width"], case["divergence"]), case["output"])
        @test B.target_position(case["width"], 100. + case["divergence"], 100., case["maximum"]) == case["position"]
    end
    for case in fixture["inverse"]
        value = B.inverse_price(case["forecast"], case["width"], case["size"])
        @test signal_equal(value, case["output"])
        if abs(case["size"]) < 1
            @test isapprox(B.bet_size_sigmoid(case["width"], case["forecast"]-value), case["size"]; rtol = 1e-10, atol = 1e-12)
        end
    end
    for case in fixture["width"]
        @test signal_equal(B.compute_sigmoid_width(case["divergence"], case["size"]), case["output"])
    end
    for case in fixture["limit"]
        @test signal_equal(B.limit_price(case["target"], case["current"], case["forecast"], case["width"], case["maximum"]), case["output"])
    end
end
