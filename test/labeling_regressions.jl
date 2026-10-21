using Test
using TOML
using Dates
using DataFrames
using RiskLabAI

label_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isinf(right) ? left == right : isapprox(left, right; rtol = 1e-10, atol = 1e-12)
label_equal(left::AbstractVector, right::AbstractVector) = length(left) == length(right) && all(label_equal(a, b) for (a, b) in zip(left, right))
label_date(offset) = DateTime(2020, 1, 1)+Minute(offset)
label_ends(values) = Union{Missing,DateTime}[value == -99999 ? missing : label_date(value) for value in values]

@testset "Shared strict CUSUM boundaries and missing differences" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "labeling_python_320.toml"))
    for case in fixture["cusum"]
        values, thresholds = Float64[case["values"]...], Float64[case["dynamic"]...]
        index = label_date.(collect(0:(length(values)-1)))
        original = deepcopy((values, thresholds, index))
        @test D.symmetric_cusum_filter(index, values, case["threshold"]) == label_date.(Int[case["events"]...])
        @test D.cusum_filter_events_dynamic_threshold(index, values, thresholds) == label_date.(Int[case["dynamic_events"]...])
        @test isequal((values, thresholds, index), original)
    end
end

@testset "Shared daily lookbacks, debiased EWM and vertical dates" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "labeling_python_320.toml"))
    for case in fixture["volatility"]
        index, prices = label_date.(Int[case["minutes"]...]), Float64[case["prices"]...]
        original = deepcopy((index, prices))
        actual = D.daily_volatility_with_log_returns(index, prices; span = case["span"])
        @test actual.index == label_date.(Int[case["output_index"]...])
        @test label_equal(actual.volatility, case["output"])
        @test isequal((index, prices), original)
    end
    for case in fixture["vertical"]
        actual = D.vertical_barrier(label_date.(case["minutes"]), label_date.(case["events"]), case["days"])
        @test actual.event == label_date.(Int[case["output_events"]...])
        @test actual.barrier == label_date.(Int[case["output_barriers"]...])
    end
    @test_throws ArgumentError D._ewm_std([.1, .2], 0)
end

@testset "Shared first-touch barriers and next-price meta labels" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "labeling_python_320.toml"))
    for case in fixture["triple"]
        events = DataFrame(event_start = label_date.(case["starts"]), end_time = label_ends(case["ends"]), base_width = Float64.(case["widths"]))
        case["with_side"] && (events.side = Float64.(case["sides"]))
        index, prices = label_date.(case["minutes"]), Float64.(case["prices"])
        original = deepcopy((events, index, prices))
        actual = D.triple_barrier(index, prices, events, Tuple(Float64.(case["multipliers"])))
        @test isequal(actual, label_ends(case["output"]))
        @test isequal((events, index, prices), original)
    end
    for case in fixture["meta"]
        events = DataFrame(event_start = label_date.(case["starts"]), end_time = label_ends(case["ends"]))
        case["with_side"] && (events.side = Float64.(case["sides"]))
        index, prices = label_date.(case["minutes"]), Float64.(case["prices"])
        original = deepcopy((events, index, prices))
        actual = D.meta_labeling(events, index, prices)
        @test actual.event_start == label_date.(case["output_starts"])
        @test actual.end_time == label_date.(case["output_ends"])
        @test label_equal(actual.ret, case["returns"])
        @test label_equal(actual.label, case["labels"])
        @test ("side" in names(actual)) == case["with_side"]
        if case["with_side"]
            @test label_equal(actual.side, case["output_sides"])
        end
        @test isequal((events, index, prices), original)
    end
    index = label_date.([0, 10, 20])
    prices = [100., 102., 105.]
    events = D.meta_events(index, prices, index[1:2], (1., 1.), Dict(index[1] => .01, index[2] => .01), 0.; side = Dict(index[1] => 1.))
    @test events.end_time[1] == index[2]
    @test ismissing(events.end_time[2])
    @test isnan(events.side[2])
end
