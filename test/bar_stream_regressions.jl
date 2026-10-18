using Test
using TOML
using Dates
using RiskLabAI

bar_close(left, right) = isequal(left, right) || isapprox(left, right; atol = 1e-10, rtol = 2e-11, nans = true)
bar_matrix(rows) = isempty(rows) ? zeros(0, 12) : reduce(vcat, [permutedims(Vector{Float64}(row)) for row in rows])
bar_milliseconds(value::DateTime) = Dates.value(value - DateTime(1970))
bar_numeric(rows) = isempty(rows) ? zeros(0, 12) : reduce(vcat, [permutedims(Float64[bar_milliseconds(row[1]); row[2:end]]) for row in rows])

function bar_fixture_model(kind, metric, config)
    D, U = RiskLabAI.Data, RiskLabAI.Utils
    if kind == "TimeBars"
        return D.TimeBars(; resolution_type = config["resolution"], resolution_units = config["units"])
    end
    constructor = Core.apply_type(getfield(D, Symbol(kind)), getfield(D, Symbol(metric)))
    if kind == "StandardBars"
        key = metric == "Tick" ? U.CUMULATIVE_TICKS : metric == "Volume" ? U.CUMULATIVE_VOLUME : U.CUMULATIVE_DOLLAR
        return constructor(; bar_type = key, threshold = Float64(config["threshold"]))
    end
    window = config["tick_window"] == -1 ? nothing : Int(config["tick_window"])
    arguments = (bar_type = lowercase(metric) * (occursin("Imbalance", kind) ? "_imbalance" : "_run"), window_size_for_expected_n_ticks_estimation = window, expected_imbalance_window = Int(config["imbalance_window"]), initial_estimate_of_expected_n_ticks_in_bar = Float64(config["initial"]), does_analyse_thresholds = true)
    if config["expected"]
        return constructor(; arguments..., expected_ticks_number_bounds = Tuple(Float64.(config["bounds"])))
    end
    return constructor(; arguments...)
end

@testset "Bar streams: outputs, persistent batch state and threshold traces" begin
    D, U = RiskLabAI.Data, RiskLabAI.Utils
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "bars_python_320.toml"))
    for group in ("standard", "time", "information"), case in fixtures[group]
        model = bar_fixture_model(case["kind"], case["metric"], case["config"])
        data = Matrix{Any}(undef, length(case["times"]), 3)
        data[:, 1] = DateTime(1970) .+ Millisecond.(case["times"])
        data[:, 2] = case["prices"]
        data[:, 3] = case["volumes"]
        combined = zeros(0, 12)
        for batch in case["batches"]
            rows = D.construct_bars_from_data(model; data = data[(batch["left"]+1):batch["right"], :])
            numeric = bar_numeric(rows)
            @test bar_close(numeric, bar_matrix(batch["output"]))
            combined = vcat(combined, numeric)
            @test all(bar_close(Float64(getfield(model, Symbol(key))), Float64(value)) for (key, value) in batch["state"])
            for (key, value) in batch["histories"]
                @test bar_close(Vector{Float64}(getfield(model, Symbol(key))), Vector{Float64}(value))
            end
        end
        @test bar_close(combined, bar_matrix(case["output"]))
        @test model.tick_counter == length(case["times"])
        if group == "information"
            @test length(model.analyse_thresholds) == length(case["traces"])
            for (snapshot, trace) in zip(model.analyse_thresholds, case["traces"])
                actual = Float64[bar_milliseconds(snapshot["timestamp"]); [snapshot[key] for key in case["trace_columns"]]]
                @test bar_close(actual, Vector{Float64}(trace))
                @test snapshot[U.PREVIOUS_BARS_N_TICKS_LIST] === model.previous_bars_number_of_ticks
            end
            if startswith(case["kind"], "Fixed")
                @test model.window_size_for_expected_n_ticks_estimation === nothing
            end
            whole = bar_fixture_model(case["kind"], case["metric"], case["config"])
            @test bar_close(bar_numeric(D.construct_bars_from_data(whole; data = data)), combined)
        end
    end
end

@testset "Bar streams: expected-count fallback and disabled diagnostics" begin
    D = RiskLabAI.Data
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "bars_python_320.toml"))
    for case in fixtures["expected_ticks"]
        model = bar_fixture_model(case["kind"], "Tick", case["config"])
        append!(model.previous_bars_number_of_ticks, Int.(case["history"]))
        @test bar_close(D.expected_number_of_ticks(model), Float64(case["output"]))
    end
    for constructor in (D.FixedImbalanceBars{D.Tick}, D.FixedRunBars{D.Tick})
        model = constructor(; bar_type = "tick_imbalance", expected_imbalance_window = 8, initial_estimate_of_expected_n_ticks_in_bar = 3.)
        @test model.analyse_thresholds === nothing
        @test isempty(D.construct_bars_from_data(model; data = Matrix{Any}(undef, 0, 3)))
        @test model.tick_counter == 0
    end
    @test_throws ArgumentError D.TimeBars(; resolution_type = "year", resolution_units = 1)
end
