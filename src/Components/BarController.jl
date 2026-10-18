const BAR_COLUMNS = [Utils.DATE_TIME, Utils.TICK_NUMBER, Utils.OPEN_PRICE,
    Utils.HIGH_PRICE, Utils.LOW_PRICE, Utils.CLOSE_PRICE, Utils.CUMULATIVE_VOLUME,
    Utils.CUMULATIVE_BUY_VOLUME, Utils.CUMULATIVE_SELL_VOLUME,
    Utils.CUMULATIVE_TICKS, Utils.CUMULATIVE_DOLLAR, Utils.THRESHOLD]

"""Native bar dispatch; caller-loaded tables keep filesystem operations explicit."""
struct BarController end

function bar_methods()
    information = [expectation*"_"*metric*"_"*family*"_bars" for family in ("imbalance", "run") for expectation in ("expected", "fixed") for metric in ("dollar", "volume", "tick")]
    return vcat(information, [metric*"_standard_bars" for metric in ("dollar", "volume", "tick")], ["time_bars"])
end

"""
Initialize any of the sixteen public bar routes with their controller defaults.
Public keyword names are translated to the native metric-dispatched constructors;
extra settings are ignored as in the public initializer.
"""
function initialize_bars(method::AbstractString;
    threshold = nothing,
    window_size_for_expected_n_ticks_estimation = 10000,
    window_size_for_expected_imbalance_estimation::Integer = 10000,
    initial_estimate_of_expected_n_ticks_in_bar::Real = 20000,
    expected_ticks_number_bounds = nothing, analyse_thresholds::Bool = false,
    resolution_type::AbstractString = "D", resolution_units::Integer = 1, kwargs...)
    method in bar_methods() || throw(KeyError("unknown bar method: $method"))
    method == "time_bars" && return Data.TimeBars(resolution_type = String(resolution_type), resolution_units = Int(resolution_units))
    metric_name = occursin("dollar", method) ? "dollar" : occursin("volume", method) ? "volume" : "tick"
    metric = metric_name == "dollar" ? Data.Dollar : metric_name == "volume" ? Data.Volume : Data.Tick
    if occursin("standard", method)
        default = metric_name == "dollar" ? 70000000. : metric_name == "volume" ? 30000. : 6000.
        bar_type = metric_name == "dollar" ? Utils.CUMULATIVE_DOLLAR : metric_name == "volume" ? Utils.CUMULATIVE_VOLUME : Utils.CUMULATIVE_TICKS
        return Data.StandardBars{metric}(bar_type = bar_type, threshold = Float64(threshold === nothing ? default : threshold))
    end
    imbalance = occursin("imbalance", method)
    expected = startswith(method, "expected")
    type = expected ? (imbalance ? Data.ExpectedImbalanceBars : Data.ExpectedRunBars) : (imbalance ? Data.FixedImbalanceBars : Data.FixedRunBars)
    common = (bar_type = metric_name*(imbalance ? "_imbalance" : "_run"),
        expected_imbalance_window = Int(window_size_for_expected_imbalance_estimation),
        initial_estimate_of_expected_n_ticks_in_bar = Float64(initial_estimate_of_expected_n_ticks_in_bar),
        does_analyse_thresholds = analyse_thresholds)
    if expected
        bounds = expected_ticks_number_bounds === nothing ? nothing : Tuple(Float64.(expected_ticks_number_bounds))
        return type{metric}(; common...,
            window_size_for_expected_n_ticks_estimation = window_size_for_expected_n_ticks_estimation,
            expected_ticks_number_bounds = bounds)
    end
    return type{metric}(; common...)
end

function _bar_dataframe(rows)
    isempty(rows) && return DataFrame([Any[] for _ in BAR_COLUMNS], BAR_COLUMNS)
    return DataFrame([getindex.(rows, column) for column in eachindex(BAR_COLUMNS)], BAR_COLUMNS)
end

"""
    process_bars(BarController(), method, arguments, data; batch_size=1000000)

Process chronological `(time, price, volume)` rows in batches. Return
`(bars, completed_input, processed_rows, error)` so an empty or partial result
does not imply successful input completion. Unknown routes and construction
errors return the standard bar columns and explicit failure status. CSV loading
and output writing belong to the caller; no implicit IO is performed.
"""
function process_bars(::BarController, method::AbstractString, arguments, data;
    batch_size::Integer = 1000000)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    rows = Any[]
    processed = 0
    error_message = nothing
    try
        settings = (; (Symbol(key) => value for (key, value) in pairs(arguments))...)
        generator = initialize_bars(method; settings...)
        values = data isa AbstractDataFrame ? Matrix(data) : data
        size(values, 2) == 3 || throw(DimensionMismatch("tick input must have time, price and volume columns"))
        for first_row in 1:batch_size:size(values, 1)
            last_row = min(first_row+batch_size-1, size(values, 1))
            append!(rows, Data.construct_bars_from_data(generator; data = values[first_row:last_row, :]))
            processed = last_row
        end
    catch failure
        failure isa InterruptException && rethrow()
        error_message = sprint(showerror, failure)
    end
    return (bars = _bar_dataframe(rows), completed_input = error_message === nothing,
        processed_rows = processed, error = error_message)
end
