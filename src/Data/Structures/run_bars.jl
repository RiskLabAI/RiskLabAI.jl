# Concrete run bars: ExpectedRunBars (EWMA-based E[T]) and FixedRunBars
# (constant E[T]). Mirrors RiskLabAI.py data/structures/run_bars.py.
# `using` centralized in `Data`.

@field_inherit ExpectedRunBars{T<:Metric} ExpectedRunBarsType{T} AbstractRunBars{
    T,
} where {T<:Metric} begin
    expected_ticks_number_lower_bound::Float64
    expected_ticks_number_upper_bound::Float64
end

"""
    ExpectedRunBars{T}(; ...) where {T<:Metric}

Run bars whose E[T] is an EWMA of previous bars' tick counts, optionally
clamped to `expected_ticks_number_bounds`.
"""
function ExpectedRunBars{T}(;
    bar_type::String,
    window_size_for_expected_n_ticks_estimation::Int,
    expected_imbalance_window::Int,
    initial_estimate_of_expected_n_ticks_in_bar::Float64,
    expected_ticks_number_bounds::Union{Tuple{Float64,Float64},Nothing} = nothing,
    does_analyse_thresholds::Bool = false,
) where {T<:Metric}
    base = AbstractRunBars{T}(
        bar_type = bar_type,
        window_size_for_expected_n_ticks_estimation = window_size_for_expected_n_ticks_estimation,
        expected_imbalance_window = expected_imbalance_window,
        initial_estimate_of_expected_n_ticks_in_bar = initial_estimate_of_expected_n_ticks_in_bar,
        does_analyse_thresholds = does_analyse_thresholds,
    )

    lower, upper =
        isnothing(expected_ticks_number_bounds) ? (0.0, typemax(Float64)) :
        expected_ticks_number_bounds

    return ExpectedRunBars{T}(values(base)..., lower, upper)
end

function expected_number_of_ticks(bars::ExpectedRunBarsType)::Float64
    previous = bars.previous_bars_number_of_ticks
    isempty(previous) && return bars.expected_ticks_number

    window = bars.window_size_for_expected_n_ticks_estimation
    expected = ewma(collect(Float64, last(previous, window)), window)[end]
    return min(
        max(expected, bars.expected_ticks_number_lower_bound),
        bars.expected_ticks_number_upper_bound,
    )
end

@field_inherit FixedRunBars{T<:Metric} FixedRunBarsType{T} AbstractRunBars{
    T,
} where {T<:Metric} begin end

"""
    FixedRunBars{T}(; ...) where {T<:Metric}

Run bars with a constant E[T] (the initial estimate).
"""
function FixedRunBars{T}(;
    bar_type::String,
    expected_imbalance_window::Int,
    initial_estimate_of_expected_n_ticks_in_bar::Float64,
    window_size_for_expected_n_ticks_estimation::Union{Int,Nothing} = nothing,
    does_analyse_thresholds::Bool = false,
) where {T<:Metric}
    base = AbstractRunBars{T}(
        bar_type = bar_type,
        window_size_for_expected_n_ticks_estimation = window_size_for_expected_n_ticks_estimation,
        expected_imbalance_window = expected_imbalance_window,
        initial_estimate_of_expected_n_ticks_in_bar = initial_estimate_of_expected_n_ticks_in_bar,
        does_analyse_thresholds = does_analyse_thresholds,
    )
    return FixedRunBars{T}(values(base)...)
end

expected_number_of_ticks(bars::FixedRunBarsType)::Float64 = bars.expected_ticks_number
