# Concrete imbalance bars: ExpectedImbalanceBars (EWMA-based E[T]) and
# FixedImbalanceBars (constant E[T]). Mirrors RiskLabAI.py
# data/structures/imbalance_bars.py. `using` centralized in `Data`.

@field_inherit ExpectedImbalanceBars{T<:Metric} ExpectedImbalanceBarsType{T} AbstractImbalanceBars{T} where {T<:Metric} begin
    expected_ticks_number_lower_bound::Float64
    expected_ticks_number_upper_bound::Float64
end

"""
    ExpectedImbalanceBars{T}(; ...) where {T<:Metric}

Imbalance bars whose E[T] is an EWMA of previous bars' tick counts, optionally
clamped to `expected_ticks_number_bounds`.
"""
function ExpectedImbalanceBars{T}(;
    bar_type::String,
    window_size_for_expected_n_ticks_estimation::Int,
    expected_imbalance_window::Int,
    initial_estimate_of_expected_n_ticks_in_bar::Float64,
    expected_ticks_number_bounds::Union{Tuple{Float64,Float64},Nothing} = nothing,
    does_analyse_thresholds::Bool = false,
) where {T<:Metric}
    base = AbstractImbalanceBars{T}(
        bar_type = bar_type,
        window_size_for_expected_n_ticks_estimation = window_size_for_expected_n_ticks_estimation,
        expected_imbalance_window = expected_imbalance_window,
        initial_estimate_of_expected_n_ticks_in_bar = initial_estimate_of_expected_n_ticks_in_bar,
        does_analyse_thresholds = does_analyse_thresholds,
    )

    lower, upper = isnothing(expected_ticks_number_bounds) ?
        (0.0, typemax(Float64)) : expected_ticks_number_bounds

    return ExpectedImbalanceBars{T}(values(base)..., lower, upper)
end

function expected_number_of_ticks(bars::ExpectedImbalanceBarsType)::Float64
    previous = bars.previous_bars_number_of_ticks
    isempty(previous) && return bars.expected_ticks_number  # no bars yet

    window = bars.window_size_for_expected_n_ticks_estimation
    expected = ewma(collect(Float64, last(previous, window)), window)[end]
    return min(
        max(expected, bars.expected_ticks_number_lower_bound),
        bars.expected_ticks_number_upper_bound,
    )
end

@field_inherit FixedImbalanceBars{T<:Metric} FixedImbalanceBarsType{T} AbstractImbalanceBars{T} where {T<:Metric} begin
end

"""
    FixedImbalanceBars{T}(; ...) where {T<:Metric}

Imbalance bars with a constant E[T] (the initial estimate).
"""
function FixedImbalanceBars{T}(;
    bar_type::String,
    expected_imbalance_window::Int,
    initial_estimate_of_expected_n_ticks_in_bar::Float64,
    window_size_for_expected_n_ticks_estimation::Union{Int,Nothing} = nothing,
    does_analyse_thresholds::Bool = false,
) where {T<:Metric}
    base = AbstractImbalanceBars{T}(
        bar_type = bar_type,
        window_size_for_expected_n_ticks_estimation = window_size_for_expected_n_ticks_estimation,
        expected_imbalance_window = expected_imbalance_window,
        initial_estimate_of_expected_n_ticks_in_bar = initial_estimate_of_expected_n_ticks_in_bar,
        does_analyse_thresholds = does_analyse_thresholds,
    )
    return FixedImbalanceBars{T}(values(base)...)
end

expected_number_of_ticks(bars::FixedImbalanceBarsType)::Float64 = bars.expected_ticks_number
