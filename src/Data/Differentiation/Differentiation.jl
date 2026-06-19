"""
Fractional differentiation of time series — native Julia port mirroring the
Python `RiskLabAI.data.differentiation` API (López de Prado, AFML Ch. 5).

The Python package operates on `DataFrame`s (one column per series); the Julia
port operates on a single `AbstractVector` (the idiomatic unit), and callers map
over columns for the multi-series case. Function names mirror Python exactly.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 5.
"""

using Statistics: cor
using HypothesisTests: ADFTest, pvalue

"""
    calculate_weights_std(degree, size) -> Vector{Float64}

Weights for standard (expanding-window) fractional differentiation, ordered for
a dot product with the most recent observation last (`w₀` at the end), matching
Python's `calculate_weights_std` (Snippet 5.2).
"""
function calculate_weights_std(degree::Real, size::Integer)
    weights = Vector{Float64}(undef, size)
    weights[1] = 1.0
    for k in 1:(size - 1)
        weights[k + 1] = -weights[k] / k * (degree - k + 1)
    end
    return reverse(weights)
end

"""
    calculate_weights_ffd(degree, threshold=1e-5) -> Vector{Float64}

Weights for the fixed-width-window (FFD) method, generated until the magnitude
drops below `threshold`; ordered with `w₀` at the end (Snippet 5.3).
"""
function calculate_weights_ffd(degree::Real, threshold::Real=1e-5)
    weights = Float64[1.0]
    k = 1
    while true
        next = -weights[end] / k * (degree - k + 1)
        abs(next) < threshold && break
        push!(weights, next)
        k += 1
    end
    return reverse(weights)
end

"""
    fractional_difference_std(series, degree; threshold=0.01) -> Vector{Float64}

Standard (expanding-window) fractionally differentiated series. Returns a vector
the same length as `series`; the warm-up region (where the cumulative relative
weight loss is below `threshold`) is filled with `NaN`. Mirrors Python's
`fractional_difference_std` for a single series.
"""
function fractional_difference_std(
    series::AbstractVector{<:Real}, degree::Real; threshold::Real=0.01
)
    n = length(series)
    result = fill(NaN, n)
    n == 0 && return result

    weights = calculate_weights_std(degree, n)
    cumulative = cumsum(abs.(weights))
    cumulative ./= cumulative[end]
    skip = count(<(threshold), cumulative)

    weights_natural = reverse(weights)  # [w0, w1, ...]
    for k in (skip + 1):n
        acc = 0.0
        for m in 0:(k - 1)
            acc += weights_natural[m + 1] * series[k - m]
        end
        result[k] = acc
    end
    return result
end

"""
    fractional_difference_fixed(series, degree; threshold=1e-5) -> Vector{Float64}

Fixed-width-window (FFD) fractionally differentiated series. Returns a vector
the same length as `series`, with the first `width-1` entries (the warm-up)
filled with `NaN`. Mirrors Python's `fractional_difference_fixed_single`.
"""
function fractional_difference_fixed(
    series::AbstractVector{<:Real}, degree::Real; threshold::Real=1e-5
)
    weights = reverse(calculate_weights_ffd(degree, threshold))  # natural order
    width = length(weights)
    n = length(series)
    result = fill(NaN, n)
    n < width && return result

    for k in width:n
        acc = 0.0
        for m in 0:(width - 1)
            acc += weights[m + 1] * series[k - m]
        end
        result[k] = acc
    end
    return result
end

"""
    find_optimal_ffd(close_prices; p_value_threshold=0.05) -> NamedTuple

For `d` in 11 equal steps over [0, 1], FFD-differentiate `log(close_prices)`
(threshold 0.01), run the ADF test on the result, and report the statistics.
Returns columnar vectors (`d`, `adf_stat`, `p_value`, `correlation`). Mirrors
Python's `find_optimal_ffd_simple` (Snippet 5.4); ADF values come from
`HypothesisTests.ADFTest`, so they are implementation-defined rather than
bit-identical to statsmodels.
"""
function find_optimal_ffd(close_prices::AbstractVector{<:Real}; p_value_threshold::Real=0.05)
    log_prices = log.(close_prices)
    ds = range(0.0, 1.0; length=11)
    d_out = Float64[]
    adf_stat = Float64[]
    p_value = Float64[]
    correlation = Float64[]
    for d in ds
        differentiated = fractional_difference_fixed(log_prices, d; threshold=0.01)
        mask = .!isnan.(differentiated)
        sum(mask) < 3 && continue
        test = ADFTest(differentiated[mask], :constant, 1)
        push!(d_out, d)
        push!(adf_stat, test.stat)
        push!(p_value, pvalue(test))
        push!(correlation, cor(log_prices[mask], differentiated[mask]))
    end
    return (d=d_out, adf_stat=adf_stat, p_value=p_value, correlation=correlation)
end

"""
    fractionally_differentiated_log_price(prices; threshold=1e-5, step=0.01,
                                          p_value_threshold=0.05) -> Vector{Float64}

Increase `d` by `step` until the ADF test on the FFD-differentiated log-price
series rejects the unit-root null at `p_value_threshold`, then return that
series (same length as `prices`, with a `NaN` warm-up). Mirrors Python's
`fractionally_differentiated_log_price`.
"""
function fractionally_differentiated_log_price(
    prices::AbstractVector{<:Real};
    threshold::Real=1e-5,
    step::Real=0.01,
    p_value_threshold::Real=0.05,
)
    log_prices = log.(prices)
    degree = 0.0
    while true
        degree += step
        degree > 2.0 && throw(ErrorException("Failed to find stationary 'd' < 2.0"))
        differentiated = fractional_difference_fixed(log_prices, degree; threshold=threshold)
        mask = .!isnan.(differentiated)
        sum(mask) < 3 && continue
        if pvalue(ADFTest(differentiated[mask], :constant, 1)) <= p_value_threshold
            return differentiated
        end
    end
end
