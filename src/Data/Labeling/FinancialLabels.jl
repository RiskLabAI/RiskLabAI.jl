"""
Trend-scanning labels — native Julia port mirroring the Python
`RiskLabAI.data.labeling.financial_labels` API (López de Prado, AFML Ch. 4):
the OLS-slope t-value and the trend-scanning labeller.

The t-value is computed from closed-form OLS (slope / standard error), matching
SciPy's `linregress`; no regression package dependency is required.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 4.
"""

using DataFrames
using Dates
using Statistics: mean

"""
    calculate_t_value_linear_regression(prices) -> Float64

t-value of the slope of an OLS fit of `prices` against `0:n-1`. Returns `NaN`
for fewer than two points or a constant series, and `±Inf` for a perfect trend.
Mirrors Python's `calculate_t_value_linear_regression`.
"""
function calculate_t_value_linear_regression(prices::AbstractVector{<:Real})
    n = length(prices)
    n < 2 && return NaN
    x = 0:(n - 1)
    x_mean = mean(x)
    y_mean = mean(prices)
    sxx = sum((xi - x_mean)^2 for xi in x)
    sxy = sum((x[i] - x_mean) * (prices[i] - y_mean) for i in 1:n)
    syy = sum((p - y_mean)^2 for p in prices)
    slope = sxy / sxx
    sse = max(syy - slope * sxy, 0.0)
    std_err = sqrt(sse / (n - 2) / sxx)
    std_err == 0 && return slope != 0 ? sign(slope) * Inf : NaN
    return slope / std_err
end

"""
    find_trend_using_trend_scanning(molecule, close_index, close, span) -> DataFrame

For each event in `molecule`, scan forward over window lengths in
`span[1]:(span[2]-1)` and keep the window whose OLS t-value has the largest
magnitude. Returns a `DataFrame` with `event_start`, `end_time` (end of the
maximum window), `t_value`, and `trend` (the sign). Mirrors Python's
`find_trend_using_trend_scanning` (Snippet 4.1).
"""
function find_trend_using_trend_scanning(
    molecule::AbstractVector, close_index::AbstractVector, close::AbstractVector{<:Real}, span::Tuple{Integer,Integer}
)
    DT = eltype(close_index)
    results = DataFrame(
        event_start = DT[], end_time = DT[], t_value = Float64[], trend = Float64[]
    )
    (span[1] >= span[2] || span[1] < 2) && return results

    max_span_val = span[2] - 1
    position = Dict(t => i for (i, t) in enumerate(close_index))
    n = length(close_index)

    for idx in molecule
        haskey(position, idx) || continue
        loc = position[idx]
        loc + max_span_val > n && continue
        vertical_time = close_index[loc + max_span_val]

        best_t = 0.0
        best_abs = -1.0
        for s in span[1]:(span[2] - 1)
            window = close[loc:(loc + s - 1)]
            t = calculate_t_value_linear_regression(window)
            magnitude = isfinite(t) ? abs(t) : 0.0
            if magnitude > best_abs
                best_abs = magnitude
                best_t = t
            end
        end
        push!(results, (idx, vertical_time, best_t, sign(best_t)))
    end
    return results
end
