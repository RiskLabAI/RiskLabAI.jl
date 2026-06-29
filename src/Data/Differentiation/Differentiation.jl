"""
Fractional differentiation of time series — native Julia port mirroring the
Python `RiskLabAI.data.differentiation` API (López de Prado, AFML Ch. 5).

The Python package operates on `DataFrame`s (one column per series); the Julia
port operates on a single `AbstractVector` (the idiomatic unit), and callers map
over columns for the multi-series case. Function names mirror Python exactly.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 5.
"""

using Statistics: cor, mean, std
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
    for k = 1:(size-1)
        weights[k+1] = -weights[k] / k * (degree - k + 1)
    end
    return reverse(weights)
end

"""
    calculate_weights_ffd(degree, threshold=1e-5) -> Vector{Float64}

Weights for the fixed-width-window (FFD) method, generated until the magnitude
drops below `threshold`; ordered with `w₀` at the end (Snippet 5.3).
"""
function calculate_weights_ffd(degree::Real, threshold::Real = 1e-5)
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
    series::AbstractVector{<:Real},
    degree::Real;
    threshold::Real = 0.01,
)
    n = length(series)
    result = fill(NaN, n)
    n == 0 && return result

    weights = calculate_weights_std(degree, n)
    cumulative = cumsum(abs.(weights))
    cumulative ./= cumulative[end]
    skip = count(<(threshold), cumulative)

    weights_natural = reverse(weights)  # [w0, w1, ...]
    for k = (skip+1):n
        acc = 0.0
        for m = 0:(k-1)
            acc += weights_natural[m+1] * series[k-m]
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
    series::AbstractVector{<:Real},
    degree::Real;
    threshold::Real = 1e-5,
)
    weights = reverse(calculate_weights_ffd(degree, threshold))  # natural order
    width = length(weights)
    n = length(series)
    result = fill(NaN, n)
    n < width && return result

    for k = width:n
        acc = 0.0
        for m = 0:(width-1)
            acc += weights[m+1] * series[k-m]
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
function find_optimal_ffd(
    close_prices::AbstractVector{<:Real};
    p_value_threshold::Real = 0.05,
)
    log_prices = log.(close_prices)
    ds = range(0.0, 1.0; length = 11)
    d_out = Float64[]
    adf_stat = Float64[]
    p_value = Float64[]
    correlation = Float64[]
    for d in ds
        differentiated = fractional_difference_fixed(log_prices, d; threshold = 0.01)
        mask = .!isnan.(differentiated)
        sum(mask) < 3 && continue
        test = ADFTest(differentiated[mask], :constant, 1)
        push!(d_out, d)
        push!(adf_stat, test.stat)
        push!(p_value, pvalue(test))
        push!(correlation, cor(log_prices[mask], differentiated[mask]))
    end
    return (d = d_out, adf_stat = adf_stat, p_value = p_value, correlation = correlation)
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
    threshold::Real = 1e-5,
    step::Real = 0.01,
    p_value_threshold::Real = 0.05,
)
    log_prices = log.(prices)
    degree = 0.0
    while true
        degree += step
        degree > 2.0 && throw(ErrorException("Failed to find stationary 'd' < 2.0"))
        differentiated =
            fractional_difference_fixed(log_prices, degree; threshold = threshold)
        mask = .!isnan.(differentiated)
        sum(mask) < 3 && continue
        if pvalue(ADFTest(differentiated[mask], :constant, 1)) <= p_value_threshold
            return differentiated
        end
    end
end

# --------------------------------------------------------------------------- #
# Adaptive Fractional Differencing (AFD; IEEE Access 2025, clean-room
# approximation). de Prado's min-d-via-ADF is finite-sample biased and
# under-differences; AFD anchors the order on a bias-corrected Hurst-blend
# estimate of the increments with a CV-chosen FFD truncation, raised to the ADF
# stationarity boundary only if needed. Admitted in Appraisal 13
# (`library_extension/appraisals/13_verdict.md`; real-data predictive-lift
# confirmation a tracked obligation).
#
# Deliberate divergence: the wavelet-variance Hurst component is an OPTIONAL
# enhancement (`pywt` in Python). The Julia port carries no wavelet dependency, so
# `wavelet_variance_hurst` returns `NaN` and the blend uses the R/S estimate alone
# — exactly Python's pywt-absent fallback (the validated default blend collapses to
# this). The ADF p-values come from `HypothesisTests.ADFTest` (not statsmodels), so
# the order/threshold/p-value decisions are behavioural; `d_hat` (the R/S Hurst
# blend) and `rescaled_range_hurst` are parity-matched exactly in `test/runtests.jl`.
# --------------------------------------------------------------------------- #

const DEFAULT_DELTA_GRID = [round(i * 0.05; digits = 3) for i = 0:24]   # 0.00 .. 1.20
const DEFAULT_CV_THRESHOLDS = (1e-3, 1e-4, 1e-5)

# Least-squares slope of y on x (numpy polyfit degree 1).
_polyfit_slope(x, y) =
    (xm = mean(x); ym = mean(y); sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2))

"""
    wavelet_variance_hurst(series; wavelet="db2", skip_fine=1) -> Float64

Wavelet-variance Hurst estimate. Optional component (no wavelet dependency in the
Julia port): always returns `NaN`, so `adaptive_differencing_order` falls back to
the R/S estimate alone (Python's pywt-absent path). Mirrors Python's
`wavelet_variance_hurst` when `pywt` is unavailable.
"""
wavelet_variance_hurst(series; wavelet = "db2", skip_fine::Integer = 1) = NaN

"""
    rescaled_range_hurst(series) -> Float64

Classical rescaled-range (R/S) Hurst estimate: the log-log slope of R/S against
window size. Returns `NaN` for fewer than 32 observations. Mirrors Python's
`rescaled_range_hurst`.
"""
function rescaled_range_hurst(series::AbstractVector{<:Real})
    x = float.(series)
    n = length(x)
    n < 32 && return NaN
    raw = floor.(Int, 10.0 .^ range(log10(8), log10(n ÷ 2); length = 8))
    sizes = unique(raw)
    rs_means = Float64[]
    used = Int[]
    for s in sizes
        k = n ÷ s
        k < 1 && continue
        values = Float64[]
        for i = 0:(k-1)
            segment = x[(i*s+1):((i+1)*s)]
            m = mean(segment)
            deviations = cumsum(segment .- m)
            spread = maximum(deviations) - minimum(deviations)
            scale = sqrt(sum((segment .- m) .^ 2) / length(segment))   # population std
            scale > 0 && push!(values, spread / scale)
        end
        if !isempty(values)
            push!(rs_means, mean(values))
            push!(used, s)
        end
    end
    length(used) < 2 && return NaN
    return _polyfit_slope(log.(used), log.(rs_means))
end

"""
    adaptive_differencing_order(increments) -> Float64

Bias-corrected memory order `d̂` of a series of increments: a ridge-style blend of
the wavelet-variance and R/S Hurst estimates (`d̂ = H̄ - 0.5`, clipped to
`[0, 0.99]`). With no wavelet dependency the blend is the R/S estimate alone.
Mirrors Python's `adaptive_differencing_order`.
"""
function adaptive_differencing_order(increments::AbstractVector{<:Real})
    estimates = filter(isfinite, [wavelet_variance_hurst(increments), rescaled_range_hurst(increments)])
    isempty(estimates) && return NaN
    return clamp(mean(estimates) - 0.5, 0.0, 0.99)
end

function _adf_pvalue(values::AbstractVector{<:Real}, maxlag::Integer)
    v = filter(!isnan, values)
    (length(v) < 20 || isapprox(std(v; corrected = false), 0.0; atol = 1e-8)) && return 1.0
    try
        return pvalue(ADFTest(v, :constant, maxlag))
    catch
        return 1.0
    end
end

function _memory_retained(differenced::AbstractVector, level::AbstractVector)
    mask = (.!isnan.(differenced)) .& (.!isnan.(level))
    sum(mask) < 5 && return NaN
    d = differenced[mask]
    l = level[mask]
    (std(d; corrected = true) == 0 || std(l; corrected = true) == 0) && return NaN
    return abs(cor(d, l))
end

"""
    adaptive_fractional_difference(series; delta_grid=nothing,
        cv_thresholds=(1e-3,1e-4,1e-5), adf_significance=0.05, adf_maxlag=1)

Adaptive Fractional Differencing of a price/level series. Estimates the memory
order from the increments with the bias-corrected Hurst blend
(`adaptive_differencing_order`), anchors the differencing order at `0.5 + d̂`,
chooses the fixed-width-FFD weight-truncation threshold by cross-validation (the
threshold retaining the most memory subject to ADF stationarity), and raises the
order along `delta_grid` only as far as the ADF stationarity boundary if needed.
Returns a `NamedTuple` `(order, d_hat, adf_pvalue, memory_retained, threshold,
series)`.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer AFD over fixed-width FFD when the differencing order itself must be right —
strong long memory and finite samples, where min-d under-differences (order error
0.064 vs FFD 0.371); on weak memory the gap narrows. The implemented AFD is a
tractable clean-room approximation of the published wavelet-Hurst + ridge + CV
method.

Mirrors Python's `adaptive_fractional_difference`.
"""
function adaptive_fractional_difference(
    series::AbstractVector{<:Real};
    delta_grid::Union{Nothing,AbstractVector{<:Real}} = nothing,
    cv_thresholds = DEFAULT_CV_THRESHOLDS,
    adf_significance::Real = 0.05,
    adf_maxlag::Integer = 1,
)
    grid = delta_grid === nothing ? DEFAULT_DELTA_GRID : collect(delta_grid)
    level = float.(filter(!isnan, series))
    increments = diff(level)
    d_hat = adaptive_differencing_order(increments)
    if !isfinite(d_hat)
        return (
            order = NaN,
            d_hat = NaN,
            adf_pvalue = 1.0,
            memory_retained = NaN,
            threshold = NaN,
            series = Float64[],
        )
    end
    order_anchor = clamp(0.5 + d_hat, 0.0, maximum(grid))

    best_threshold = cv_thresholds[2]
    best = nothing
    for threshold in cv_thresholds
        diffv = fractional_difference_fixed(level, order_anchor; threshold = threshold)
        count(!isnan, diffv) < 20 && continue
        pv = _adf_pvalue(diffv, adf_maxlag)
        mem = _memory_retained(diffv, level)
        candidate = (pv < adf_significance, isfinite(mem) ? mem : -1.0, threshold, pv)
        if best === nothing || (candidate[1], candidate[2]) > (best[1], best[2])
            best = candidate
        end
    end
    best !== nothing && (best_threshold = best[3])

    order = order_anchor
    diffv = fractional_difference_fixed(level, order; threshold = best_threshold)
    pv = count(!isnan, diffv) >= 20 ? _adf_pvalue(diffv, adf_maxlag) : 1.0
    grid_up = [g for g in grid if g > order_anchor]
    gi = 1
    while pv >= adf_significance && gi <= length(grid_up)
        order = grid_up[gi]
        gi += 1
        diffv = fractional_difference_fixed(level, order; threshold = best_threshold)
        pv = count(!isnan, diffv) >= 20 ? _adf_pvalue(diffv, adf_maxlag) : 1.0
    end

    return (
        order = order,
        d_hat = d_hat,
        adf_pvalue = pv,
        memory_retained = _memory_retained(diffv, level),
        threshold = best_threshold,
        series = diffv,
    )
end
