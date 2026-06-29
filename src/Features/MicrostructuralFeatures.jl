"""
Microstructural features — native Julia port mirroring the Python
`RiskLabAI.features.microstructural_features` API (López de Prado, AFML Ch. 19):
the Corwin–Schultz (2012) bid-ask spread estimator and the Bekker–Parkinson
volatility estimator.

Representation note (deliberate divergence): pandas Series become plain
`Vector`s; the rolling statistics replicate pandas' `rolling(window=w)` exactly —
a result is `NaN` for the first `w-1` points and whenever the window still
contains a `NaN` — so the warm-up `NaN` pattern and values match the Python
implementation exactly (verified in `test/runtests.jl`).

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 19;
Corwin, S. A., & Schultz, P. (2012), Journal of Finance 67(2).
"""

using Statistics: mean

# Constant 'd' from Corwin–Schultz (2012).
const _DENOMINATOR = 3 - 2 * sqrt(2)

# pandas rolling(window=w).agg: NaN for the first w-1 points and for any window
# that still contains a NaN; otherwise the aggregate.
function _rolling(x::AbstractVector{<:Real}, w::Integer, agg)
    n = length(x)
    out = fill(NaN, n)
    for i = w:n
        window = @view x[(i-w+1):i]
        any(isnan, window) || (out[i] = agg(window))
    end
    return out
end

"""
    beta_estimates(high_prices, low_prices, window_span) -> Vector{Float64}

Corwin–Schultz β: the `window_span`-average of the 2-day rolling sum of squared
high/low log-ratios. Mirrors Python's `beta_estimates`.
"""
function beta_estimates(
    high_prices::AbstractVector{<:Real},
    low_prices::AbstractVector{<:Real},
    window_span::Integer,
)
    log_ratios_sq = log.(high_prices ./ low_prices) .^ 2
    beta = _rolling(log_ratios_sq, 2, sum)
    return _rolling(beta, window_span, mean)
end

"""
    gamma_estimates(high_prices, low_prices) -> Vector{Float64}

Corwin–Schultz γ: squared log-ratio of the 2-day high max to the 2-day low min.
Mirrors Python's `gamma_estimates`.
"""
function gamma_estimates(
    high_prices::AbstractVector{<:Real},
    low_prices::AbstractVector{<:Real},
)
    high_max = _rolling(high_prices, 2, maximum)
    low_min = _rolling(low_prices, 2, minimum)
    return log.(high_max ./ low_min) .^ 2
end

"""
    alpha_estimates(beta, gamma) -> Vector{Float64}

Corwin–Schultz α from β and γ, floored at zero. Mirrors Python's
`alpha_estimates`.
"""
function alpha_estimates(beta::AbstractVector{<:Real}, gamma::AbstractVector{<:Real})
    term1 = ((sqrt(2) - 1) .* sqrt.(beta)) ./ _DENOMINATOR
    term2 = sqrt.(gamma ./ _DENOMINATOR)
    return max.(term1 .- term2, 0.0)
end

"""
    corwin_schultz_estimator(high_prices, low_prices, window_span=20) -> Vector{Float64}

Corwin–Schultz bid-ask spread `2(eᵅ-1)/(1+eᵅ)`. Mirrors Python's
`corwin_schultz_estimator`.
"""
function corwin_schultz_estimator(
    high_prices::AbstractVector{<:Real},
    low_prices::AbstractVector{<:Real},
    window_span::Integer = 20,
)
    beta = beta_estimates(high_prices, low_prices, window_span)
    gamma = gamma_estimates(high_prices, low_prices)
    alpha = alpha_estimates(beta, gamma)
    return 2 .* (exp.(alpha) .- 1) ./ (1 .+ exp.(alpha))
end

"""
    sigma_estimates(beta, gamma) -> Vector{Float64}

Bekker–Parkinson volatility σ from Corwin–Schultz β and γ, floored at zero.
Mirrors Python's `sigma_estimates`.
"""
function sigma_estimates(beta::AbstractVector{<:Real}, gamma::AbstractVector{<:Real})
    k2 = sqrt(8 / pi)
    term1 = ((sqrt(2) - 1) .* sqrt.(beta)) ./ _DENOMINATOR
    term2 = sqrt.(gamma ./ (k2^2 * _DENOMINATOR))
    return max.(term1 .+ term2, 0.0)
end

"""
    bekker_parkinson_volatility_estimates(high_prices, low_prices, window_span=20) -> Vector{Float64}

Bekker–Parkinson volatility from high/low prices via Corwin–Schultz β and γ.
Mirrors Python's `bekker_parkinson_volatility_estimates`.
"""
function bekker_parkinson_volatility_estimates(
    high_prices::AbstractVector{<:Real},
    low_prices::AbstractVector{<:Real},
    window_span::Integer = 20,
)
    beta = beta_estimates(high_prices, low_prices, window_span)
    gamma = gamma_estimates(high_prices, low_prices)
    return sigma_estimates(beta, gamma)
end

# --------------------------------------------------------------------------- #
# EDGE bid-ask spread estimator (Ardia–Guidotti–Kroencke 2024).
# --------------------------------------------------------------------------- #

# numpy `nanmean`/`nansum`: aggregate over the non-NaN entries (NaN if none).
function _nanmean(x::AbstractVector{<:Real})
    s = 0.0
    n = 0
    @inbounds for v in x
        if !isnan(v)
            s += v
            n += 1
        end
    end
    return n == 0 ? NaN : s / n
end

function _nansum(x::AbstractVector{<:Real})
    s = 0.0
    @inbounds for v in x
        isnan(v) || (s += v)
    end
    return s
end

# Shift by one period; the first element (no predecessor) becomes NaN.
_lag_nan(x::AbstractVector{<:Real}) = [NaN; x[1:(end-1)]]

"""
    edge_estimator(open_prices, high_prices, low_prices, close_prices; sign=false) -> Float64

EDGE effective bid-ask spread estimator (Ardia, Guidotti & Kroencke 2024). Pools
the open, high, low and close prices and corrects for discrete, infrequently
traded data, giving lower bias and variance than the close-to-close Roll (1984)
estimator and the two-day high-low Corwin–Schultz (2012) estimator, and never
returning an invalid (negative) point estimate. The result is a proportional
spread (`0.01` is a 1% spread). Returns `NaN` when the estimate is undefined
(fewer than three observations, fewer than two traded periods, or a degenerate /
zero-variance input). `sign=true` returns a signed estimate (negative when the
estimated squared spread is negative).

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer EDGE over Roll and Abdi-Ranaldo for low-frequency spread estimation in all
regimes; over Corwin-Schultz at small spreads (the edge narrows at very high
illiquidity and very large spreads).

Clean-room Julia port of the published algorithm, mirroring the validated Python
`RiskLabAI.features.microstructural_features.edge.edge_estimator` reference
(numeric parity asserted in `test/runtests.jl`). Admitted in Appraisal 03
(`library_extension/appraisals/03_verdict.md`; real-data confirmation a logged
follow-up pending an adequate public intraday / quote dataset).

Reference: Ardia, D., Guidotti, E., & Kroencke, T. A. (2024). Efficient
estimation of bid-ask spreads from open, high, low, and close prices. Journal of
Financial Economics, 161, 103916.
"""
function edge_estimator(
    open_prices::AbstractVector{<:Real},
    high_prices::AbstractVector{<:Real},
    low_prices::AbstractVector{<:Real},
    close_prices::AbstractVector{<:Real};
    sign::Bool = false,
)
    n = length(open_prices)
    if !(length(high_prices) == n && length(low_prices) == n && length(close_prices) == n)
        throw(ArgumentError("open, high, low, and close must have the same length."))
    end
    n < 3 && return NaN

    o = log.(float.(open_prices))
    h = log.(float.(high_prices))
    ll = log.(float.(low_prices))
    c = log.(float.(close_prices))
    m = (h .+ ll) ./ 2.0

    h1 = _lag_nan(h)
    l1 = _lag_nan(ll)
    c1 = _lag_nan(c)
    m1 = _lag_nan(m)

    # Log-returns; r1's first element is masked to align with the lagged quantities.
    r1 = m .- o
    r1[1] = NaN
    r2 = o .- m1
    r3 = m .- c1
    r4 = c1 .- m1
    r5 = o .- c1

    # Trade indicator: the bar traded if its range is non-zero or the low differs
    # from the previous close. NaN where any required input is missing.
    tau = fill(NaN, n)
    @inbounds for i = 1:n
        if !(isnan(h[i]) || isnan(ll[i]) || isnan(c1[i]))
            tau[i] = ((h[i] != ll[i]) || (ll[i] != c1[i])) ? 1.0 : 0.0
        end
    end

    # (tau and a != b) as 0/1, NaN where tau or any required input is missing.
    function _indicator(a, b)
        out = fill(NaN, n)
        @inbounds for i = 1:n
            if !(isnan(tau[i]) || isnan(a[i]) || isnan(b[i]))
                out[i] = (tau[i] == 1.0 && a[i] != b[i]) ? 1.0 : 0.0
            end
        end
        return out
    end

    po1 = _indicator(o, h)
    po2 = _indicator(o, ll)
    pc1 = _indicator(c1, h1)
    pc2 = _indicator(c1, l1)

    pt = _nanmean(tau)
    po = _nanmean(po1) + _nanmean(po2)
    pc = _nanmean(pc1) + _nanmean(pc2)

    (_nansum(tau) < 2 || po == 0.0 || pc == 0.0) && return NaN

    # De-meaned log-returns, weighted by the trade indicator and trade probability.
    d1 = r1 .- (_nanmean(r1) / pt) .* tau
    d3 = r3 .- (_nanmean(r3) / pt) .* tau
    d5 = r5 .- (_nanmean(r5) / pt) .* tau

    # Two unbiased squared-spread estimators, from the open and the previous close.
    x1 = (-4.0 / po) .* d1 .* r2 .+ (-4.0 / pc) .* d3 .* r4
    x2 = (-4.0 / po) .* d1 .* r5 .+ (-4.0 / pc) .* d5 .* r4

    e1 = _nanmean(x1)
    e2 = _nanmean(x2)
    v1 = _nanmean(x1 .* x1) - e1 * e1
    v2 = _nanmean(x2 .* x2) - e2 * e2

    # Variance-weighted (GMM-optimal) average of the two estimators; equal weight
    # if the total variance is not positive.
    total_variance = v1 + v2
    squared_spread =
        total_variance > 0.0 ? (v2 * e1 + v1 * e2) / total_variance : (e1 + e2) / 2.0

    spread = sqrt(abs(squared_spread))
    if sign && squared_spread < 0.0
        spread = -spread
    end
    return spread
end
