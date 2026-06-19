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
