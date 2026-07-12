"""
Probabilistic Sharpe Ratio — native Julia port mirroring the Python
`RiskLabAI.backtest.probabilistic_sharpe_ratio` API (López de Prado): the PSR
and the expected-maximum (benchmark) Sharpe ratio.

The normal CDF / quantile come from `Distributions` and match SciPy's
`norm.cdf` / `norm.ppf`, so the metrics match the Python implementation exactly
(verified in `test/runtests.jl`). `benchmark_sharpe_ratio` uses the
full-precision Euler–Mascheroni constant (`Base.MathConstants.eulergamma`,
matching `numpy.euler_gamma`).

Reference: De Prado, M. (2018), Advances in Financial Machine Learning.
"""

"""
    probabilistic_sharpe_ratio(observed_sharpe_ratio, benchmark_sharpe_ratio,
        number_of_returns; skewness_of_returns=0.0, kurtosis_of_returns=3.0,
        return_test_statistic=false) -> Float64

Probability that the observed Sharpe ratio exceeds the benchmark given the track
length and the return moments. The test statistic is
`Z = (SR̂ - SR*)·√(T-1) / √(1 - S·SR̂ + (K-1)/4·SR̂²)` and `PSR = Φ(Z)`. A
non-positive denominator returns `0.0` (or `-Inf` when `return_test_statistic`).
Mirrors Python's `probabilistic_sharpe_ratio`.
"""
function probabilistic_sharpe_ratio(
    observed_sharpe_ratio::Real,
    benchmark_sharpe_ratio::Real,
    number_of_returns::Integer;
    skewness_of_returns::Real = 0.0,
    kurtosis_of_returns::Real = 3.0,
    return_test_statistic::Bool = false,
)
    denominator =
        1 - skewness_of_returns * observed_sharpe_ratio +
        (kurtosis_of_returns - 1) / 4 * observed_sharpe_ratio^2
    if denominator <= 0
        return return_test_statistic ? -Inf : 0.0
    end
    test_statistic =
        (observed_sharpe_ratio - benchmark_sharpe_ratio) * sqrt(number_of_returns - 1) /
        sqrt(denominator)
    return return_test_statistic ? test_statistic : cdf(Normal(), test_statistic)
end

"""
    benchmark_sharpe_ratio(sharpe_ratio_estimates) -> Float64

Expected maximum Sharpe ratio across `N` trials, used as the PSR benchmark:
`σ_SR·[(1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))]`, where `σ_SR` is the population
std of the estimates and `γ` is the Euler–Mascheroni constant. Returns the mean
for a single estimate and `0.0` for none. Mirrors Python's
`benchmark_sharpe_ratio`.
"""
function benchmark_sharpe_ratio(sharpe_ratio_estimates::AbstractVector{<:Real})
    n = length(sharpe_ratio_estimates)
    n == 0 && return 0.0
    n == 1 && return float(mean(sharpe_ratio_estimates))
    standard_deviation = std(sharpe_ratio_estimates; corrected = false)
    γ = Base.MathConstants.eulergamma
    term1 = (1 - γ) * quantile(Normal(), 1 - 1 / n)
    term2 = γ * quantile(Normal(), 1 - 1 / (n * ℯ))
    return standard_deviation * (term1 + term2)
end

# --------------------------------------------------------------------------- #
# LPLZ HAC Sharpe inference (López de Prado–Lipton–Zoonekynd 2025).
#
# The sampling variance of the Sharpe estimator depends on the higher moments AND
# the serial correlation of returns. The PSR corrects the higher moments but
# assumes serial independence, so under autocorrelation its interval is too narrow
# and its test over-rejects. LPLZ takes the Newey–West (HAC) long-run variance of
# the Sharpe influence function, correcting for both at once; it converges to the
# PSR denominator (1 − S·SR̂ + (K−1)/4·SR̂²) under i.i.d. returns. Clean-room from
# the influence-function / HAC math; numeric parity asserted in `test/runtests.jl`.
# Admitted in Appraisal 08 (`library_extension/appraisals/08_verdict.md`).
# --------------------------------------------------------------------------- #

using Distributions: ccdf

# Per-period Sharpe ratio with the sample (ddof=1) standard deviation.
_sharpe_sample(r::AbstractVector{<:Real}) =
    (sd = std(r; corrected = true); sd > 0 ? mean(r) / sd : 0.0)

"""
    sharpe_ratio_influence_function(returns) -> Vector{Float64}

The Sharpe-ratio influence function `IFₜ = zₜ - ½·SR(zₜ² - 1)` with
`zₜ = (rₜ - μ)/σ` (mean ≈ 0). Mirrors Python's `sharpe_ratio_influence_function`.
"""
function sharpe_ratio_influence_function(returns::AbstractVector{<:Real})
    r = float.(returns)
    mu = mean(r)
    sigma = std(r; corrected = true)
    sigma == 0 && return zeros(length(r))
    z = (r .- mu) ./ sigma
    sr = mu / sigma
    return z .- 0.5 .* sr .* (z .^ 2 .- 1.0)
end

"""
    newey_west_long_run_variance(series, lag) -> Float64

Newey–West (Bartlett-kernel) long-run variance
`Ω̂ = γ₀ + 2 Σₖ₌₁ᴸ (1 - k/(L+1)) γₖ`, with `γₖ` the lag-`k` autocovariance; `lag=0`
gives the sample variance. Floored at `1e-12`. Mirrors Python's
`newey_west_long_run_variance`.
"""
function newey_west_long_run_variance(series::AbstractVector{<:Real}, lag::Integer)
    x = float.(series)
    x = x .- mean(x)
    t = length(x)
    t == 0 && return 1e-12
    total = (x' * x) / t
    for k = 1:lag
        weight = 1.0 - k / (lag + 1.0)
        total += 2.0 * weight * (x[(k+1):end]' * x[1:(end-k)]) / t
    end
    return max(total, 1e-12)
end

"""
    newey_west_automatic_lag(number_of_returns) -> Int

Newey–West automatic bandwidth `⌊4(T/100)^(2/9)⌋` (at least 1). Mirrors Python's
`newey_west_automatic_lag`.
"""
newey_west_automatic_lag(number_of_returns::Integer) =
    max(Int(floor(4.0 * (number_of_returns / 100.0)^(2.0 / 9.0))), 1)

"""
    lplz_sharpe_inference(returns; confidence_level=0.95, lag=nothing,
                          null_sharpe_ratio=0.0)

López de Prado–Lipton–Zoonekynd (2025) Sharpe-ratio inference (HAC of the
influence function). Returns a `NamedTuple` `(sharpe_ratio, standard_error,
confidence_interval, test_statistic, p_value, significant, lag)`. The standard
error is `√(Ω̂/T)` with `Ω̂` the Newey–West long-run variance of the Sharpe
influence function.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer LPLZ for Sharpe-ratio inference when returns show material serial
correlation and/or non-normality (estimable from the sample): it restores
near-nominal CI coverage and test size where the PSR under-covers and over-rejects
(PSR size ≈ 0.20 vs nominal 0.05 under AR(1)). It converges to the PSR on
near-normal iid returns with no over-coverage; the honest cost is wider intervals.
Lo (2002) is the autocorrelation-only intermediate, dominated by LPLZ under
non-normality.

Mirrors Python's `lplz_sharpe_inference`. Reference: López de Prado, Lipton &
Zoonekynd (2025); Newey & West (1987); Lo (2002).
"""
function lplz_sharpe_inference(
    returns::AbstractVector{<:Real};
    confidence_level::Real = 0.95,
    lag::Union{Integer,Nothing} = nothing,
    null_sharpe_ratio::Real = 0.0,
)
    r = float.(returns)
    t = length(r)
    sr = _sharpe_sample(r)
    if t < 3 || std(r; corrected = true) == 0
        return (
            sharpe_ratio = sr,
            standard_error = NaN,
            confidence_interval = (NaN, NaN),
            test_statistic = NaN,
            p_value = NaN,
            significant = false,
            lag = 0,
        )
    end
    L = lag === nothing ? newey_west_automatic_lag(t) : lag
    lrv = newey_west_long_run_variance(sharpe_ratio_influence_function(r), L)
    standard_error = sqrt(lrv / t)
    z = quantile(Normal(), 0.5 + confidence_level / 2.0)
    ci = (sr - z * standard_error, sr + z * standard_error)
    test_statistic = standard_error > 0 ? (sr - null_sharpe_ratio) / standard_error : NaN
    p_value = 2.0 * ccdf(Normal(), abs(test_statistic))
    return (
        sharpe_ratio = sr,
        standard_error = standard_error,
        confidence_interval = ci,
        test_statistic = test_statistic,
        p_value = p_value,
        significant = p_value < (1.0 - confidence_level),
        lag = L,
    )
end
