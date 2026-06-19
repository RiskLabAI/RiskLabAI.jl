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
