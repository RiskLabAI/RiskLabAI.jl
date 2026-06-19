"""
Strategy risk — native Julia port mirroring the Python
`RiskLabAI.backtest.strategy_risk` API (López de Prado, AFML Ch. 15): binomial
betting Sharpe ratio, implied precision / frequency, mixture-of-Gaussians bet
outcomes, and the probability of strategy failure.

The normal CDF comes from `Distributions` (matching SciPy's `norm.cdf`); the
deterministic metrics match the Python implementation exactly (verified in
`test/runtests.jl`).

Deliberate divergence: Python's `target_sharpe_ratio_symbolic` returns a SymPy
expression for the binomial variance; the Julia port returns the equivalent
closed form `p·(1-p)·(d-u)²` as a value of `(p, u, d)` (no SymPy dependency).
The Monte-Carlo helpers (`sharpe_ratio_trials`, `mix_gaussians`,
`calculate_strategy_risk`) take an optional `rng` keyword for reproducibility.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 15.
"""

"""
    sharpe_ratio_trials(p, n_run; rng=Random.default_rng()) -> (mean, std, sharpe)

Simulate `n_run` binomial bets (`+1` with probability `p`, else `-1`) and return
the mean, population std, and Sharpe ratio (`0.0` when the std is zero).
Stochastic. Mirrors Python's `sharpe_ratio_trials`.
"""
function sharpe_ratio_trials(
    p::Real,
    n_run::Integer;
    rng::AbstractRNG = Random.default_rng(),
)
    outcomes = [rand(rng) < p ? 1.0 : -1.0 for _ = 1:n_run]
    mean_outcome = mean(outcomes)
    std_outcome = std(outcomes; corrected = false)
    sharpe_ratio = std_outcome > 0 ? mean_outcome / std_outcome : 0.0
    return (mean_outcome, std_outcome, sharpe_ratio)
end

"""
    target_sharpe_ratio_symbolic(p, u, d) -> Float64

Closed-form variance of a binomial bet, `p·(1-p)·(d-u)²` (the factored form of
`E[X²] - E[X]²`). Python returns this as a SymPy expression; the Julia port
returns the value for given `(p, u, d)`. Mirrors Python's
`target_sharpe_ratio_symbolic`.
"""
target_sharpe_ratio_symbolic(p::Real, u::Real, d::Real) = p * (1 - p) * (d - u)^2

"""
    implied_precision(stop_loss, profit_taking, frequency, target_sharpe_ratio) -> Float64

Precision (win probability) implied by a target Sharpe ratio, from the binomial
SR quadratic. Returns `NaN` when the discriminant is negative (no real root).
Mirrors Python's `implied_precision`.
"""
function implied_precision(
    stop_loss::Real,
    profit_taking::Real,
    frequency::Real,
    target_sharpe_ratio::Real,
)
    spread = profit_taking - stop_loss
    a = (frequency + target_sharpe_ratio^2) * spread^2
    b = (2 * frequency * stop_loss - target_sharpe_ratio^2 * spread) * spread
    c = frequency * stop_loss^2
    discriminant = b^2 - 4 * a * c
    discriminant < 0 && return NaN
    return (-b + sqrt(discriminant)) / (2.0 * a)
end

"""
    bin_frequency(stop_loss, profit_taking, precision, target_sharpe_ratio) -> Float64

Bets-per-year frequency implied by a target Sharpe ratio at a given precision.
Returns `Inf` for a degenerate precision (`≤0` or `≥1`) or a zero denominator.
Mirrors Python's `bin_frequency`.
"""
function bin_frequency(
    stop_loss::Real,
    profit_taking::Real,
    precision::Real,
    target_sharpe_ratio::Real,
)
    (precision <= 0 || precision >= 1) && return Inf
    spread = profit_taking - stop_loss
    numerator = (target_sharpe_ratio * spread)^2 * precision * (1 - precision)
    denominator = (spread * precision + stop_loss)^2
    denominator == 0 && return Inf
    return numerator / denominator
end

"""
    binomial_sharpe_ratio(stop_loss, profit_taking, frequency, probability) -> Float64

Annualised Sharpe ratio of a binary-outcome strategy:
`(E[R]/σ[R])·√frequency`, with `E[R] = p·pt + (1-p)·sl` and
`σ[R] = (pt-sl)·√(p(1-p))`. A zero dispersion yields `0.0` (zero mean) or a
signed `Inf`. Mirrors Python's `binomial_sharpe_ratio`.
"""
function binomial_sharpe_ratio(
    stop_loss::Real,
    profit_taking::Real,
    frequency::Real,
    probability::Real,
)
    expected_return = profit_taking * probability + stop_loss * (1 - probability)
    stdev_return = (profit_taking - stop_loss) * sqrt(probability * (1 - probability))
    if stdev_return == 0
        return expected_return == 0 ? 0.0 : Inf * sign(expected_return)
    end
    return (expected_return / stdev_return) * sqrt(frequency)
end

"""
    mix_gaussians(mu1, mu2, sigma1, sigma2, probability, n_obs;
        rng=Random.default_rng()) -> Vector{Float64}

Mixture of two Gaussian bet outcomes: `⌊n_obs·probability⌋` draws from
`N(mu1, sigma1)` and the rest from `N(mu2, sigma2)`, shuffled. Stochastic.
Mirrors Python's `mix_gaussians`.
"""
function mix_gaussians(
    mu1::Real,
    mu2::Real,
    sigma1::Real,
    sigma2::Real,
    probability::Real,
    n_obs::Integer;
    rng::AbstractRNG = Random.default_rng(),
)
    n_obs1 = trunc(Int, n_obs * probability)
    n_obs2 = n_obs - n_obs1
    returns1 = mu1 .+ sigma1 .* randn(rng, n_obs1)
    returns2 = mu2 .+ sigma2 .* randn(rng, n_obs2)
    returns = vcat(returns1, returns2)
    shuffle!(rng, returns)
    return returns
end

"""
    failure_probability(returns, frequency, target_sharpe_ratio) -> Float64

Probability that the strategy fails: the normal-CDF Z-score comparing the
observed precision (share of positive returns) with the precision implied by the
target Sharpe ratio. Returns `0.0` when there are no positive or no negative
returns, and `1.0` when the target is unachievable (`NaN` required precision).
Mirrors Python's `failure_probability`.
"""
function failure_probability(
    returns::AbstractVector{<:Real},
    frequency::Real,
    target_sharpe_ratio::Real,
)
    positive_returns = returns[returns .> 0]
    negative_returns = returns[returns .<= 0]
    (isempty(positive_returns) || isempty(negative_returns)) && return 0.0

    profit_taking = mean(positive_returns)
    stop_loss = mean(negative_returns)
    observed_precision = length(positive_returns) / length(returns)
    required_precision =
        implied_precision(abs(stop_loss), profit_taking, frequency, target_sharpe_ratio)
    isnan(required_precision) && return 1.0

    p_var = observed_precision * (1 - observed_precision)
    if p_var == 0
        return observed_precision >= required_precision ? 0.0 : 1.0
    end
    p_std = sqrt(p_var / length(returns))
    z_score = (observed_precision - required_precision) / p_std
    return cdf(Normal(), z_score)
end

"""
    calculate_strategy_risk(mu1, mu2, sigma1, sigma2, probability, n_obs,
        frequency, target_sharpe_ratio; rng=Random.default_rng()) -> Float64

Simulate mixture-of-Gaussians bet outcomes and return the probability of
strategy failure. Stochastic. Mirrors Python's `calculate_strategy_risk`.
"""
function calculate_strategy_risk(
    mu1::Real,
    mu2::Real,
    sigma1::Real,
    sigma2::Real,
    probability::Real,
    n_obs::Integer,
    frequency::Real,
    target_sharpe_ratio::Real;
    rng::AbstractRNG = Random.default_rng(),
)
    returns = mix_gaussians(mu1, mu2, sigma1, sigma2, probability, n_obs; rng = rng)
    return failure_probability(returns, frequency, target_sharpe_ratio)
end
