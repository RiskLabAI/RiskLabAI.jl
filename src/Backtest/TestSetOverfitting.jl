"""
Test-set overfitting — native Julia port mirroring the Python
`RiskLabAI.backtest.test_set_overfitting` API (López de Prado): the expected
maximum Sharpe ratio, its sampling error, the Sharpe-ratio Z-statistic, and the
multiple-testing type-1 / type-2 error probabilities.

The normal CDF / quantile come from `Distributions` (matching SciPy's
`norm.cdf` / `norm.ppf`); the deterministic metrics match the Python
implementation exactly (verified in `test/runtests.jl`). Note that
`expected_max_sharpe_ratio` uses the truncated Euler constant `0.5772156649`
exactly as the Python source does. `generate_max_sharpe_ratios` and
`mean_std_error` are Monte-Carlo helpers (stochastic; an optional `rng` keyword
is provided for reproducibility).

Reference: De Prado, M. (2018), Advances in Financial Machine Learning.
"""

"""
    expected_max_sharpe_ratio(n_trials, mean_sharpe_ratio, std_sharpe_ratio) -> Float64

Expected maximum of `n_trials` Sharpe ratios:
`μ + σ·[(1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-(N·e)⁻¹)]`. Returns `0.0` for zero trials and
`μ` for one. Mirrors Python's `expected_max_sharpe_ratio`.
"""
function expected_max_sharpe_ratio(
    n_trials::Integer,
    mean_sharpe_ratio::Real,
    std_sharpe_ratio::Real,
)
    n_trials == 0 && return 0.0
    n_trials == 1 && return float(mean_sharpe_ratio)
    euler_gamma = 0.5772156649   # truncated constant, matching the Python source
    term1 = (1 - euler_gamma) * quantile(Normal(), 1.0 - 1.0 / n_trials)
    term2 = euler_gamma * quantile(Normal(), 1.0 - (n_trials * ℯ)^-1)
    return mean_sharpe_ratio + std_sharpe_ratio * (term1 + term2)
end

"""
    generate_max_sharpe_ratios(n_sims, n_trials_list, std_sharpe_ratio, mean_sharpe_ratio;
        rng=Random.default_rng()) -> DataFrame

Monte-Carlo maximum Sharpe ratios: for each `n_trials`, draw `n_sims × n_trials`
standard-normal rows, z-score each row (population std), rescale to
`(mean, std)`, and take the row maxima. Long-format `DataFrame` with columns
`max_SR`, `n_trials`. Stochastic. Mirrors Python's `generate_max_sharpe_ratios`.
"""
function generate_max_sharpe_ratios(
    n_sims::Integer,
    n_trials_list::AbstractVector{<:Integer},
    std_sharpe_ratio::Real,
    mean_sharpe_ratio::Real;
    rng::AbstractRNG = Random.default_rng(),
)
    output = DataFrame(max_SR = Float64[], n_trials = Int[])
    for n_trials in n_trials_list
        sims = randn(rng, n_sims, n_trials)
        for i = 1:n_sims
            row = @view sims[i, :]
            row .= (row .- mean(row)) ./ std(row; corrected = false)
            row .= mean_sharpe_ratio .+ row .* std_sharpe_ratio
            push!(output, (maximum(row), n_trials))
        end
    end
    return output
end

"""
    mean_std_error(n_sims0, n_sims1, n_trials; std_sharpe_ratio=1.0,
        mean_sharpe_ratio=0.0, rng=Random.default_rng()) -> DataFrame

Mean and standard deviation of the relative error between the analytical
`expected_max_sharpe_ratio` and the Monte-Carlo average of `n_sims0` maxima,
over `n_sims1` repetitions. `DataFrame` with columns `n_trials`, `meanErr`,
`stdErr`. Stochastic. Mirrors Python's `mean_std_error`.
"""
function mean_std_error(
    n_sims0::Integer,
    n_sims1::Integer,
    n_trials::AbstractVector{<:Integer};
    std_sharpe_ratio::Real = 1.0,
    mean_sharpe_ratio::Real = 0.0,
    rng::AbstractRNG = Random.default_rng(),
)
    expected =
        Dict(n => expected_max_sharpe_ratio(n, mean_sharpe_ratio, std_sharpe_ratio) for n in n_trials)
    errors = Dict(n => Float64[] for n in n_trials)
    for _ = 1:Int(n_sims1)
        simulated = generate_max_sharpe_ratios(
            n_sims0,
            n_trials,
            std_sharpe_ratio,
            mean_sharpe_ratio;
            rng = rng,
        )
        averaged = combine(groupby(simulated, :n_trials), :max_SR => mean => :avg)
        for row in eachrow(averaged)
            push!(errors[row.n_trials], row.avg / expected[row.n_trials] - 1.0)
        end
    end
    output = DataFrame(n_trials = Int[], meanErr = Float64[], stdErr = Float64[])
    for n in n_trials
        push!(output, (n, mean(errors[n]), std(errors[n])))
    end
    return output
end

"""
    estimated_sharpe_ratio_z_statistics(sharpe_ratio, t; true_sharpe_ratio=0.0,
        skew=0.0, kurt=3) -> Float64

Sharpe-ratio Z-statistic `(SR̂ - SR₀)·√(T-1) / √(1 - S·SR̂ + (K-1)/4·SR̂²)`.
Returns `NaN` on a non-positive denominator. Mirrors Python's
`estimated_sharpe_ratio_z_statistics`.
"""
function estimated_sharpe_ratio_z_statistics(
    sharpe_ratio::Real,
    t::Integer;
    true_sharpe_ratio::Real = 0.0,
    skew::Real = 0.0,
    kurt::Real = 3,
)
    denominator = 1 - skew * sharpe_ratio + (kurt - 1) / 4.0 * sharpe_ratio^2
    denominator <= 0 && return NaN
    return (sharpe_ratio - true_sharpe_ratio) * sqrt(t - 1) / sqrt(denominator)
end

"""
    strategy_type1_error_probability(z; k=1) -> Float64

Family-wise type-1 error `1 - (1 - Φ(-z))^k` over `k` independent tests.
Mirrors Python's `strategy_type1_error_probability`.
"""
function strategy_type1_error_probability(z::Real; k::Integer = 1)
    alpha_single_test = cdf(Normal(), -z)
    return 1 - (1 - alpha_single_test)^k
end

"""
    theta_for_type2_error(sharpe_ratio, t, true_sharpe_ratio; skew=0.0, kurt=3) -> Float64

The `θ = SR_true·√(T-1) / √(1 - S·SR̂ + (K-1)/4·SR̂²)` parameter used in the
type-2 error. Returns `NaN` on a non-positive denominator. Mirrors Python's
`theta_for_type2_error`.
"""
function theta_for_type2_error(
    sharpe_ratio::Real,
    t::Integer,
    true_sharpe_ratio::Real;
    skew::Real = 0.0,
    kurt::Real = 3,
)
    denominator = 1 - skew * sharpe_ratio + (kurt - 1) / 4.0 * sharpe_ratio^2
    denominator <= 0 && return NaN
    return true_sharpe_ratio * sqrt(t - 1) / sqrt(denominator)
end

"""
    strategy_type2_error_probability(alpha_k, k, theta) -> Float64

Type-2 error `β = Φ(Φ⁻¹((1 - α_k)^{1/k}) - θ)`. Mirrors Python's
`strategy_type2_error_probability`.
"""
function strategy_type2_error_probability(alpha_k::Real, k::Integer, theta::Real)
    z_alpha = quantile(Normal(), (1 - alpha_k)^(1.0 / k))
    return cdf(Normal(), z_alpha - theta)
end
