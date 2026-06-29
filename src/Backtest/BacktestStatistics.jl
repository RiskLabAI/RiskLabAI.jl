"""
Backtest statistics — native Julia port mirroring the Python
`RiskLabAI.backtest.backtest_statistics` API (López de Prado, AFML Ch. 14):
Sharpe ratio, bet timing, average holding period, Herfindahl–Hirschman
concentration, and drawdown / time-under-water.

Representation note (deliberate divergence): the Python API takes time-indexed
pandas Series. The Julia port passes the series as parallel sorted vectors
`(index::AbstractVector, values::AbstractVector)` (timestamps are `DateTime`),
and returns `DataFrame`s / `NamedTuple`s. The numerics match the Python
implementation exactly (verified in `test/runtests.jl`); in particular
`sharpe_ratio` uses the population standard deviation (`numpy.std`, `ddof=0`).

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 14.
"""

"""
    sharpe_ratio(returns; risk_free_rate=0.0) -> Float64

Sharpe ratio `mean(r - rf) / std(r - rf)` using the **population** standard
deviation (`ddof=0`, matching `numpy.std`). Returns `0.0` when the dispersion is
zero. Mirrors Python's `sharpe_ratio`.
"""
function sharpe_ratio(returns::AbstractVector{<:Real}; risk_free_rate::Real = 0.0)
    excess = returns .- risk_free_rate
    dispersion = std(excess; corrected = false)
    dispersion == 0.0 && return 0.0
    return mean(excess) / dispersion
end

"""
    bet_timing(index, target_positions) -> Vector

Timestamps at which bets are realised: positions that return to zero from a
non-zero prior, positions that flip sign, and always the final timestamp.
Mirrors Python's `bet_timing`.
"""
function bet_timing(index::AbstractVector, target_positions::AbstractVector{<:Real})
    n = length(target_positions)
    bets = Set{eltype(index)}()
    # Position closes: zero now, non-zero on the (zero-filled) previous bar.
    for i = 1:n
        prev = i == 1 ? zero(eltype(target_positions)) : target_positions[i-1]
        if target_positions[i] == 0 && prev != 0
            push!(bets, index[i])
        end
    end
    # Sign flips: recorded at the later timestamp.
    for i = 2:n
        if target_positions[i] * target_positions[i-1] < 0
            push!(bets, index[i])
        end
    end
    result = sort(collect(bets))
    if index[n] ∉ result
        push!(result, index[n])
    end
    return result
end

"""
    calculate_holding_period(index, target_positions) -> (holding_periods, mean_holding_period)

Average holding period (in days) via the average-entry-time pairing algorithm.
Returns a `DataFrame` with columns `index`, `dT` (holding time), `w` (weight),
and the weighted-mean holding period (`NaN` when no bet closes). Mirrors
Python's `calculate_holding_period`.
"""
function calculate_holding_period(
    index::AbstractVector,
    target_positions::AbstractVector{<:Real},
)
    n = length(target_positions)
    holding = DataFrame(index = eltype(index)[], dT = Float64[], w = Float64[])
    time_entry = 0.0
    day_ms = 86_400_000.0
    time_diff = [Dates.value(index[i] - index[1]) / day_ms for i = 1:n]

    for i = 2:n
        current = target_positions[i]
        previous = target_positions[i-1]
        difference = current - previous
        if difference * previous >= 0          # increase / flat
            if current != 0
                time_entry = (time_entry * previous + time_diff[i] * difference) / current
            end
        else                                   # decrease / flip
            if current * previous < 0          # flip
                push!(holding, (index[i], time_diff[i] - time_entry, abs(previous)))
                time_entry = time_diff[i]
            else                               # partial decrease
                push!(holding, (index[i], time_diff[i] - time_entry, abs(difference)))
            end
        end
    end

    total_weight = sum(holding.w)
    mean_holding_period =
        total_weight > 0 ? sum(holding.dT .* holding.w) / total_weight : NaN
    return (holding_periods = holding, mean_holding_period = mean_holding_period)
end

"""
    calculate_hhi(bet_returns) -> Float64

Normalised Herfindahl–Hirschman Index of a return series (0 = diversified,
1 = concentrated). Returns `NaN` for two or fewer observations or a zero sum.
Mirrors Python's `calculate_hhi`.
"""
function calculate_hhi(bet_returns::AbstractVector{<:Real})
    n = length(bet_returns)
    n <= 2 && return NaN
    total = sum(bet_returns)
    total == 0 && return NaN
    weights = bet_returns ./ total
    hhi = sum(weights .^ 2)
    return (hhi - 1.0 / n) / (1.0 - 1.0 / n)
end

"""
    calculate_hhi_concentration(index, returns) -> (positive, negative, time)

HHI concentration of the positive returns, the negative returns, and the
monthly observation counts. Mirrors Python's `calculate_hhi_concentration`.
"""
function calculate_hhi_concentration(index::AbstractVector, returns::AbstractVector{<:Real})
    positive = calculate_hhi(returns[returns .>= 0])
    negative = calculate_hhi(returns[returns .< 0])
    months = [(year(t), month(t)) for t in index]
    counts = Float64[count(==(key), months) for key in unique(months)]
    time = calculate_hhi(counts)
    return (positive = positive, negative = negative, time = time)
end

"""
    compute_drawdowns_time_under_water(index, pnl; dollars=false) -> (start, drawdown, time_under_water)

Drawdowns and time under water between successive high-water marks. With
`dollars=true` drawdowns are `HWM - min`; otherwise they are the fractional
`1 - min/HWM`. Time under water is in fractional years (365.25-day). Mirrors
Python's `compute_drawdowns_time_under_water`.
"""
function compute_drawdowns_time_under_water(
    index::AbstractVector,
    pnl::AbstractVector{<:Real};
    dollars::Bool = false,
)
    n = length(pnl)
    high_water_mark = Vector{Float64}(undef, n)
    running = -Inf
    for i = 1:n
        running = max(running, pnl[i])
        high_water_mark[i] = running
    end

    starts = eltype(index)[]
    drawdown = Float64[]
    time_under_water = Float64[]
    day_ms = 86_400_000.0
    i = 1
    while i <= n
        j = i
        while j < n && high_water_mark[j+1] == high_water_mark[i]
            j += 1
        end
        group_min = minimum(@view pnl[i:j])
        if (j - i + 1) > 1 && high_water_mark[i] != group_min
            push!(starts, index[i])
            push!(
                drawdown,
                dollars ? high_water_mark[i] - group_min :
                1.0 - group_min / high_water_mark[i],
            )
            years = (Dates.value(index[j] - index[i]) / day_ms) / 365.25
            push!(time_under_water, years)
        end
        i = j + 1
    end
    return (start = starts, drawdown = drawdown, time_under_water = time_under_water)
end

# --------------------------------------------------------------------------- #
# Robust backtest statistics: Conditional Expected Drawdown (Goldberg–Mahmoud
# 2017) and the Ledoit–Wolf (2008) bootstrap Sharpe-difference test. CED is the
# tail mean (CVaR) of the maximum-drawdown distribution at a fixed horizon — a
# lower-variance, coherent, factor-attributable drawdown-risk measure than
# max-drawdown. The LW test studentizes the Sharpe difference by a Bartlett HAC
# standard error and calibrates it with a circular block bootstrap, holding
# nominal size under serial dependence where the naive z-test over-rejects.
# Admitted in Appraisal 22 (`library_extension/appraisals/22_verdict.md`).
# CED + the naive test (and the LW delta/se/stat) are parity-matched exactly; the
# LW bootstrap p-value uses Julia's RNG (behavioural).
# --------------------------------------------------------------------------- #

using SpecialFunctions: erfc

_wealth_from_returns(returns) = cumprod(1.0 .+ float.(returns))

function _drawdown_series(wealth)
    w = float.(wealth)
    return 1.0 .- w ./ accumulate(max, w)
end

# Empirical upper-tail CVaR: mean of the worst (1 - alpha) fraction of values.
function _cvar_upper(values, alpha)
    v = sort(float.(values))
    n = length(v)
    n == 0 && return 0.0
    k = (1.0 - alpha) * n
    k <= 0 && return v[end]
    n_full = floor(Int, k)
    total = n_full > 0 ? sum(@view v[(n-n_full+1):n]) : 0.0
    frac = k - n_full
    if frac > 0 && (n - n_full - 1) >= 0
        total += frac * v[n-n_full]
    end
    return total / k
end

function _rolling_max_drawdowns(wealth, horizon)
    w = float.(wealth)
    n = length(w)
    if n < horizon || horizon < 2
        d = _drawdown_series(w)
        return [isempty(d) ? 0.0 : maximum(d)]
    end
    out = Float64[]
    for s = 1:(n-horizon+1)
        win = @view w[s:(s+horizon-1)]
        hw = accumulate(max, win)
        push!(out, maximum(1.0 .- win ./ hw))
    end
    return out
end

"""
    conditional_expected_drawdown(returns, horizon, alpha=0.90) -> Float64

Conditional Expected Drawdown (CED; Goldberg–Mahmoud 2017): the CVaR (at level
`alpha`) of the maximum drawdowns within overlapping windows of length `horizon`.
A tail MEAN of the max-drawdown distribution, so lower estimator variance and
better drawdown-risk ranking than max-drawdown, converging to it on benign
returns; coherent and factor-attributable.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer CED over max-drawdown as a drawdown-risk statistic (lower estimator variance,
better ranking of true drawdown risk, most on short/heavy-tailed tracks; converges
on benign returns).

Mirrors Python's `conditional_expected_drawdown`. Reference: Goldberg & Mahmoud
(2017), Quantitative Finance 17(5).
"""
function conditional_expected_drawdown(
    returns::AbstractVector{<:Real},
    horizon::Integer,
    alpha::Real = 0.90,
)
    (0.0 < alpha < 1.0) || throw(ArgumentError("alpha must be in (0, 1)"))
    wealth = _wealth_from_returns(returns)
    return _cvar_upper(_rolling_max_drawdowns(wealth, Int(horizon)), alpha)
end

# Influence-function series of the Sharpe difference SR_A - SR_B and the estimate.
function _sharpe_influence(r_a, r_b)
    a = float.(r_a)
    b = float.(r_b)
    mu_a, mu_b = mean(a), mean(b)
    g_a, g_b = mean(a .* a), mean(b .* b)
    s_a = sqrt(max(g_a - mu_a * mu_a, 1e-300))
    s_b = sqrt(max(g_b - mu_b * mu_b, 1e-300))
    delta = mu_a / s_a - mu_b / s_b
    ga0, ga1 = g_a / s_a^3, -mu_a / (2.0 * s_a^3)
    gb0, gb1 = g_b / s_b^3, -mu_b / (2.0 * s_b^3)
    influence =
        ga0 .* (a .- mu_a) .+ ga1 .* (a .* a .- g_a) .-
        (gb0 .* (b .- mu_b) .+ gb1 .* (b .* b .- g_b))
    return delta, influence
end

# Bartlett-kernel HAC variance of the mean of the influence series.
function _bartlett_hac_var(influence, bandwidth)
    x = float.(influence)
    x = x .- mean(x)
    t = length(x)
    t < 2 && return NaN
    s = sum(x .* x) / t
    h = max(0, Int(bandwidth))
    for j = 1:h
        j >= t && break
        s += 2.0 * (1.0 - j / (h + 1.0)) * (sum(x[(j+1):end] .* x[1:(end-j)]) / t)
    end
    return max(s, 1e-300) / t
end

function _iid_var(influence)
    x = float.(influence)
    t = length(x)
    t < 2 && return NaN
    m = mean(x)
    return max(sum((x .- m) .^ 2) / t, 1e-300) / t
end

function _circular_block_indices(t, block_len, rng)
    n_blocks = cld(t, block_len)
    starts = rand(rng, 0:(t-1), n_blocks)
    idx = Int[]
    for s in starts, o = 0:(block_len-1)
        push!(idx, (s + o) % t)
    end
    return idx[1:t] .+ 1
end

"""
    sharpe_difference_test(returns_a, returns_b; method="ledoit_wolf",
        block_length=nothing, n_boot=1000, bandwidth=nothing, random_state=0)

Test H0: SR_A = SR_B. `method="naive"` is the influence-function delta-method
z-test using the i.i.d. variance (ignores serial dependence, over-rejects under
it). `method="ledoit_wolf"` (Ledoit–Wolf 2008) studentizes the Sharpe difference by
a Bartlett HAC standard error and calibrates the two-sided p-value with a
studentized circular block bootstrap. Returns a `NamedTuple`
`(delta, se, stat, pvalue, reject)`.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer the Ledoit-Wolf bootstrap Sharpe-difference test when comparing two Sharpes
under serial dependence/heavy tails (holds nominal size where the naive test
inflates ~3×; converges on i.i.d.).

Deliberate divergence: the LW bootstrap uses Julia's `rng`, so its p-value is
reproducible under `random_state` but not bit-identical to Python (delta, se and
stat are deterministic and parity-matched). Mirrors Python's
`sharpe_difference_test`. Reference: Ledoit & Wolf (2008), Journal of Empirical
Finance 15(5).
"""
function sharpe_difference_test(
    returns_a::AbstractVector{<:Real},
    returns_b::AbstractVector{<:Real};
    method::AbstractString = "ledoit_wolf",
    block_length::Union{Integer,Nothing} = nothing,
    n_boot::Integer = 1000,
    bandwidth::Union{Integer,Nothing} = nothing,
    random_state::Integer = 0,
)
    (method in ("ledoit_wolf", "naive")) ||
        throw(ArgumentError("method must be 'ledoit_wolf' or 'naive'"))
    a = float.(returns_a)
    b = float.(returns_b)
    length(a) == length(b) ||
        throw(ArgumentError("returns_a and returns_b must have the same length"))
    delta, influence = _sharpe_influence(a, b)

    if method == "naive"
        se = sqrt(_iid_var(influence))
        z = se > 0 ? delta / se : 0.0
        pvalue = erfc(abs(z) / sqrt(2.0))
        return (delta = delta, se = se, stat = z, pvalue = pvalue, reject = pvalue < 0.05)
    end

    t = length(a)
    bl = block_length === nothing ? max(1, ceil(Int, t^(1.0 / 3.0))) : block_length
    bw = bandwidth === nothing ? bl : bandwidth
    se_hat = sqrt(_bartlett_hac_var(influence, bw))
    z_hat = se_hat > 0 ? abs(delta / se_hat) : 0.0
    rng = MersenneTwister(random_state)
    count_ge = 0
    for _ = 1:n_boot
        idx = _circular_block_indices(t, bl, rng)
        delta_b, influence_b = _sharpe_influence(a[idx], b[idx])
        se_b = sqrt(_bartlett_hac_var(influence_b, bw))
        se_b <= 0 && continue
        abs((delta_b - delta) / se_b) >= z_hat && (count_ge += 1)
    end
    pvalue = (count_ge + 1.0) / (n_boot + 1.0)
    return (delta = delta, se = se_hat, stat = z_hat, pvalue = pvalue, reject = pvalue < 0.05)
end
