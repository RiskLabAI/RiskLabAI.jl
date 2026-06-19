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
