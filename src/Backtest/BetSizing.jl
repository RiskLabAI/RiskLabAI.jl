"""
Bet sizing — native Julia port mirroring the Python
`RiskLabAI.backtest.bet_sizing` API (López de Prado, AFML Ch. 10): probability /
meta-label bet sizes, concurrent-signal averaging, signal discretisation, and the
sigmoid position-sizing family.

The normal CDF comes from `Distributions` (matching SciPy's `norm.cdf`); the
deterministic metrics match the Python implementation exactly (verified in
`test/runtests.jl`). The Julia API adopts the 2.0.0 snake_case canon directly and
does **not** carry the deprecated camelCase aliases.

Representation note (deliberate divergence): the pandas-Series/DataFrame inputs
become parallel sorted vectors — bets are passed as `(start_times, end_times,
signal/probability/side)` vectors, and the concurrent-signal helpers return
`(time_points, values)` rather than time-indexed Series. The averaging is
computed serially (Python's `mp_pandas_obj` parallelism is an implementation
detail) using the same prefix-sum algorithm.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 10.
"""

"""
    probability_bet_size(probabilities, sides) -> Vector

Bet size `side · (2·Φ(p) - 1)` from meta-model probabilities. Mirrors Python's
`probability_bet_size`.
"""
probability_bet_size(probabilities::AbstractVector{<:Real}, sides::AbstractVector{<:Real}) =
    sides .* (2 .* cdf.(Normal(), probabilities) .- 1)

"""
    average_bet_sizes(price_dates, start_dates, end_dates, bet_sizes) -> Vector{Float64}

Concurrent average bet size at each `price_dates[i]`, averaging `bet_sizes[j]`
over bets active at that time (`start_dates[j] ≤ price_dates[i] ≤ end_dates[j]`).
Mirrors Python's `average_bet_sizes`.
"""
function average_bet_sizes(
    price_dates,
    start_dates,
    end_dates,
    bet_sizes::AbstractVector{<:Real},
)
    out = zeros(Float64, length(price_dates))
    for i in eachindex(price_dates)
        total = 0.0
        count = 0
        for j in eachindex(start_dates)
            if start_dates[j] <= price_dates[i] <= end_dates[j]
                total += bet_sizes[j]
                count += 1
            end
        end
        count > 0 && (out[i] = total / count)
    end
    return out
end

"""
    strategy_bet_sizing(price_timestamps, start_times, end_times, sides, probabilities) -> Vector{Float64}

Average bet size of a strategy over `price_timestamps`: convert probabilities and
sides to per-bet sizes, then take the concurrent average. The bet inputs are
parallel, already-aligned vectors. Mirrors Python's `strategy_bet_sizing`.
"""
function strategy_bet_sizing(
    price_timestamps,
    start_times,
    end_times,
    sides::AbstractVector{<:Real},
    probabilities::AbstractVector{<:Real},
)
    bet_sizes = probability_bet_size(probabilities, sides)
    return average_bet_sizes(price_timestamps, start_times, end_times, bet_sizes)
end

"""
    mp_avg_active_signals(start_times, end_times, signal_values, molecule) -> Vector{Float64}

Average of the signals active at each timestamp in `molecule`: a signal is active
when `start ≤ t` and `t < end` (an `end` of `missing` never closes). Uses
prefix sums + binary search (`O((n+m) log n)`). Mirrors Python's
`mp_avg_active_signals`.
"""
function mp_avg_active_signals(
    start_times,
    end_times,
    signal_values::AbstractVector{<:Real},
    molecule,
)
    n = length(start_times)
    m = length(molecule)
    (n == 0 || m == 0) && return zeros(Float64, m)
    values = Float64.(signal_values)

    start_order = sortperm(start_times; alg = MergeSort)
    sorted_starts = start_times[start_order]
    cum_signal_start = vcat(0.0, cumsum(values[start_order]))
    cum_count_start = collect(0:n)

    finite = .!ismissing.(end_times)
    end_finite = collect(skipmissing(end_times))
    values_end = values[finite]
    end_order = sortperm(end_finite; alg = MergeSort)
    sorted_ends = end_finite[end_order]
    cum_signal_end = vcat(0.0, cumsum(values_end[end_order]))
    cum_count_end = collect(0:length(sorted_ends))

    averages = Vector{Float64}(undef, m)
    for k = 1:m
        started = searchsortedlast(sorted_starts, molecule[k])
        ended = searchsortedlast(sorted_ends, molecule[k])
        active_signal = cum_signal_start[started+1] - cum_signal_end[ended+1]
        active_count = cum_count_start[started+1] - cum_count_end[ended+1]
        averages[k] = active_count > 0 ? active_signal / active_count : 0.0
    end
    return averages
end

"""
    avg_active_signals(start_times, end_times, signal_values) -> (time_points, averages)

Average active signal at every signal change-point (the union of start and end
times). Mirrors Python's `avg_active_signals`.
"""
function avg_active_signals(start_times, end_times, signal_values::AbstractVector{<:Real})
    time_points = sort(unique(vcat(collect(skipmissing(end_times)), collect(start_times))))
    averages = mp_avg_active_signals(start_times, end_times, signal_values, time_points)
    return (time_points, averages)
end

"""
    discrete_signal(signal, step_size) -> Vector

Discretise a signal to multiples of `step_size`, capped to `[-1, 1]`. Mirrors
Python's `discrete_signal`.
"""
function discrete_signal(signal::AbstractVector{<:Real}, step_size::Real)
    return clamp.(round.(signal ./ step_size) .* step_size, -1.0, 1.0)
end

"""
    generate_signal(start_times, end_times, sides, probability, prediction, n_classes, step_size)
        -> (time_points, discretized_signal)

End-to-end signal: one-vs-rest t-values from `probability`, mapped to
`prediction · (2·Φ(t) - 1)`, optionally meta-labelled by `sides` (pass `nothing`
to skip), then concurrently averaged and discretised. Mirrors Python's
`generate_signal`.
"""
function generate_signal(
    start_times,
    end_times,
    sides::Union{Nothing,AbstractVector{<:Real}},
    probability::AbstractVector{<:Real},
    prediction::AbstractVector{<:Real},
    n_classes::Integer,
    step_size::Real,
)
    isempty(probability) && return (eltype(start_times)[], Float64[])
    p = Float64.(probability)
    t_value = (p .- 1.0 / n_classes) ./ sqrt.(p .* (1 .- p))
    signal = Float64.(prediction) .* (2 .* cdf.(Normal(), t_value) .- 1)
    sides !== nothing && (signal = signal .* Float64.(sides))
    time_points, averages = avg_active_signals(start_times, end_times, signal)
    return (time_points, discrete_signal(averages, step_size))
end

"""
    bet_size_sigmoid(w, x) -> Float64

Sigmoid bet size `x / √(w + x²)` (in `[-1, 1]`). Mirrors Python's
`bet_size_sigmoid`.
"""
bet_size_sigmoid(w::Real, x::Real) = x / sqrt(w + x^2)

"""
    target_position(w, f, actual_price, maximum_position_size) -> Int

Target position `⌊sigmoid·max⌋` (truncated toward zero) from the sigmoid bet
size of the divergence `f - actual_price`. Mirrors Python's `target_position`.
"""
target_position(w::Real, f::Real, actual_price::Real, maximum_position_size::Integer) =
    trunc(Int, bet_size_sigmoid(w, f - actual_price) * maximum_position_size)

"""
    inverse_price(f, w, m) -> Float64

Price implied by a sigmoid bet size `m`: `f - m·√(w/(1-m²))` (returns `f` at
`m = ±1`). Mirrors Python's `inverse_price`.
"""
function inverse_price(f::Real, w::Real, m::Real)
    (m == 1.0 || m == -1.0) && return float(f)
    return f - m * sqrt(w / (1 - m^2))
end

"""
    limit_price(target_position_size, current_position, f, w, maximum_position_size) -> Float64

Average limit price to move from `current_position` to `target_position_size`.
Mirrors Python's `limit_price`.
"""
function limit_price(
    target_position_size::Integer,
    current_position::Integer,
    f::Real,
    w::Real,
    maximum_position_size::Integer,
)
    target_position_size == current_position && return float(f)
    sgn = sign(target_position_size - current_position)
    limit = 0.0
    for i = abs(current_position+sgn):(abs(target_position_size+sgn)-1)
        limit += inverse_price(f, w, i / maximum_position_size)
    end
    limit /= abs(target_position_size - current_position)
    return limit
end

"""
    compute_sigmoid_width(x, m) -> Float64

Sigmoid width `w = x²·(1/m² - 1)` implied by a divergence `x` and bet size `m`
(`Inf` when `m ∈ {0, 1, -1}`). Mirrors Python's `compute_sigmoid_width`.
"""
function compute_sigmoid_width(x::Real, m::Real)
    (m == 0.0 || m == 1.0 || m == -1.0) && return Inf
    return x^2 * (1 / m^2 - 1)
end
