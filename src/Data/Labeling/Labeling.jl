"""
Financial labeling — native Julia port mirroring the Python
`RiskLabAI.data.labeling.labeling` API (López de Prado, AFML Ch. 3): symmetric
CUSUM event filters, daily volatility, vertical/triple barriers, meta-events and
meta-labeling.

Representation note (deliberate divergence): the Python API uses time-indexed
pandas Series/DataFrames. The Julia port passes the price series as parallel
sorted vectors `(close_index::Vector{DateTime}, close::Vector{<:Real})`, and the
event tables as `DataFrame`s with explicit `event_start`/`end_time` columns. The
numerics match the Python implementation exactly (verified in
`test/runtests.jl`), including pandas' debiased EWM standard deviation.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 3.
"""

using DataFrames
using Dates
using Statistics: mean

# pandas-compatible debiased exponentially-weighted std (adjust=true).
function _ewm_std(x::AbstractVector{<:Real}, span::Integer)
    alpha = 2 / (span + 1)
    n = length(x)
    out = fill(NaN, n)
    for t in 1:n
        sum_w = 0.0
        sum_w2 = 0.0
        weighted_mean = 0.0
        for i in 1:t
            w = (1 - alpha)^(t - i)
            weighted_mean += w * x[i]
            sum_w += w
            sum_w2 += w * w
        end
        m = weighted_mean / sum_w
        ssd = 0.0
        for i in 1:t
            w = (1 - alpha)^(t - i)
            ssd += w * (x[i] - m)^2
        end
        denom = sum_w * sum_w - sum_w2
        out[t] = denom > 0 ? sqrt(sum_w / denom * ssd) : NaN
    end
    return out
end

"""
    symmetric_cusum_filter(close_index, close, threshold) -> Vector{DateTime}

Symmetric CUSUM filter with a fixed threshold; returns the timestamps where the
cumulative up/down price move first exceeds `threshold`. Mirrors Python's
`symmetric_cusum_filter` (Snippet 3.2).
"""
function symmetric_cusum_filter(
    close_index::AbstractVector, close::AbstractVector{<:Real}, threshold::Real
)
    events = eltype(close_index)[]
    shift_pos = 0.0
    shift_neg = 0.0
    for i in 2:length(close)
        delta = close[i] - close[i - 1]
        shift_pos = max(0.0, shift_pos + delta)
        shift_neg = min(0.0, shift_neg + delta)
        if shift_neg < -threshold
            shift_neg = 0.0
            push!(events, close_index[i])
        elseif shift_pos > threshold
            shift_pos = 0.0
            push!(events, close_index[i])
        end
    end
    return events
end

"""
    cusum_filter_events_dynamic_threshold(close_index, close, threshold) -> Vector{DateTime}

Symmetric CUSUM filter with a per-timestamp `threshold` vector (aligned to
`close_index`; `threshold[i]` applies to the move into bar `i`). Mirrors
Python's `cusum_filter_events_dynamic_threshold`.
"""
function cusum_filter_events_dynamic_threshold(
    close_index::AbstractVector, close::AbstractVector{<:Real}, threshold::AbstractVector{<:Real}
)
    events = eltype(close_index)[]
    shift_pos = 0.0
    shift_neg = 0.0
    for i in 2:length(close)
        delta = close[i] - close[i - 1]
        thr = threshold[i]
        shift_pos = max(0.0, shift_pos + delta)
        shift_neg = min(0.0, shift_neg + delta)
        if shift_neg < -thr
            shift_neg = 0.0
            push!(events, close_index[i])
        elseif shift_pos > thr
            shift_pos = 0.0
            push!(events, close_index[i])
        end
    end
    return events
end

"""
    daily_volatility_with_log_returns(close_index, close; span=100) -> (index, volatility)

EWMA standard deviation of (roughly) daily log returns. Mirrors Python's
`daily_volatility_with_log_returns` (Snippet 3.1), including the one-day-prior
index lookup and pandas' debiased EWM std (so the first value is `NaN`).
"""
function daily_volatility_with_log_returns(
    close_index::AbstractVector, close::AbstractVector{<:Real}; span::Integer=100
)
    n = length(close_index)
    # 0-based count of timestamps strictly before (t - 1 day), per Python searchsorted.
    prev_count = [searchsortedfirst(close_index, close_index[i] - Day(1)) - 1 for i in 1:n]
    keep = findall(>(0), prev_count)
    isempty(keep) && return (index = eltype(close_index)[], volatility = Float64[])

    span_len = length(keep)
    current_pos = (n - span_len + 1):n
    prior_pos = prev_count[keep]   # 1-based positions of the prior price
    returns = [log(close[current_pos[j]] / close[prior_pos[j]]) for j in 1:span_len]
    return (index = close_index[current_pos], volatility = _ewm_std(returns, span))
end

"""
    vertical_barrier(close_index, time_events, number_days) -> (event, barrier)

For each event, the first timestamp at least `number_days` later. Events whose
barrier falls past the end of the series are dropped. Mirrors Python's
`vertical_barrier` (Snippet 3.4).
"""
function vertical_barrier(close_index::AbstractVector, time_events::AbstractVector, number_days::Integer)
    n = length(close_index)
    events = eltype(close_index)[]
    barriers = eltype(close_index)[]
    for te in time_events
        pos = searchsortedfirst(close_index, te + Day(number_days))
        if pos <= n
            push!(events, te)
            push!(barriers, close_index[pos])
        end
    end
    return (event = events, barrier = barriers)
end

"""
    triple_barrier(close_index, close, events, ptsl) -> Vector{Union{Missing,DateTime}}

First-touch timestamp for each event among {vertical barrier, profit-taking,
stop-loss}. `events` is a `DataFrame` with `event_start`, `end_time` (vertical
barrier, may be `missing`), `base_width`, `side`. `ptsl = (pt_mult, sl_mult)`
(a non-positive multiplier disables that horizontal barrier). Mirrors Python's
`triple_barrier` (Snippet 3.3).
"""
function triple_barrier(
    close_index::AbstractVector, close::AbstractVector{<:Real}, events::DataFrame, ptsl
)
    position = Dict(t => i for (i, t) in enumerate(close_index))
    result = Vector{Union{Missing,eltype(close_index)}}(undef, nrow(events))
    for r in 1:nrow(events)
        start_pos = position[events.event_start[r]]
        vbar = events.end_time[r]
        vfill = ismissing(vbar) ? close_index[end] : vbar
        end_pos = searchsortedlast(close_index, vfill)
        segment = close[start_pos:end_pos]
        side = events.side[r]
        width = events.base_width[r]
        profit_taking = ptsl[1] > 0 ? ptsl[1] * width : Inf
        stop_loss = ptsl[2] > 0 ? -ptsl[2] * width : -Inf

        path = log.(segment ./ segment[1]) .* side
        candidates = Union{Missing,eltype(close_index)}[vbar]
        below = findfirst(<(stop_loss), path)
        below !== nothing && push!(candidates, close_index[start_pos + below - 1])
        above = findfirst(>(profit_taking), path)
        above !== nothing && push!(candidates, close_index[start_pos + above - 1])

        present = collect(skipmissing(candidates))
        result[r] = isempty(present) ? missing : minimum(present)
    end
    return result
end

"""
    meta_events(close_index, close, time_events, ptsl, target, return_min;
                vertical_barriers=nothing, side=nothing) -> DataFrame

Build the triple-barrier event table. `target` maps event timestamp → barrier
width (volatility); events with `target ≤ return_min` (or missing) are dropped.
`vertical_barriers` / `side` are optional `event → value` mappings. Mirrors
Python's `meta_events` (Snippet 3.6); the parallelism is an implementation
detail (computed serially here).
"""
function meta_events(
    close_index::AbstractVector,
    close::AbstractVector{<:Real},
    time_events::AbstractVector,
    ptsl,
    target::AbstractDict,
    return_min::Real;
    vertical_barriers::Union{Nothing,AbstractDict}=nothing,
    side::Union{Nothing,AbstractDict}=nothing,
)
    starts = eltype(close_index)[]
    widths = Float64[]
    for t in time_events
        if haskey(target, t) && target[t] > return_min
            push!(starts, t)
            push!(widths, target[t])
        end
    end

    DT = eltype(close_index)
    end_time = Vector{Union{Missing,DT}}(
        vertical_barriers === nothing ? fill(missing, length(starts)) :
        [get(vertical_barriers, t, missing) for t in starts],
    )
    if side === nothing
        sides = fill(1.0, length(starts))
        ptsl_final = (ptsl[1], ptsl[1])
    else
        sides = [get(side, t, 1.0) for t in starts]
        ptsl_final = (ptsl[1], ptsl[2])
    end

    events = DataFrame(event_start = starts, end_time = end_time, base_width = widths, side = sides)
    events.end_time = triple_barrier(close_index, close, events, ptsl_final)
    side === nothing && select!(events, Not(:side))
    return events
end

"""
    meta_labeling(events, close_index, close) -> DataFrame

Realised return and label for each event (dropping events with no barrier
touch). With a `side` column the label is meta (1 if profitable else 0);
otherwise it is the sign of the return. Mirrors Python's `meta_labeling`
(Snippet 3.7).
"""
function meta_labeling(events::DataFrame, close_index::AbstractVector, close::AbstractVector{<:Real})
    price = Dict(t => p for (t, p) in zip(close_index, close))
    valid = .!ismissing.(events.end_time)
    sub = events[valid, :]
    starts = sub.event_start
    ends = sub.end_time
    returns = [log(price[ends[i]]) - log(price[starts[i]]) for i in eachindex(starts)]

    has_side = "side" in names(sub)
    if has_side
        returns = returns .* sub.side
    end
    labels = float.(sign.(returns))
    if has_side
        labels[returns .<= 0] .= 0.0
    end

    out = DataFrame(event_start = starts, end_time = ends, ret = returns, label = labels)
    has_side && (out.side = sub.side)
    return out
end
