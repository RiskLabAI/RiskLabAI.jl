"""
Sample-weighting techniques — native Julia port mirroring the Python
`RiskLabAI.data.weights.sample_weights` API (López de Prado, AFML Ch. 4):
label concurrency, average uniqueness, absolute-return sample weights, and
time decay.

Representation note (deliberate divergence): the Python API uses pandas Series
whose *index* carries event start times. The Julia port passes event starts and
ends as parallel vectors (`event_start`, `event_end`) and the bar/price index as
a sorted vector. The numerics match the Python implementation exactly (verified
against reference values in `test/runtests.jl`).

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 4.
"""

"""
    expand_label_for_meta_labeling(close_index, event_start, event_end, molecule)
        -> (index, concurrency)

Count concurrent events at each timestamp over the affected span. Mirrors
Python's `expand_label_for_meta_labeling`. `event_start[i] → event_end[i]` define
event `i`; `molecule` is the subset of start times to process.
"""
function expand_label_for_meta_labeling(
    close_index::AbstractVector,
    event_start::AbstractVector,
    event_end::AbstractVector,
    molecule::AbstractVector,
)
    last_label = close_index[end]
    m_first = molecule[1]
    keep = [
        (event_start[i] in molecule) && (event_end[i] > m_first)
        for i in eachindex(event_start)
    ]
    starts = event_start[keep]
    ends = event_end[keep]
    isempty(starts) && return (index = empty(close_index), concurrency = Float64[])

    first_start = starts[1]
    max_end = maximum(ends)
    span = [t for t in close_index if t >= first_start && t <= max_end]
    concurrency = zeros(Float64, length(span))
    for (s, e) in zip(starts, ends)
        capped = min(e, span[end])
        concurrency[(span .>= s) .& (span .<= capped)] .+= 1.0
    end

    keepmask = (span .>= m_first) .& (span .<= max_end)
    return (index = span[keepmask], concurrency = concurrency[keepmask])
end

"""
    calculate_average_uniqueness(index_matrix) -> Vector{Float64}

Average uniqueness of each event (column) given a T×N indicator matrix:
`mean_t( I[t,j] / c_t )` over the event's active periods, where
`c_t = Σ_j I[t,j]`. Mirrors Python's `calculate_average_uniqueness`.
"""
function calculate_average_uniqueness(index_matrix::AbstractMatrix{<:Real})
    n_periods, n_events = size(index_matrix)
    concurrency = vec(sum(index_matrix; dims = 2))
    result = zeros(Float64, n_events)
    for j in 1:n_events
        total = 0.0
        duration = 0
        for t in 1:n_periods
            if index_matrix[t, j] > 0
                duration += 1
                if concurrency[t] > 0
                    total += index_matrix[t, j] / concurrency[t]
                end
            end
        end
        result[j] = duration > 0 ? total / duration : 0.0
    end
    return result
end

"""
    sample_weight_absolute_return_meta_labeling(event_start, event_end,
        price_index, price, molecule) -> Vector{Float64}

Sample weights from absolute log-return attribution: `w_i = Σ |r_t| / c_t` over
the event's span, then normalised to sum to the number of events. Mirrors
Python's `sample_weight_absolute_return_meta_labeling` (NaN returns are skipped,
as pandas `.sum()` does).
"""
function sample_weight_absolute_return_meta_labeling(
    event_start::AbstractVector,
    event_end::AbstractVector,
    price_index::AbstractVector,
    price::AbstractVector,
    molecule::AbstractVector,
)
    expanded = expand_label_for_meta_labeling(price_index, event_start, event_end, molecule)
    concurrency = zeros(Float64, length(price_index))
    pos = Dict(t => i for (i, t) in enumerate(price_index))
    for (t, c) in zip(expanded.index, expanded.concurrency)
        concurrency[pos[t]] = c
    end

    # |log-return|; first entry NaN (no prior price), matching pandas diff().
    log_return = fill(NaN, length(price))
    for i in 2:length(price)
        log_return[i] = abs(log(price[i]) - log(price[i - 1]))
    end

    # Map molecule start -> event end.
    end_of = Dict(event_start[i] => event_end[i] for i in eachindex(event_start))
    weight = zeros(Float64, length(molecule))
    for (k, t_in) in enumerate(molecule)
        t_out = end_of[t_in]
        if !haskey(pos, t_out)
            j = searchsortedlast(price_index, t_out)
            t_out = price_index[max(j, 1)]
        end
        mask = (price_index .>= t_in) .& (price_index .<= t_out)
        acc = 0.0
        for i in findall(mask)
            if concurrency[i] > 0 && !isnan(log_return[i])
                acc += log_return[i] / concurrency[i]
            end
        end
        weight[k] = abs(acc)
    end

    total = sum(weight)
    total == 0 && return ones(Float64, length(molecule))
    return weight .* (length(weight) / total)
end

"""
    calculate_time_decay(weight; clf_last_weight=1.0) -> Vector{Float64}

Apply a linear time decay to `weight` (assumed in chronological order): the most
recent observation keeps weight 1, the oldest gets `clf_last_weight`
(∈ [0, 1]). Mirrors Python's `calculate_time_decay`.
"""
function calculate_time_decay(weight::AbstractVector{<:Real}; clf_last_weight::Real=1.0)
    (clf_last_weight < 0 || clf_last_weight > 1) &&
        throw(ArgumentError("clf_last_weight must be between 0 and 1"))
    cumulative = cumsum(weight)
    if clf_last_weight == 1.0 || isempty(cumulative) || cumulative[end] == 0
        slope = 0.0
        const_term = 1.0
    else
        slope = (1.0 - clf_last_weight) / cumulative[end]
        const_term = 1.0 - slope * cumulative[end]
    end
    decayed = const_term .+ slope .* cumulative
    decayed[decayed .< 0] .= 0.0
    return decayed
end
