using Random

"""
Sample-weighting techniques — native Julia port mirroring the Python
`RiskLabAI.data.weights.sample_weights` API (López de Prado, AFML Ch. 4):
label concurrency, average uniqueness, sequential bootstrap, absolute-return
sample weights, and time decay.

Representation note (deliberate divergence): the Python API uses pandas Series
whose *index* carries event start times. The Julia port passes event starts and
ends as parallel vectors (`event_start`, `event_end`) and the bar/price index as
a sorted vector. Shared Python 3.2.0 fixtures check concurrency, return weights
and sequential-bootstrap choices in `test/sample_weight_regressions.jl`.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 4.
"""

"""
    expand_label_for_meta_labeling(close_index, event_start, event_end, molecule)
        -> (index, concurrency)

Count concurrent events at each timestamp over the affected span. Mirrors
Python's `expand_label_for_meta_labeling`. `event_start[i] → event_end[i]` define
event `i`; `molecule` is the subset of start times to process. All events that
overlap the selected span contribute, including events outside `molecule`.
Endpoints are inclusive; missing or NaN ends use the last close timestamp.
"""
function expand_label_for_meta_labeling(
    close_index::AbstractVector,
    event_start::AbstractVector,
    event_end::AbstractVector,
    molecule::AbstractVector,
)
    last_label = close_index[end]
    m_first = molecule[1]
    length(event_start) == length(event_end) ||
        throw(DimensionMismatch("event starts and ends must have equal lengths"))
    filled_end = [
        ismissing(value) || (value isa AbstractFloat && isnan(value)) ? last_label : value for
        value in event_end
    ]
    end_of = Dict(zip(event_start, filled_end))
    max_end = maximum(end_of[start] for start in molecule)
    keep = [
        (filled_end[i] >= m_first) && (event_start[i] <= max_end) for
        i in eachindex(event_start)
    ]
    starts = event_start[keep]
    ends = filled_end[keep]
    isempty(starts) && return (index = empty(close_index), concurrency = Float64[])

    span = [t for t in close_index if t >= m_first && t <= max_end]
    concurrency = zeros(Float64, length(span))
    for (s, e) in zip(starts, ends)
        concurrency[(span .>= s) .& (span .<= e)] .+= 1.0
    end

    return (index = span, concurrency = concurrency)
end

"""
    calculate_average_uniqueness(index_matrix) -> Vector{Float64}

Average uniqueness of each event (column) given a T×N indicator matrix:
`mean_t( I[t,j] / c_t )` over the event's active periods, where
`c_t = Σ_j I[t,j]`. Mirrors Python's `calculate_average_uniqueness`.
"""
function calculate_average_uniqueness(index_matrix::AbstractMatrix{<:Real})
    n_periods, n_events = size(index_matrix)
    concurrency = vec(sum(value -> isnan(value) ? zero(value) : value, index_matrix; dims = 2))
    result = zeros(Float64, n_events)
    for j = 1:n_events
        total = 0.0
        duration = 0
        for t = 1:n_periods
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
    sequential_bootstrap(indicator_matrix; sample_length=nothing, random_state=nothing)

Draw one-based event-column positions with replacement using average added
uniqueness. Rows are observation intervals and columns are events. The matrix
must contain finite real zeros and ones, and every event must be active in at
least one row. The default draw count is the number of event columns.

For candidate event `j`, average `1 / (c_t + 1)` over its active rows, where
`c_t` counts all previous selections, including repeats. Normalize these scores
to obtain the next draw probabilities. The first draw is uniform.

`random_state` accepts a nonnegative integer seed, an `AbstractRNG`, or `nothing`
for a fresh `Xoshiro` generator. A supplied generator advances in place. Zero
draws consume no random numbers. Julia and NumPy seeds do not imply identical
streams; parity fixtures supply identical uniform draws explicitly.
"""
function sequential_bootstrap(
    indicator_matrix::AbstractMatrix;
    sample_length = nothing,
    random_state = nothing,
)
    eltype(indicator_matrix) <: Real ||
        throw(ArgumentError("indicator_matrix must contain real binary values"))
    all(value -> isfinite(value) && (value == 0 || value == 1), indicator_matrix) ||
        throw(ArgumentError("indicator_matrix must contain only finite zeros and ones"))
    event_count = size(indicator_matrix, 2)
    durations = vec(sum(indicator_matrix; dims = 1))
    any(iszero, durations) &&
        throw(ArgumentError("each event must contain at least one active interval"))
    draw_count = isnothing(sample_length) ? event_count : sample_length
    draw_count isa Integer && !(draw_count isa Bool) && draw_count >= 0 ||
        throw(ArgumentError("sample_length must be a nonnegative integer"))
    event_count == 0 && draw_count > 0 &&
        throw(ArgumentError("cannot draw from an indicator matrix with no events"))
    generator = if isnothing(random_state)
        Xoshiro()
    elseif random_state isa AbstractRNG
        random_state
    elseif random_state isa Integer && !(random_state isa Bool) && random_state >= 0
        Xoshiro(random_state)
    else
        throw(ArgumentError("random_state must be a nonnegative seed or AbstractRNG"))
    end
    selected = Vector{Int}(undef, draw_count)
    concurrency = zeros(Int, size(indicator_matrix, 1))
    scores = zeros(Float64, event_count)
    for draw in eachindex(selected)
        for event in 1:event_count
            total = 0.0
            for time in axes(indicator_matrix, 1)
                if indicator_matrix[time, event] == 1
                    total += 1.0 / (concurrency[time] + 1)
                end
            end
            scores[event] = total / durations[event]
        end
        threshold = rand(generator) * sum(scores)
        cumulative = 0.0
        chosen = event_count
        for event in 1:event_count
            cumulative += scores[event]
            if threshold < cumulative
                chosen = event
                break
            end
        end
        selected[draw] = chosen
        for time in axes(indicator_matrix, 1)
            concurrency[time] += indicator_matrix[time, chosen] == 1
        end
    end
    return selected
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
    for i = 2:length(price)
        log_return[i] = abs(log(price[i]) - log(price[i-1]))
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
function calculate_time_decay(weight::AbstractVector{<:Real}; clf_last_weight::Real = 1.0)
    (clf_last_weight < 0 || clf_last_weight > 1) &&
        throw(ArgumentError("clf_last_weight must be between 0 and 1"))
    cumulative = Vector{Float64}(undef, length(weight))
    running = 0.0
    for (index, value) in enumerate(weight)
        if isnan(value)
            cumulative[index] = NaN
        else
            running += value
            cumulative[index] = running
        end
    end
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
