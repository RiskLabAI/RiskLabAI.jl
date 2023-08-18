using DataFrames
using StatsBase

"""
    concurrencyEvents(closeIndex::DataFrame, timestamp::DataFrame, molecule::Vector) -> DataFrame

Calculate the concurrency events.

# Arguments

- `closeIndex::DataFrame`: DataFrame that has events.
- `timestamp::DataFrame`: DataFrame that has return and label of each period.
- `molecule::Vector`: Index that function must apply on.

# Returns

- `DataFrame`: Result of the specified function.
"""
function concurrencyEvents(
    closeIndex::DataFrame,
    timestamp::DataFrame,
    molecule::Vector
)::DataFrame

    eventsFiltered = filter(row -> row[:date] in molecule, timestamp)
    startTime = eventsFiltered.date[1]
    endTime = maximum(eventsFiltered.timestamp)
    concurrencyIndex = closeIndex[(closeIndex .>= startTime) .& (closeIndex .<= endTime)]
    concurrency = DataFrame(date=concurrencyIndex, concurrency=zeros(size(concurrencyIndex, 1)))
    for (i, idx) in enumerate(eventsFiltered.date)
        startIndex = concurrency.date .>= idx
        endIndex = concurrency.date .<= eventsFiltered.timestamp[i]
        selectedIndex = startIndex .& endIndex
        concurrency.concurrency[selectedIndex] .+= 1
    end
    return concurrency
end

"""
    sampleWeight(timestamp::DataFrame, concurrencyEvents::DataFrame, molecule::Vector) -> DataFrame

Compute the sample weight with triple barrier for meta-labeling.

# Arguments

- `timestamp::DataFrame`: DataFrame of events start and end for labeling.
- `concurrencyEvents::DataFrame`: DataFrame of concurrent events for each event.
- `molecule::Vector`: Index that function must apply on.

# Returns

- `DataFrame`: Result of the specified function.
"""
function sampleWeight(
    timestamp::DataFrame,
    concurrencyEvents::DataFrame,
    molecule::Vector
)::DataFrame

    eventsFiltered = filter(row -> row[:date] in molecule, timestamp)
    weight = DataFrame(date=molecule, weight=zeros(length(molecule)))
    for i in 1:size(weight, 1)
        startTime, endTime = eventsFiltered.date[i], eventsFiltered.timestamp[i]
        concurrencyEventsForSpecificTime = concurrencyEvents[(concurrencyEvents.date .>= startTime) .& (concurrencyEvents.date .<= endTime), "concurrency"]
        weight[i, "weight"] = mean(1.0 ./ concurrencyEventsForSpecificTime)
    end
    return weight
end

"""
    indexMatrix(barIndex::Vector, timestamp::DataFrame) -> Matrix{Float64}

Create an index matrix that shows whether an index is in a time horizon or not for meta-labeling.

# Arguments

- `barIndex::Vector`: Index of all data.
- `timestamp::DataFrame`: Times of events containing starting and ending time.

# Returns

- `Matrix{Float64}`: Index matrix.
"""
function indexMatrix(
    barIndex::Vector,
    timestamp::DataFrame
)::Matrix{Float64}

    indexMatrix = zeros((size(barIndex, 1), size(timestamp, 1)))
    for (j, (t0, t1)) in enumerate(timestamp)
        indicator = [(i <= t1 && i >= t0) ? 1 : 0 for i in barIndex]
        indexMatrix[:, j] = indicator
    end
    return indexMatrix
end

"""
    averageUniqueness(indexMatrix::Matrix{Float64}) -> Vector{Float64}

Compute the average uniqueness for meta-labeling.

# Arguments

- `indexMatrix::Matrix{Float64}`: Matrix that indicates events.

# Returns

- `Vector{Float64}`: Average uniqueness for each event.
"""
function averageUniqueness(indexMatrix::Matrix{Float64})::Vector{Float64}

    concurrency = sum(indexMatrix, dims=2)
    uniqueness = copy(indexMatrix)
    for i in 1:size(indexMatrix, 1)
        if concurrency[i] > 0
            uniqueness[i, :] = uniqueness[i, :] / concurrency[i]
        end
    end
    averageUniqueness_ = [sum(uniqueness[:, i]) / sum(indexMatrix[:, i]) for i in 1:size(indexMatrix, 2)]
    return averageUniqueness_
end


"""
    calculateConcurrency(events::DataFrame, molecule::Vector{Date})::DataFrame

Calculates concurrency events.

Computes concurrency events for the given time range specified by the molecule index.

# Arguments
- `events::DataFrame`: DataFrame that has return and label of each period.
- `molecule::Vector{Date}`: Index that function must apply on.

# Returns
- `DataFrame`: Result of the specified function.

# Related Mathematical Formulae
- Concurrency of an event is the number of overlapping events at a specific time.
"""
function calculateConcurrency(
        events::DataFrame,
        molecule::Vector{Date}
    )::DataFrame
    eventsFiltered = filter(row -> row[:date] in molecule, events)
    startTime = eventsFiltered.date[1]
    endTime = maximum(eventsFiltered.timestamp)
    concurrencyIndex = filter(row -> row[:date] >= startTime && row[:date] <= endTime, events)
    concurrency = DataFrame(date=concurrencyIndex.date, concurrency=zeros(size(concurrencyIndex)[1]))
    for (i, idx) in enumerate(eventsFiltered.date)
        startIndex = concurrency.date .>= idx
        endIndex = concurrency.date .<= eventsFiltered.timestamp[i]
        selectedIndex = startIndex .& endIndex
        concurrency.concurrency[selectedIndex] .+= 1
    end
    return concurrency
end

"""
    calculateSampleWeight(
        timestamp::DataFrame,
        concurrencyEvents::DataFrame,
        molecule::Vector{Date}
    )::DataFrame

Computes sample weight with triple barrier.

Computes the sample weight with triple barrier for meta-labeling.

# Arguments
- `timestamp::DataFrame`: DataFrame of events start and end for labeling.
- `concurrencyEvents::DataFrame`: DataFrame of concurrent events for each event.
- `molecule::Vector{Date}`: Index that function must apply on.

# Returns
- `DataFrame`: Result of the specified function.
"""
function calculateSampleWeight(
        timestamp::DataFrame,
        concurrencyEvents::DataFrame,
        molecule::Vector{Date}
    )::DataFrame
    eventsFiltered = filter(row -> row[:date] in molecule, timestamp)
    weight = DataFrame(date=molecule, weight=zeros(length(molecule)))
    for i in 1:size(weight)[1]
        startTime, endTime = eventsFiltered.date[i], eventsFiltered.timestamp[i]
        concurrencyEventsForSpecificTime = concurrencyEvents[(concurrencyEvents.date .>= startTime) .& (concurrencyEvents.date .<= endTime), :concurrency]
        weight[i, :weight] = mean(1 ./ concurrencyEventsForSpecificTime)
    end
    return weight
end

"""
    createIndexMatrix(
        barIndex::Vector{Date},
        timestamp::DataFrame
    )::Matrix

Creates index matrix.

Creates an index matrix that shows whether an index is in a time horizon or not for meta-labeling.

# Arguments
- `barIndex::Vector{Date}`: Index of all data.
- `timestamp::DataFrame`: Times of events containing starting and ending time.

# Returns
- `Matrix`: Index matrix.
"""
function createIndexMatrix(
        barIndex::Vector{Date},
        timestamp::DataFrame
    )::Matrix
    indexMatrix = zeros(Int, (length(barIndex), size(timestamp)[1]))
    for (j, (t0, t1)) in enumerate(timestamp)
        indicator = [(i <= t1 && i >= t0) ? 1 : 0 for i in barIndex]
        indexMatrix[!, j] = indicator
    end
    return indexMatrix
end

"""
    computeAverageUniqueness(
        indexMatrix::Matrix
    )::Vector

Computes average uniqueness.

Computes the average uniqueness for meta-labeling.

# Arguments
- `indexMatrix::Matrix`: Matrix that indicates events.

# Returns
- `Vector`: Average uniqueness for each event.
"""
function computeAverageUniqueness(
        indexMatrix::Matrix
    )::Vector
    concurrency = sum(indexMatrix, dims=2)
    uniqueness = copy(indexMatrix)
    for i in 1:size(indexMatrix)[1]
        if concurrency[i] > 0
            uniqueness[i, :] = uniqueness[i, :] / concurrency[i]
        end
    end
    averageUniqueness = zeros(size(indexMatrix)[2])
    for i in 1:size(indexMatrix)[2]
        averageUniqueness[i] = sum(uniqueness[:, i]) / sum(indexMatrix[:, i])
    end
    return averageUniqueness
end

"""
    performSequentialBootstrap(
        indexMatrix::Matrix,
        sampleLength::Int
    )::Vector

Performs sequential bootstrap.

Performs sequential bootstrap for meta-labeling.

# Arguments
- `indexMatrix::Matrix`: Matrix that indicates events.
- `sampleLength::Int`: Number of samples.

# Returns
- `Vector`: Sequence of indices for sequential bootstrap.
"""
function performSequentialBootstrap(
        indexMatrix::Matrix,
        sampleLength::Int
    )::Vector
    if isnan(sampleLength)
        sampleLength = size(indexMatrix)[2]
    end
    ϕ = []
    while length(ϕ) < sampleLength
        averageUniqueness = zeros(size(indexMatrix)[2])
        for i in 1:size(indexMatrix)[2]
            tempIndexMatrix = indexMatrix[:, vcat(ϕ, i)]
            averageUniqueness[i] = computeAverageUniqueness(tempIndexMatrix)[end]
        end
        probability = averageUniqueness / sum(averageUniqueness)
        append!(ϕ, sample(1:size(indexMatrix)[2], Weights(probability)))
    end
    return ϕ
end

"""
    calculateSampleWeightWithReturns(
        timestamp::DataFrame,
        concurrencyEvents::DataFrame,
        returns::DataFrame,
        molecule::Vector{Date}
    )::DataFrame

Computes sample weight with returns.

Computes the sample weight with returns for meta-labeling.

# Arguments
- `timestamp::DataFrame`: DataFrame for events.
- `concurrencyEvents::DataFrame`: DataFrame that contains the number of concurrent events for each event.
- `returns::DataFrame`: Data frame that contains returns.
- `molecule::Vector{Date}`: Molecule.

# Returns
- `DataFrame`: Result of the specified function.
"""
function calculateSampleWeightWithReturns(
        timestamp::DataFrame,
        concurrencyEvents::DataFrame,
        returns::DataFrame,
        molecule::Vector{Date}
    )::DataFrame
    eventsFiltered = filter(row -> row[:date] in molecule, timestamp)
    weight = DataFrame(date=molecule, weight=zeros(length(molecule)))
    priceReturn = copy(returns)
    priceReturn.returns = log.(returns.returns .+ 1)
    for i in 1:size(weight)[1]
        startTime, endTime = eventsFiltered.date[i], eventsFiltered.timestamp[i]
        concurrencyEventsForSpecificTime = concurrencyEvents[(concurrencyEvents.date .>= startTime) .& (concurrencyEvents.date .<= endTime), :concurrency]
        returnForSpecificTime = priceReturn[(priceReturn.date .>= startTime) .& (priceReturn.date .<= endTime), :returns]
        weight.weight[i] = sum(returnForSpecificTime ./ concurrencyEventsForSpecificTime)
    end
    weight.weight = abs.(weight.weight)
    return weight
end

"""
    calculateTimeDecay(
        weight::DataFrame,
        clfLastW::Float64=1.0
    )::DataFrame

Computes time decay.

Computes the time decay for meta-labeling.

# Arguments
- `weight::DataFrame`: Weight that is computed for each event.
- `clfLastW::Float64=1.0`: Weight of the oldest observation.

# Returns
- `DataFrame`: Time decay result.
"""
function calculateTimeDecay(
        weight::DataFrame,
        clfLastW::Float64=1.0
    )::DataFrame
    timeDecay = sort(weight, [:date])
    timeDecay[!, :timeDecay] = cumsum(timeDecay[!, :weight])
    slope = 0.0
    if clfLastW >= 0
        slope = (1 - clfLastW) / timeDecay.timeDecay[end]
    else
        slope = 1.0 / ((clfLastW + 1) * timeDecay.timeDecay[end])
    end
    constant = 1.0 - slope * timeDecay.timeDecay[end]
    timeDecay.timeDecay = slope .* timeDecay.timeDecay .+ constant
    timeDecay.timeDecay[timeDecay.timeDecay .< 0] .= 0
    timeDecay = select!(timeDecay, Not(:weight))
    return timeDecay
end
