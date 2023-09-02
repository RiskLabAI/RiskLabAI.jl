using MLJ, TimeSeries, DataFrames  # Required packages

"""
    crossValidationScore(
        classifier,
        data::TimeArray,
        labels::TimeArray,
        sampleWeights::Vector{Float64},
        scoring::String = "LogLoss",
        times::TimeArray = nothing,
        crossValidationGenerator::PurgedKFold = nothing,
        numSplits::Int = nothing,
        percentEmbargo::Float64 = 0.0
    )::Vector{Float64}

Calculate cross-validation scores with purging.

- **Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 110, snippet 7.4

:param classifier: The classifier model.
:param data: The TimeArray with the data.
:param labels: The TimeArray with the labels.
:param sampleWeights: The sample weights.
:param scoring: The scoring method. Options are "LogLoss" or "Accuracy".
:param times: The TimeArray with the times. If not provided, it will be created from the data.
:param crossValidationGenerator: The cross-validation generator. If not provided, a default PurgedKFold will be created.
:param numSplits: The number of splits for the cross-validation generator.
:param percentEmbargo: The percentage of embargo.

:returns: The cross-validation scores.
:rtype: Vector{Float64}
"""
function crossValidationScore(
    classifier,
    data::TimeArray,
    labels::TimeArray,
    sampleWeights::Vector{Float64},
    scoring::String = "LogLoss",
    times::TimeArray = nothing,
    crossValidationGenerator::PurgedKFold = nothing,
    numSplits::Int = nothing,
    percentEmbargo::Float64 = 0.0
)::Vector{Float64}
    
    if !(scoring in ["LogLoss", "Accuracy"])
        error("Invalid scoring method.")
    end

    times = times === nothing ? createTimeArray(data) : times
    crossValidationGenerator = crossValidationGenerator === nothing ? PurgedKFold(numSplits, times, percentEmbargo) : crossValidationGenerator

    labelName = colnames(labels)[1]
    data, labels = DataFrame(data), DataFrame(labels)
    select!(data, Not(:timestamp))
    select!(labels, Not(:timestamp))

    sampleData = hcat(data, labels)
    sampleData = coerceTypes(sampleData)

    labels, data = unpack(sampleData, ==(labelName), colname -> true)

    machineModel = machine(classifier, data, labels)
    scores = Vector{Float64}()

    for (trainIndices, testIndices) in purgedKFoldSplit(crossValidationGenerator, times)
        fit!(machineModel, rows = trainIndices)
        score = calculateScore(machineModel, data, labels, sampleWeights, scoring, testIndices)
        push!(scores, score)
    end

    return scores
end

# Utility functions to help maintain code clarity and modularity

function createTimeArray(data::TimeArray)
    startTime = timestamp(data)
    endTime = append!(startTime[2:end], [startTime[end] + (startTime[2] - startTime[1])])
    return TimeArray(DataFrame(startTime = startTime, endTime = endTime), timestamp = :startTime)
end

function coerceTypes(sample::DataFrame)
    for column in names(sample)
        if eltype(sample[!, column]) in [Float64, Int]
            sample = coerce(sample, Symbol(column) => Continuous)
        else
            sample = coerce(sample, Symbol(column) => Multiclass)
        end
    end
    return sample
end

function calculateScore(machineModel, data, labels, sampleWeights, scoring, testIndices)
    if scoring == "LogLoss"
        predictions = predict(machineModel, data[testIndices, :])
        return mean(log_loss(predictions, labels[testIndices]))
    else
        predictions = predict_mode(machineModel, data[testIndices, :])
        return mean(accuracy(predictions, labels[testIndices], sampleWeights[testIndices]))
    end
end
