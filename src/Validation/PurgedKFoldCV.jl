using TimeSeries
using DataFrames
using Statistics
using MLJ
using MLDataUtils

"""
Function to purge test observations in the training set.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: page 106, snippet 7.1
"""
function purgedTrainTimes(data::TimeArray, test::TimeArray)::TimeArray
    trainTimes = deepcopy(data)

    for (startTime, endTime) in zip(timestamp(test), values(test)[:, 1])
        startWithinTestTimes = timestamp(trainTimes)[(timestamp(trainTimes) .>= startTime) .* (timestamp(trainTimes) .<= endTime)]
        endWithinTestTimes = timestamp(trainTimes)[(values(trainTimes)[:, 1] .>= startTime) .* (values(trainTimes)[:, 1] .<= endTime)]
        envelopeTestTimes = timestamp(trainTimes)[(timestamp(trainTimes) .<= startTime) .* (values(trainTimes)[:, 1] .>= endTime)]
        filteredTimes = setdiff(timestamp(trainTimes), union(startWithinTestTimes, endWithinTestTimes, envelopeTestTimes))
        trainTimes = trainTimes[filteredTimes]
    end

    return trainTimes
end

"""
Function to get embargo time for each bar.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: page 108, snippet 7.2
"""
function embargoTimes(times::Array, percentEmbargo::Float64)::TimeArray
    step = round(Int, length(times) * percentEmbargo)

    if step == 0
        embargo = TimeArray((Times = times, timestamp = times), timestamp = :timestamp)
    else
        embargo = TimeArray((Times = times[step + 1:end], timestamp = times[1:end - step]), timestamp = :timestamp)
        tailTimes = TimeArray((Times = repeat([times[end]], step), timestamp = times[end - step + 1:end]), timestamp = :timestamp)
        embargo = [embargo; tailTimes]
    end

    return embargo
end

"""
Custom struct for cross validation with purging of overlapping observations.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: page 109, snippet 7.3
"""
mutable struct PurgedKFold
    nSplits::Int64
    times::TimeArray
    percentEmbargo::Float64

    function PurgedKFold(nSplits::Int = 3, times::TimeArray = nothing, percentEmbargo::Float64 = 0.)
        times isa TimeArray ? new(nSplits, times, percentEmbargo) : error("The times parameter should be a TimeArray.")
    end
end

"""
Function to split data when observations overlap.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: page 109, snippet 7.3
"""
function purgedKFoldSplit(
        self::PurgedKFold,
        data::TimeArray
    )

    if timestamp(data) != timestamp(self.times)
        error("data and ThruDateValues must have the same index.")
    end

    indices = collect(1:length(data))
    embargo = round(Int64, length(data) * self.percentEmbargo)
    finalTest = []
    finalTrain = []

    testRanges = [(i[1], i[end]) for i in [kfolds(collect(1:length(self.times)), k = self.nSplits)[i][2] for i in collect(1:self.nSplits)]]

    for (startIndex, endIndex) in testRanges
        firstTestIndex = timestamp(self.times)[startIndex]
        testIndices = indices[startIndex:endIndex]
        maxTestIndex = searchsortedfirst(timestamp(self.times), maximum(values(self.times)[testIndices, 1]))
        searchTimes(y) = findfirst(x -> x == y, timestamp(self.times))
        trainIndices = searchTimes.(timestamp(self.times)[values(self.times)[:, 1] .<= firstTestIndex])

        if maxTestIndex + embargo <= length(values(data))
            append!(trainIndices, indices[maxTestIndex + embargo:end])
        end

        append!(finalTest, [Int64.(testIndices)])
        append!(finalTrain, [Int64.(trainIndices)])
    end

    return Tuple(zip(finalTrain, finalTest))
end

"""
Function to calculate cross-validation scores with purging.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: page 110, snippet 7.4
"""
function crossValidationScore(
        classifier,
        data::TimeArray,
        labels::TimeArray,
        sampleWeights::Array,
        scoring::String = "Log Loss",
        times::TimeArray = nothing,
        crossValidationGenerator::PurgedKFold = nothing,
        nSplits::Int = nothing,
        percentEmbargo::Float64 = 0.0
    )
    
    if scoring ∉ ["Log Loss", "Accuracy"]
        error("Wrong scoring method.")
    end

    if isnothing(times)
        times = TimeArray(DataFrame(startTime = timestamp(data), endTime = append!(timestamp(data)[2:end], [timestamp(data)[end] + (timestamp(data)[2] - timestamp(data)[1])])), timestamp = :startTime)
    end

    if isnothing(crossValidationGenerator)
        crossValidationGenerator = PurgedKFold(nSplits, times, percentEmbargo)
    end

    name = colnames(labels)[1]
    data, labels = DataFrame(data), DataFrame(labels)
    select!(data, Not(:timestamp))
    select!(labels, Not(:timestamp))

    sample = hcat(data, labels)
    for column in names(sample)
        if eltype(sample[!, column]) == Float64 || eltype(sample[!, column]) == Int
            sample = coerce(sample, Symbol(column) => Continuous)
        else
            sample = coerce(sample, Symbol(column) => Multiclass)
        end
    end
    labels, data = unpack(sample, ==(name), colname -> true)

    machine = MLJ.MLJBase.machine(classifier, data, labels)
    scores = []

    for (train, test) in purgedKFoldSplit(crossValidationGenerator, times)
        fit!(machine, rows = train)

        if scoring == "Log Loss"
            predictions = MLJ.MLJBase.predict(machine, data[test, :])
            score = log_loss(predictions, labels[test]) |> mean
        else
            predictions = MLJ.MLJBase.predict_mode(machine, data[test, :])
            score = accuracy(predictions, labels[test], sampleWeights[test]) |> mean
        end

        append!(scores, [score])
    end

    return scores
end
