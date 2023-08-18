using TimeSeries
using DataFrames
using Statistics
using MLJ
using MLDataUtils

"""
.. function:: purgeTrainingTimes(data::TimeArray, test::TimeArray)::TimeArray

    Remove test observations from the training set.

    This function removes observations from the training data that fall within the time periods covered by the test data.

    :param data: Time series data for training.
    :type data: TimeArray
    :param test: Time series data for testing.
    :type test: TimeArray

    :returns: Training data with test observations removed.
    :rtype: TimeArray

    **Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 106, snippet 7.1
"""
function purgeTrainingTimes(
    data::TimeArray,
    test::TimeArray
)::TimeArray

    trainingTimes = deepcopy(data)

    for (startTime, endTime) in zip(timestamp(test), values(test)[:, 1])
        startWithinTestTimes = filter(t -> t >= startTime && t <= endTime, timestamp(trainingTimes))
        endWithinTestTimes = filter(t -> values(trainingTimes)[findfirst(==(t), timestamp(trainingTimes)), 1] >= startTime && values(trainingTimes)[findfirst(==(t), timestamp(trainingTimes)), 1] <= endTime, timestamp(trainingTimes))
        envelopeTestTimes = filter(t -> t <= startTime && values(trainingTimes)[findfirst(==(t), timestamp(trainingTimes)), 1] >= endTime, timestamp(trainingTimes))
        filteredTimes = setdiff(timestamp(trainingTimes), union(startWithinTestTimes, endWithinTestTimes, envelopeTestTimes))
        trainingTimes = trainingTimes[filteredTimes]
    end

    return trainingTimes
end

"""
.. function:: getEmbargoTimes(times::Array{DateTime}, percentEmbargo::Float64)::TimeArray

    Calculate the embargo time for each bar based on a given percentage.

    This function calculates the embargo time for each bar by adding a certain percentage of the total bars as embargo time to each bar.

    :param times: Array of bar times.
    :type times: Array{DateTime}
    :param percentEmbargo: Percentage of embargo.
    :type percentEmbargo: Float64

    :returns: Array of embargo times.
    :rtype: TimeArray

    **Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 108, snippet 7.2

    .. math::

        \text{step} = \text{round}\left(\text{length}(\text{times}) \times \text{percentEmbargo}\right)
"""
function getEmbargoTimes(
        times::Array{DateTime},
        percentEmbargo::Float64
    )::TimeArray

    step = round(Int, length(times) * percentEmbargo)

    if step == 0
        embargo = TimeArray((Times = times, Timestamp = times), timestamp = :Timestamp)
    else
        embargo = TimeArray((Times = times[step + 1:end], Timestamp = times[1:end - step]), timestamp = :Timestamp)
        tailTimes = TimeArray((Times = repeat([times[end]], step), Timestamp = times[end - step + 1:end]), timestamp = :Timestamp)
        embargo = [embargo; tailTimes]
    end

    return embargo
end


"""
.. struct:: PurgedKFold

    Custom struct for cross-validation with purging of overlapping observations.

    **Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 109, snippet 7.3
"""
mutable struct PurgedKFold
nSplits::Int64
times::TimeArray
percentEmbargo::Float64

    """
    .. function:: PurgedKFold(nSplits::Int = 3, times::TimeArray = nothing, percentEmbargo::Float64 = 0.0)

        Create a new PurgedKFold instance.

        :param nSplits: Number of splits.
        :type nSplits: Int
        :param times: TimeArray of data times.
        :type times: TimeArray
        :param percentEmbargo: Percentage of embargo.
        :type percentEmbargo: Float64

        :returns: A new PurgedKFold instance.
        :rtype: PurgedKFold
    """
    function PurgedKFold(
        nSplits::Int = 3,
        times::TimeArray = nothing,
        percentEmbargo::Float64 = 0.0
    )
    
        times isa TimeArray ? new(nSplits, times, percentEmbargo) : error("The times parameter should be a TimeArray.")
    end
end

"""
.. function:: purgedKFoldSplit(self::PurgedKFold, data::TimeArray)

    Split data when observations overlap with purging.

    **Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 109, snippet 7.3
"""
function purgedKFoldSplit(
    self::PurgedKFold,
    data::TimeArray
)::Tuple{Vector{Vector{Int64}}, Vector{Vector{Int64}}}

    if timestamp(data) != timestamp(self.times)
        error("data and times must have the same index.")
    end

    indices = collect(1:length(data))
    embargo = round(Int64, length(data) * self.percentEmbargo)
    finalTest = Vector{Vector{Int64}}()
    finalTrain = Vector{Vector{Int64}}()

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

        push!(finalTest, Int64.(testIndices))
        push!(finalTrain, Int64.(trainIndices))
    end

    return (finalTrain, finalTest)
end

"""
.. function:: crossValidationScore(
        classifier,
        data::TimeArray,
        labels::TimeArray,
        sampleWeights::Vector{Float64},
        scoring::String = "LogLoss",
        times::TimeArray = nothing,
        crossValidationGenerator::PurgedKFold = nothing,
        nSplits::Int = nothing,
        percentEmbargo::Float64 = 0.0)

    Function to calculate cross-validation scores with purging.

    **Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 110, snippet 7.4

    :param classifier: The classifier model.
    :param data: The TimeArray with the data.
    :type data: TimeArray
    :param labels: The TimeArray with the labels.
    :type labels: TimeArray
    :param sampleWeights: The sample weights.
    :type sampleWeights: Vector{Float64}
    :param scoring: The scoring method. Options are "LogLoss" or "Accuracy".
    :type scoring: String
    :param times: The TimeArray with the times. If not provided, it will be created from the data.
    :type times: TimeArray
    :param crossValidationGenerator: The cross-validation generator. If not provided, a default PurgedKFold will be created.
    :type crossValidationGenerator: PurgedKFold
    :param nSplits: The number of splits for the cross-validation generator.
    :type nSplits: Int
    :param percentEmbargo: The percentage of embargo.
    :type percentEmbargo: Float64

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
    nSplits::Int = nothing,
    percentEmbargo::Float64 = 0.0
)::Vector{Float64}
    
    if scoring ∉ ["LogLoss", "Accuracy"]
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

    machine = machine(classifier, data, labels)
    scores = Vector{Float64}()

    for (train, test) in purgedKFoldSplit(crossValidationGenerator, times)
        fit!(machine, rows = train)

        if scoring == "LogLoss"
            predictions = predict(machine, data[test, :])
            score = mean(log_loss(predictions, labels[test]))
        else
            predictions = predict_mode(machine, data[test, :])
            score = mean(accuracy(predictions, labels[test], sampleWeights[test]))
        end

        push!(scores, score)
    end

    return scores
end
