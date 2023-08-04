using TimeSeries
using DataFrames
using Statistics
using MLJ
using MLDataUtils

"""
    function: purges test observations in the training set
    reference: De Prado, M. (2018) Advances in financial machine learning.
    methodology: page 106, snippet 7.1
"""
function purgedTrainTimes(data::TimeArray, # Times of entire observations.
                          test::TimeArray)::TimeArray # Times of testing observations.
    # timestamp(TimeArray): Time when the observation started.
    # values(TimeArray)[:, 1]: Time when the observation ended.

    trainTimes = deepcopy(data) # get a deep copy of train times

    for (startTime, endTime) ∈ zip(timestamp(test), values(test)[:, 1]) 
        startWithinTestTimes = timestamp(trainTimes)[(timestamp(trainTimes) .>= startTime) .* (timestamp(trainTimes) .<= endTime)] # get times when train starts within test
        endWithinTestTimes = timestamp(trainTimes)[(values(trainTimes)[:, 1] .>= startTime) .* (values(trainTimes)[:, 1] .<= endTime)] # get times when train ends within test
        envelopeTestTimes = timestamp(trainTimes)[(timestamp(trainTimes) .<= startTime) .* (values(trainTimes)[:, 1] .>= endTime)] # get times when train envelops test
        filteredTimes = setdiff(timestamp(trainTimes), union(startWithinTestTimes, endWithinTestTimes, envelopeTestTimes)) # filter timestamps that are going to be purged
        trainTimes = trainTimes[filteredTimes] # purge observations
    end

    return trainTimes
end

"""
    function: gets embargo time for each bar
    reference: De Prado, M. (2018) Advances in financial machine learning.
    methodology: page 108, snippet 7.2
"""
function embargoTimes(times::Array, # Entire observation times
                      percentEmbargo::Float64)::TimeArray # Embargo size percentage divided by 100
    
    step = round(Int, length(times)*percentEmbargo) # find the number of embargo bars

    if step == 0 
        embargo = TimeArray((Times = times, timestamp = times), timestamp = :timestamp) # do not perform embargo when the step equals zero
    else
        embargo = TimeArray((Times = times[step + 1:end], timestamp = times[1:end - step]), timestamp = :timestamp) # find the embargo time for each time
        tailTimes = TimeArray((Times = repeat([times[end]], step), timestamp = times[end - step + 1:end]), timestamp = :timestamp) # find the embargo time for the last "step" number of bars
        embargo = [embargo; tailTimes] # join all embargo times
    end

    return embargo
end

"""
    struct: performes cross validation when observations overlap
    reference: De Prado, M. (2018) Advances in financial machine learning.
    methodology: page 109, snippet 7.3
"""
mutable struct PurgedKFold
    # Modified KFold class to work with labels that span intervals
    # The train is purged of observations overlapping test-label intervals

    nSplits::Int64 # The number of KFold splits
    times::TimeArray # Entire observation times
    percentEmbargo::Float64 # Embargo size percentage divided by 100

    # timestamp(TimeArray): Time when the observation started.
    # values(TimeArray)[:, 1]: Time when the observation ended.

    function PurgedKFold(nSplits::Int = 3, # The number of KFold splits
                         times::TimeArray = nothing, # Entire observation times
                         percentEmbargo::Float64 = 0.) # Embargo size percentage divided by 100
        
        times isa TimeArray ? new(nSplits, times, percentEmbargo) : error("The times parameter should be a TimeArray.") # return struct if "times" is a TimeArray
    end
end

"""
    function: splits the data when observations overlap
    reference: De Prado, M. (2018) Advances in financial machine learning.
    methodology: page 109, snippet 7.3
"""
function purgedKFoldSplit(self::PurgedKFold, # The PurgedKFold struct containing observations and split information
                          data::TimeArray) # The sample that is going be splited

    if timestamp(data) != timestamp(self.times) # check if data and times have the same index (starting time)
        error("data and ThruDateValues must have the same index.") # raise error
    end

    indices = collect(1:length(data)) # get data positions
    embargo = round(Int64, length(data)*self.percentEmbargo) # get embargo size
    finalTest = [] # initialize final test indices
    finalTrain = [] # initialize final train indices

    testRanges = [(i[1], i[end]) for i ∈ [kfolds(collect(1:length(self.times)), k = self.nSplits)[i][2] for i ∈ collect(1:self.nSplits)]] # get all test indices
    
    for (startIndex, endIndex) ∈ testRanges

        firstTestIndex = timestamp(self.times)[startIndex] # get the start of the current test set
        testIndices = indices[startIndex:endIndex] # get test indices for current split
        maxTestIndex = searchsortedfirst(timestamp(self.times), maximum(values(self.times)[testIndices, 1])) # get the farthest test index
        searchTimes(y) = findfirst(x -> x == y, timestamp(self.times)) # create function to find train indices
        trainIndices = searchTimes.(timestamp(self.times)[values(self.times)[:, 1] .<= firstTestIndex]) # find the left side of the training data

        if maxTestIndex + embargo <= length(values(data))
            append!(trainIndices, indices[maxTestIndex + embargo:end]) # find the right side of the training data with embargo
        end

        append!(finalTest, [Int64.(testIndices)]) # append test indices to the final list
        append!(finalTrain, [Int64.(trainIndices)]) # append train indices to the final list
    end

    return Tuple(zip(finalTrain, finalTest))
end


"""
    function: uses the PurgedKFold struct and functions
    reference: De Prado, M. (2018) Advances in financial machine learning.
    methodology: page 110, snippet 7.4
"""
function crossValidationScore(classifier, # A classifier model from MLJ package 
                              data::TimeArray, # The sample that is going be used
                              labels::TimeArray, # The sample labels that are going to be used
                              sampleWeights::Array, # The sample weights for the classifier
                              scoring::String = "Log Loss", # Scoring type: ["Log Loss", "Accuracy"]
                              times::TimeArray = nothing, # Entire observation times
                              crossValidationGenerator::PurgedKFold = nothing, # The PurgedKFold struct containing observations and split information
                              nSplits::Int = nothing, # The number of KFold splits
                              percentEmbargo::Float64 = 0.0) # Embargo size percentage divided by 100

    if scoring ∉ ["Log Loss", "Accuracy"] # check if the scoring method is correct
        error("Wrong scoring method.") # raise error
    end

    if isnothing(times) # check if the observation time are nothing
        times = TimeArray(DataFrame(startTime = timestamp(data), endTime = append!(timestamp(data)[2:end], [timestamp(data)[end] + (timestamp(data)[2]-timestamp(data)[1])])), timestamp = :startTime) # initialize
    end

    if isnothing(crossValidationGenerator) # check if the PurgedKFold is nothing
        crossValidationGenerator = PurgedKFold(nSplits, times, percentEmbargo) # initialize
    end

    name = colnames(labels)[1] # labels name
    data, labels = DataFrame(data), DataFrame(labels) # convert timearrays to dataframe
    select!(data, Not(:timestamp)) # remove timestamp
    select!(labels, Not(:timestamp)) # remove timestamp

    sample = hcat(data, labels) # merge entire data 
    for column ∈ names(sample)
        if eltype(sample[!, column]) == Float64 || eltype(sample[!, column]) == Int # check scitypes
            sample = coerce(sample, Symbol(column) => Continuous) # change scitypes
        else
            sample = coerce(sample, Symbol(column) => Multiclass) # change scitypes
        end    
    end
    labels, data = unpack(sample, ==(name), colname -> true) # unpack data

    machine = MLJ.MLJBase.machine(classifier, data, labels) # initialize learner
    scores = [] # initialize scores

    for (train, test) ∈ purgedKFoldSplit(crossValidationGenerator, times)
        fit!(machine, rows=train) # fit model

        if scoring == "Log Loss"
            predictions = MLJ.MLJBase.predict(machine, data[test, :]) # predict test
            score = log_loss(predictions, labels[test]) |> mean # calculate score
        else
            predictions = MLJ.MLJBase.predict_mode(machine, data[test, :]) # predict test    
            score = accuracy(predictions, labels[test], sampleWeights[test]) |> mean # calculate score
        end

        append!(scores, [score]) # append score
    end

    return scores
end