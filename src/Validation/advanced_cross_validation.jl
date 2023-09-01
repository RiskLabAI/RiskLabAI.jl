using TimeSeries
using DataFrames
using Statistics
using MLJ
using MLDataUtils
include("Chapter 7 CV in Finance Snippets.jl")

mutable struct PurgedKFoldStacked
    # Modified KFold class to work with labels that span intervals for multiple assets 
    # The train is purged of observations overlapping test-label intervals

    nSplits::Int64 # The number of KFold splits
    times::Dict # Dictionary of entire observation times for multiple assets
    percentEmbargo::Float64 # Embargo size percentage divided by 100

    # timestamp(TimeArray): Time when the observation started.
    # values(TimeArray)[:, 1]: Time when the observation ended.

    function PurgedKFoldStacked(nSplits::Int = 3, # The number of KFold splits
                                times::Dict = nothing, # Dictionary of entire observation times for multiple assets
                                percentEmbargo::Float64 = 0.) # Embargo size percentage divided by 100
        
        times isa Dict ? new(nSplits, times, percentEmbargo) : error("The times parameter should be a Dictionary of TimeArrays.") # return struct if "times" is a Dictionary of TimeArrays
    end
end

function purgedKFoldStackedSplit(self::PurgedKFoldStacked, # The PurgedKFoldStacked struct containing observations and split information
                                 assetsData::Dict) # The dictionary of samples that are going be splited

    firstAsset = collect(keys(assetsData))[1] # get first asset data

    for asset ∈ keys(assetsData) # iterate over assets
        if timestamp(assetsData[asset]) != timestamp(self.times[asset]) # check if data and times have the same index (starting time)
            error("data and ThruDateValues must have the same index.") # raise error
        end
    end

    testRanges = Dict() # initialize test indices
    for asset ∈ keys(assetsData) # iterate over assets
        testRanges[asset] = [(i[1], i[end]) for i ∈ [kfolds(collect(1:length(self.times[asset])), k = self.nSplits)[i][2] for i ∈ collect(1:self.nSplits)]] # get all test indices
    end

    finalTest = [] # initialize final test indices
    finalTrain = [] # initialize final train indices

    for i ∈ 1:length(testRanges[firstAsset]) # iterate over folds
        assetsTrainIndices = Dict()  # initialize dictionary of train indices
        assetsTestIndices = Dict() # initialize dictionary of test indices

        for (asset, data) in assetsData # iterate over assets
            indices = collect(1:length(data)) # get data positions
            embargo = round(Int64, length(data)*self.percentEmbargo) # get embargo size

            startIndex, endIndex = testRanges[asset][i] # get start and end of test indices

            firstTestIndex = timestamp(self.times[asset])[startIndex] # get the start of the current test set
            testIndices = indices[startIndex:endIndex] # get test indices for current split
            maxTestIndex = searchsortedfirst(timestamp(self.times[asset]), maximum(values(self.times[asset])[testIndices, 1])) # get the farthest test index
            searchTimes(y) = findfirst(x -> x == y, timestamp(self.times[asset])) # create function to find train indices
            trainIndices = searchTimes.(timestamp(self.times[asset])[values(self.times[asset])[:, 1] .<= firstTestIndex]) # find the left side of the training data

            if maxTestIndex + embargo <= length(values(data))
                append!(trainIndices, indices[maxTestIndex + embargo:end]) # find the right side of the training data with embargo
            end

            assetsTrainIndices[asset] = Int64.(trainIndices) # add asset train indices
            assetsTestIndices[asset] = Int64.(testIndices) # add asset test indices
        end

        append!(finalTest, [assetsTrainIndices]) # append test indices to the final list
        append!(finalTrain, [assetsTestIndices]) # append train indices to the final list
    end

    return Tuple(zip(finalTrain, finalTest))
end

function crossValidationScore(classifier, # A classifier model from MLJ package 
                              assetsData::Dict, # Dictionary of samples that are going be used
                              assetsLabels::Dict, # Dictionary of labels that are going to be used
                              sampleWeights::Dict, # Dictionary of sample weights for the classifier
                              scoring::String = "Log Loss", # Scoring type: ["Log Loss", "Accuracy"]
                              times::Dict = nothing, # Dictionary of entire observation times for multiple assets
                              crossValidationGenerator::PurgedKFold = nothing, # The PurgedKFoldStacked struct containing observations and split information
                              nSplits::Int = nothing, # The number of KFold splits
                              percentEmbargo::Float64 = 0.0) # Embargo size percentage divided by 100

    if scoring ∉ ["Log Loss", "Accuracy"] # check if the scoring method is correct
        error("Wrong scoring method.") # raise error
    end

    if isnothing(times) # check if the observation time are nothing
        times = Dict()
        for asset ∈ keys(assetsData) # iterate over assets
            times[asset] = TimeArray(DataFrame(startTime = timestamp(assetsData[asset]), endTime = append!(timestamp(assetsData[asset])[2:end], [timestamp(assetsData[asset])[end] + (timestamp(assetsData[asset])[2]-timestamp(data)[1])])), timestamp = :startTime) # initialize
        end
    end

    if isnothing(crossValidationGenerator) # check if the PurgedKFold is nothing
        crossValidationGenerator = PurgedKFoldStacked(nSplits, times, percentEmbargo) # initialize
    end

    assetScores = Dict() # initialize the scores

    for asset ∈ keys(assetsData) # iterate over assets
        name = colnames(assetsLabels[asset])[1] # labels name
        data, labels = DataFrame(assetsData[asset]), DataFrame(assetsLabels[asset]) # convert timearrays to dataframe
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

        # println(scitype(data))
        # println(scitype(labels))
        # println(models(matching(data,labels)))

        machine = MLJ.MLJBase.machine(classifier, data, labels) # initialize learner
        scores = [] # initialize scores

        for (train, test) ∈ purgedKFoldSplit(crossValidationGenerator, times[asset])
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

        assetScores[asset] = scores # add asset scores
    end

    return assetScores
end

function backtestPathsNumber(nTrainSplits::Int, # Number of train splits in sample
                             nTestSplits::Int)::Int # Number of test splits in sample

    return Int64(binomial(nTrainSplits, nTrainSplits - nTestSplits)*nTestSplits / nTrainSplits) # get number of combinatorial backtest paths
end

mutable struct PurgedKFoldCombinatorial
    # Modified PurgedKFold class to work with labels that span intervals in combinatorial fashion
    # The train is purged of observations overlapping test-label intervals in combinatorial fashion
    
    nSplits::Int # The number of combinatorial splits
    nTestSplits::Int # Number of test splits in sample
    times::TimeArray # Entire observation times
    percentEmbargo::Float64 # Embargo size percentage divided by 100
    nBacktestPaths::Int # Number of combinatorial backtest paths
    backtestPaths::Array # Combinatorial backtest paths

    # timestamp(TimeArray): Time when the observation started.
    # values(TimeArray)[:, 1]: Time when the observation ended.

    function PurgedKFoldCombinatorial(nSplits::Int = 3, # The number of combinatorial splits
                                      nTestSplits::Int = 2, # Number of test splits in sample
                                      times::TimeArray = nothing, # Entire observation times
                                      percentEmbargo::Float64 = 0.0) # Embargo size percentage divided by 100
        
        times isa TimeArray ? 
        new(nSplits, nTestSplits, times, percentEmbargo, backtestPathsNumber(nSplits, nTestSplits), []):
        error("The times parameter must be a TimeArray.") # return struct if "times" is a TimeArray
    end
end    

function purgedKFoldCombinatorialSplit(self::PurgedKFoldCombinatorial, # The PurgedKFoldCombinatorial struct containing observations and split information
                                       data::TimeArray) # The sample that is going be splited
    
    if timestamp(data) != timestamp(self.times) # check if data and times have the same index (starting time)
        error("data and ThruDateValues must have the same index.") # raise error
    end

    self.backtestPaths = []  # initialize backtest paths as empty 

    testRanges = [(i[1], i[end]) for i ∈ [kfolds(collect(1:length(self.times)), k = self.nSplits)[i][2] for i ∈ collect(1:self.nSplits)]] # get all folds

    splitsIndices = Dict() # initialize splits ranges dictionary
    for (index, (startIndex, endIndex)) ∈ enumerate(testRanges)
        splitsIndices[index] = [startIndex, endIndex] # assign the test range with index
    end

    combinatorialSplits = collect(combinations(sort(keys(splitsIndices)), self.nTestSplits)) # get all combinations of self.nTestSplits splits in all splits
    combinatorialTestRanges = []  # initialize combinatorial test ranges

    for combination in combinatorialSplits
        testIndices = []  # initialize test indices for our combination

        for index in combination
            append!(testIndices, [splitsIndices[index]]) # append test ranges of current split to the current combination list
        end

        append!(combinatorialTestRanges, [testIndices]) # append test ranges of current combination to the combinations list
    end

    for _ ∈ 1:self.nBacktestPaths
        path = [] # initialize path
        for splitIndex ∈ values(splitsIndices)
            append!(path, [Dict("Train"=> nothing, "Test"=> splitIndex)]) # create path
        end

        append!(self.backtestPaths, [path]) # append path to list
    end

    embargo = round(Int64, length(data)*self.percentEmbargo) # get embargo size

    finalTest = [] # initialize final test indices
    finalTrain = [] # initialize final train indices

    for testSplits ∈ combinatorialTestRanges

        testTimes = DataFrame(timestamp = [self.times[index[1]] for index ∈ testSplits], data = [  
            begin 
                if index[2] + embargo > length(data)
                    maximum(values(self.times[index[1]:index[2]]))
                else
                    maximum(values(self.times[index[1]:index[2] + embargo]))
                end
            end     
            for index ∈ testSplits]) # perform embargo on sample times

        testTimes = TimeArray(testTimes, timestamp = :timestamp) # convert to TimeArray 

        testIndices = [] # initialize test indices
        for (startIndex, endIndex) in testSplits
            append!(testIndices, [collect(startIndex:endIndex)]) # collect test indices
        end

        trainTimes = purgedTrainTimes(self.times, testTimes) # purge test times from observations to get train times

        trainIndices = [] # initialize train indices
        for trainIndex ∈ timestamp(trainTimes)
            append!(trainIndices, timestamp(self.times)[timestamp(self.times) .== trainIndex]) # collect train indices
        end

        for split in testSplits
            found = false  # indicate that the split was not found and filled
            for path in self.backtestPaths
                for subPath in path
                    if isnothing(subPath["Train"]) && split == subPath["Test"] && found == false
                        subPath["Train"] = trainIndices # set the train indices in path
                        subPath["Test"] = collect(split[1]:split[end]) # set the test indices in path
                        found = true # indicate that the split was found and filled
                    end
                end
            end
        end

        append!(finalTest, [testIndices]) # append test indices to the final list
        append!(finalTrain, [trainIndices]) # append train indices to the final list
    end

    return (Tuple(zip(finalTrain, finalTest)), self)
end

mutable struct PurgedKFoldCombinatorialStacked
    # Modified PurgedKFold class to work with labels that span intervals in combinatorial fashion for multiple assets
    # The train is purged of observations overlapping test-label intervals in combinatorial fashion
    
    nSplits::Int # The number of combinatorial splits
    nTestSplits::Int # Number of test splits in sample
    times::Dict # Dictionary of entire observation times for multiple assets
    percentEmbargo::Float64 # Embargo size percentage divided by 100
    nBacktestPaths::Int # Number of combinatorial backtest paths
    backtestPaths::Dict # Combinatorial backtest paths

    # timestamp(TimeArray): Time when the observation started.
    # values(TimeArray)[:, 1]: Time when the observation ended.

    function PurgedKFoldCombinatorialStacked(nSplits::Int = 3, # The number of combinatorial splits
                                      nTestSplits::Int = 2, # Number of test splits in sample
                                      times::Dict = nothing, # Dictionary of entire observation times for multiple assets
                                      percentEmbargo::Float64 = 0.0) # Embargo size percentage divided by 100
        
        times isa Dict ? 
        new(nSplits, nTestSplits, times, percentEmbargo, backtestPathsNumber(nSplits, nTestSplits), Dict()):
        error("The times parameter should be a Dictionary of TimeArrays.") # return struct if "times" is a Dictionary of TimeArrays
    end
end    

function purgedKFoldCombinatorialSplit(self::PurgedKFoldCombinatorialStacked, # The PurgedKFoldCombinatorialStacked struct containing observations and split information
                                       assetsData::Dict) # The dictionary of samples that are going be splited

    firstAsset = collect(keys(assetsData))[1] # get first asset data

    for asset ∈ keys(assetsData) # iterate over assets
        if timestamp(assetsData[asset]) != timestamp(self.times[asset]) # check if data and times have the same index (starting time)
            error("data and ThruDateValues must have the same index.") # raise error
        end
    end                        
    
    assetTestRanges = Dict() # initialize test indices
    for asset ∈ keys(assetsData) # iterate over assets
        assetTestRanges[asset] = [(i[1], i[end]) for i ∈ [kfolds(collect(1:length(self.times[asset])), k = self.nSplits)[i][2] for i ∈ collect(1:self.nSplits)]] # get all test indices
        self.backtestPaths[asset] = [] # initialize asset backtest paths
    end

    assetSplitIndices = Dict() # initialize asset splits ranges
    assetCombinatorialTestRanges = Dict() # initialize asset combinatorial test ranges
    for asset ∈ keys(assetsData) # iterate over assets
        splitsIndices = Dict() # initialize splits ranges dictionary
        for (index, (startIndex, endIndex)) ∈ enumerate(assetTestRanges[asset])
            splitsIndices[index] = [startIndex, endIndex] # assign the test range with index
        end

        assetSplitIndices[asset] = splitsIndices # set asset splits ranges

        combinatorialSplits = collect(combinations(sort(keys(splitsIndices)), self.nTestSplits)) # get all combinations of self.nTestSplits splits in all splits
        combinatorialTestRanges = []  # initialize combinatorial test ranges

        for combination in combinatorialSplits
            testIndices = []  # initialize test indices for our combination

            for index in combination
                append!(testIndices, [splitsIndices[index]]) # append test ranges of current split to the current combination list
            end

            append!(combinatorialTestRanges, [testIndices]) # append test ranges of current combination to the combinations list
        end

        assetCombinatorialTestRanges[asset] = combinatorialTestRanges # set asset combinatorial test ranges
    end

    for asset ∈ keys(assetsData) # iterate over assets
        for _ ∈ 1:self.nBacktestPaths
            path = [] # initialize path
            for splitIndex ∈ values(assetSplitIndices[asset])
                append!(path, [Dict("Train"=> nothing, "Test"=> splitIndex)]) # create path
            end
    
            append!(self.backtestPaths[asset], [path]) # append path to list
        end
    end

    finalTest = [] # initialize final test indices
    finalTrain = [] # initialize final train indices

    for i ∈ 1:length(assetCombinatorialTestRanges[firstAsset]) # iterate over folds
        assetsTrainIndices = Dict()  # initialize dictionary of train indices
        assetsTestIndices = Dict() # initialize dictionary of test indices

        for (asset, data) in assetsData # iterate over assets
            embargo = round(Int64, length(data)*self.percentEmbargo) # get embargo size
            testSplits = assetCombinatorialTestRanges[asset][i] # get asset test split

            testTimes = DataFrame(timestamp = [self.times[asset][index[1]] for index ∈ testSplits], data = [  
            begin 
                if index[2] + embargo > length(data)
                    maximum(values(self.times[asset][index[1]:index[2]]))
                else
                    maximum(values(self.times[asset][index[1]:index[2] + embargo]))
                end
            end     
            for index ∈ testSplits]) # perform embargo on asset sample times

            testTimes = TimeArray(testTimes, timestamp = :timestamp) # convert to TimeArray 

            testIndices = [] # initialize test indices
            for (startIndex, endIndex) in testSplits
                append!(testIndices, [collect(startIndex:endIndex)]) # collect test indices
            end

            trainTimes = purgedTrainTimes(self.times[asset], testTimes) # purge test times from observations to get train times

            trainIndices = [] # initialize train indices
            for trainIndex ∈ timestamp(trainTimes)
                append!(trainIndices, timestamp(self.times[asset])[timestamp(self.times[asset]) .== trainIndex]) # collect train indices
            end

            assetsTrainIndices[asset] = trainIndices # set asset train indices
            assetsTestIndices[asset] = testIndices # set asset test indices

            for split in testSplits
                found = false  # indicate that the split was not found and filled
                for path in self.backtestPaths[asset]
                    for subPath in path
                        if isnothing(subPath["Train"]) && split == subPath["Test"] && found == false
                            subPath["Train"] = trainIndices # set the train indices in path
                            subPath["Test"] = collect(split[1]:split[end]) # set the test indices in path
                            found = true # indicate that the split was found and filled
                        end
                    end
                end
            end
        end

        append!(finalTest, [assetsTestIndices]) # append test indices to the final list
        append!(finalTrain, [assetsTrainIndices]) # append train indices to the final list
    end

    return (Tuple(zip(finalTrain, finalTest)), self)
end



