using Combinatorics, TimeSeries, DataFrames

"""
    purgedKFoldCombinatorialSplit(
        self::PurgedKFoldCombinatorial,
        data::TimeArray
    ) -> Tuple{Vector{Vector{Int}}, PurgedKFoldCombinatorial}

Split the sample using combinatorial purged k-fold cross-validation.

This function takes in a PurgedKFoldCombinatorial struct and a TimeArray sample.
It generates combinatorial training and test splits based on the parameters 
defined in the struct and updates the struct with the generated paths.

:param self: PurgedKFoldCombinatorial struct containing observation and split information.
:type self: PurgedKFoldCombinatorial
:param data: The TimeArray sample that is going to be split.
:type data: TimeArray

:returns: A tuple containing a list of training and test index pairs and the updated PurgedKFoldCombinatorial struct.
:rtype: Tuple{Vector{Vector{Int}}, PurgedKFoldCombinatorial}

.. math::
    \\text{Embargo size} = \\text{round}(\\text{length}(data) \\times \\text{percentEmbargo})
"""
function purgedKFoldCombinatorialSplit(
    self::PurgedKFoldCombinatorial,
    data::TimeArray
) -> Tuple{Vector{Vector{Int}}, PurgedKFoldCombinatorial}

    if timestamp(data) != timestamp(self.times)
        error("data and times must have the same index.")
    end

    self.backtestPaths = Vector{Dict{String, Any}}()
    
    testRanges = [(i[1], i[end]) for i in [kfolds(collect(1:length(self.times)), k = self.nSplits)[i][2] for i in 1:self.nSplits]]
    
    splitIndices = Dict{Int, Tuple{Int, Int}}()
    for (index, range) in enumerate(testRanges)
        splitIndices[index] = range
    end
    
    combinatorialSplits = collect(combinations(sort!(collect(keys(splitIndices))), self.nTestSplits))
    
    combinatorialTestRanges = Vector{Vector{Tuple{Int, Int}}}()
    
    for combination in combinatorialSplits
        testIndices = [splitIndices[index] for index in combination]
        push!(combinatorialTestRanges, testIndices)
    end
    
    for _ in 1:self.nBacktestPaths
        path = [Dict("Train" => nothing, "Test" => testRange) for testRange in values(splitIndices)]
        push!(self.backtestPaths, path)
    end

    embargoSize = round(Int, length(data) * self.percentEmbargo)
    
    finalTest = Vector{Vector{Int}}()
    finalTrain = Vector{Vector{Int}}()
    
    for testSplits in combinatorialTestRanges
        testTimes = calculateTestTimes(self.times, testSplits, embargoSize, length(data))
        
        testIndices = collect.(Iterators.flatten(testSplits))
        
        trainTimes = purgedTrainTimes(self.times, testTimes)
        
        trainIndices = findall(x -> x in timestamp(trainTimes), timestamp(self.times))
        
        updateBacktestPaths!(self.backtestPaths, testSplits, trainIndices, testIndices)
        
        push!(finalTest, testIndices)
        push!(finalTrain, trainIndices)
    end

    return (finalTrain, finalTest), self
end

# Helper function to update backtestPaths
function updateBacktestPaths!(
        backtestPaths::Vector{Dict{String, Any}},
        testSplits::Vector{Tuple{Int, Int}},
        trainIndices::Vector{Int},
        testIndices::Vector{Int}
    )
    for split in testSplits
        found = false
        for path in backtestPaths
            for subPath in path
                if isnothing(subPath["Train"]) && split == subPath["Test"] && !found
                    subPath["Train"] = trainIndices
                    subPath["Test"] = testIndices
                    found = true
                end
            end
        end
    end
end

# Helper function to calculate test times after applying embargo
function calculateTestTimes(times::TimeArray, testSplits::Vector{Tuple{Int, Int}}, embargoSize::Int, dataLength::Int)
    testTimeData = [(times[startIndex], (endIndex + embargoSize > dataLength) ? maximum(values(times[startIndex:endIndex])) : maximum(values(times[startIndex:endIndex + embargoSize]))) for (startIndex, endIndex) in testSplits]
    return TimeArray(DataFrame(timestamp = first.(testTimeData), data = last.(testTimeData)), timestamp = :timestamp)
end
