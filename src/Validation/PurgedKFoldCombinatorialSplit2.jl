using TimeSeries, DataFrames, Combinatorics

"""
    purgedKFoldCombinatorialSplit(
        self::PurgedKFoldCombinatorialStacked,
        assetsData::Dict{String, TimeArray}
    ) -> Tuple{Vector{Dict{String, Vector{Int}}}, PurgedKFoldCombinatorialStacked}

Purge observations from k-fold splits in a combinatorial fashion for multiple assets.

This function purges observations that overlap with test-label intervals in a combinatorial manner for
multiple assets. It updates the `backtestPaths` field of the `PurgedKFoldCombinatorialStacked` struct with the
train and test indices.

:param self: The PurgedKFoldCombinatorialStacked struct containing observations and split information.
:type self: PurgedKFoldCombinatorialStacked
:param assetsData: The dictionary of samples that are going to be split.
:type assetsData: Dict{String, TimeArray}

:returns: A tuple containing the final train and test indices, as well as the updated `PurgedKFoldCombinatorialStacked` struct.
:rtype: Tuple{Vector{Dict{String, Vector{Int}}}, PurgedKFoldCombinatorialStacked}
"""

function purgedKFoldCombinatorialSplit(
        self::PurgedKFoldCombinatorialStacked,
        assetsData::Dict{String, TimeArray}
    ) -> Tuple{Vector{Dict{String, Vector{Int}}}, PurgedKFoldCombinatorialStacked}

    firstAsset = first(keys(assetsData))

    for asset in keys(assetsData)
        if timestamp(assetsData[asset]) != timestamp(self.times[asset])
            error("Data and times must have the same index.")
        end
    end

    assetTestRanges = Dict()
    assetSplitIndices = Dict()
    assetCombinatorialTestRanges = Dict()

    for asset in keys(assetsData)
        assetTestRanges[asset] = [(i[1], i[end]) for i in [kfolds(collect(1:length(self.times[asset])), k = self.nSplits)[i][2] for i in 1:self.nSplits]]
        self.backtestPaths[asset] = []
        splitsIndices = Dict([(index, (startIndex, endIndex)) for (index, (startIndex, endIndex)) in enumerate(assetTestRanges[asset])])
        assetSplitIndices[asset] = splitsIndices
        assetCombinatorialTestRanges[asset] = collect(combinations(sort(keys(splitsIndices)), self.nTestSplits))
    end

    finalTest = []
    finalTrain = []

    for i in 1:length(assetCombinatorialTestRanges[firstAsset])
        assetsTrainIndices = Dict()
        assetsTestIndices = Dict()

        for (asset, data) in assetsData
            embargo = round(Int, length(data) * self.percentEmbargo)
            testSplits = assetCombinatorialTestRanges[asset][i]
            testTimes = purgedTestTimes(self, asset, data, testSplits, embargo)
            trainTimes = purgedTrainTimes(self.times[asset], testTimes)
            trainIndices = getIndices(self.times[asset], trainTimes)
            testIndices = [collect(startIndex:endIndex) for (startIndex, endIndex) in testSplits]
            assetsTrainIndices[asset] = trainIndices
            assetsTestIndices[asset] = testIndices
            fillBacktestPaths!(self, asset, trainIndices, testSplits)
        end

        push!(finalTest, assetsTestIndices)
        push!(finalTrain, assetsTrainIndices)
    end

    return (zip(finalTrain, finalTest), self)
end

# Assuming purgedTestTimes, purgedTrainTimes, getIndices, fillBacktestPaths! are helper functions defined elsewhere.

