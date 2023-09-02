using DataFrames # Required for DataFrame-related functions
using TimeSeries # Required for TimeArray

"""
    purgedKFoldStackedSplit(
        purgedKFoldStruct::PurgedKFoldStacked,
        assetsData::Dict{Symbol, TimeArray}
    ) -> Vector{Tuple{Dict{Symbol, Vector{Int64}}, Dict{Symbol, Vector{Int64}}}}

Generate training and test index sets for each fold in purged k-fold cross-validation
with support for multiple assets and labels that span intervals.

- **Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 110, snippet 7.4

:param purgedKFoldStruct: The `PurgedKFoldStacked` struct containing observation times and split information.
:type purgedKFoldStruct: PurgedKFoldStacked
:param assetsData: The dictionary of TimeArray samples that will be split. The key is the asset name.
:type assetsData: Dict{Symbol, TimeArray}

:return: A vector of tuples, where each tuple contains two dictionaries for the training and test indices.
:rtype: Vector{Tuple{Dict{Symbol, Vector{Int64}}, Dict{Symbol, Vector{Int64}}}}

"""
function purgedKFoldStackedSplit(
        purgedKFoldStruct::PurgedKFoldStacked,
        assetsData::Dict{Symbol, TimeArray}
    )::Vector{Tuple{Dict{Symbol, Vector{Int64}}, Dict{Symbol, Vector{Int64}}}}

    firstAsset = first(keys(assetsData))
    assetKeys = keys(assetsData)

    for asset in assetKeys
        if timestamp(assetsData[asset]) != timestamp(purgedKFoldStruct.observationTimes[asset])
            error("Data and observation times must have the same index.")
        end
    end

    testRanges = Dict{Symbol, Vector{Tuple{Int64, Int64}}}()
    for asset in assetKeys
        testRanges[asset] = [(i[1], i[end]) for i in kfolds(collect(1:length(purgedKFoldStruct.observationTimes[asset])), k = purgedKFoldStruct.numSplits)]
    end

    finalTrain, finalTest = [], []

    for i in 1:length(testRanges[firstAsset])
        assetsTrainIndices = Dict{Symbol, Vector{Int64}}()
        assetsTestIndices = Dict{Symbol, Vector{Int64}}()

        for (asset, data) in assetsData
            indices = collect(1:length(data))
            embargoSize = round(Int64, length(data) * purgedKFoldStruct.percentEmbargo)

            startIndex, endIndex = testRanges[asset][i]
            firstTestIndex = timestamp(purgedKFoldStruct.observationTimes[asset])[startIndex]
            testIndices = indices[startIndex:endIndex]
            maxTestIndex = searchsortedfirst(timestamp(purgedKFoldStruct.observationTimes[asset]), maximum(values(purgedKFoldStruct.observationTimes[asset])[testIndices, 1]))

            trainIndices = findall(x -> x in timestamp(purgedKFoldStruct.observationTimes[asset]), timestamp(data)[values(data)[:, 1] .<= firstTestIndex])

            if maxTestIndex + embargoSize <= length(values(data))
                append!(trainIndices, indices[maxTestIndex + embargoSize:end])
            end

            assetsTrainIndices[asset] = Int64.(trainIndices)
            assetsTestIndices[asset] = Int64.(testIndices)
        end

        push!(finalTest, assetsTrainIndices)
        push!(finalTrain, assetsTestIndices)
    end

    return collect(zip(finalTrain, finalTest))
end
