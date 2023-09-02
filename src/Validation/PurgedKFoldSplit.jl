using TimeSeries  # Required package for TimeArray

"""
    purgedKFoldSplit(
        self::PurgedKFold,
        data::TimeArray
    )::Tuple{Vector{Vector{Int64}}, Vector{Vector{Int64}}}

Split data with purging for overlapping observations.

This function splits the data into training and test sets, ensuring that observations that overlap are purged according to the configuration in the given `PurgedKFold` object.

- **Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 109, snippet 7.3
"""
function purgedKFoldSplit(
    self::PurgedKFold,
    data::TimeArray
)::Tuple{Vector{Vector{Int64}}, Vector{Vector{Int64}}}
    
    if timestamp(data) != timestamp(self.times)
        error("Data and times must have the same index.")
    end

    allIndices = collect(1:length(data))
    embargoSize = round(Int64, length(data) * self.percentEmbargo)
    testIndicesList = Vector{Vector{Int64}}()
    trainIndicesList = Vector{Vector{Int64}}()

    testRanges = [(rng[1], rng[end]) for rng in [kfolds(1:length(self.times), k = self.nSplits)[i][2] for i in 1:self.nSplits]]

    for (startIdx, endIdx) in testRanges
        firstTestTime = timestamp(self.times)[startIdx]
        currentTestIndices = allIndices[startIdx:endIdx]
        maxTestTimeIdx = searchsortedfirst(timestamp(self.times), maximum(values(self.times)[currentTestIndices, 1]))

        searchTimes(time) = findfirst(==(time), timestamp(self.times))
        currentTrainIndices = searchTimes.(timestamp(self.times)[values(self.times)[:, 1] .<= firstTestTime])

        if maxTestTimeIdx + embargoSize <= length(values(data))
            append!(currentTrainIndices, allIndices[maxTestTimeIdx + embargoSize:end])
        end

        push!(testIndicesList, Int64.(currentTestIndices))
        push!(trainIndicesList, Int64.(currentTrainIndices))
    end

    return (trainIndicesList, testIndicesList)
end
