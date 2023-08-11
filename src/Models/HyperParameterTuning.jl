include("PurgedKFoldCV.jl")

"""
Generate train-test pairs using PurgedKFold.

Reference: Original code
"""
function generateTrainTestPairs(
        purgedKFold::PurgedKFold, 
        rows::AbstractVector
    )

    return collect(purged_kfold_split(purgedKFold, rows))
end

"""
Generate train-test pairs using PurgedKFold for UnitRange.

Reference: Original code
"""
function generateTrainTestPairs(
        purgedKFold::PurgedKFold,
        rows::UnitRange
    )

    startTime = timestamp(purgedKFold.times)[1]
    data = DataFrame(t = [startTime + Day(i - 1) for i in rows], k = rows)
    data = TimeArray(data, timestamp = :t)
    return collect(purgedKFold_split(purgedKFold, data))
end
