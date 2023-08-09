include("PurgedKFoldCV.jl")

"""
Generate train-test pairs using PurgedKFold.

Reference: Original code
"""
function generateTrainTestPairs(purged_kfold::PurgedKFold, rows::AbstractVector)
    return collect(purged_kfold_split(purged_kfold, rows))
end

"""
Generate train-test pairs using PurgedKFold for UnitRange.

Reference: Original code
"""
function generateTrainTestPairs(purged_kfold::PurgedKFold, rows::UnitRange)
    startTime = timestamp(purged_kfold.times)[1]
    data = DataFrame(t = [startTime + Day(i - 1) for i in rows], k = rows)
    data = TimeArray(data, timestamp = :t)
    return collect(purged_kfold_split(purged_kfold, data))
end
