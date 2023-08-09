include("PurgedKFoldCV.jl")

"""
Generate train-test pairs using PurgedKFold.

Reference: Original code
"""
function generate_train_test_pairs(purged_kfold::PurgedKFold, rows::AbstractVector)
    return collect(purged_kfold_split(purged_kfold, rows))
end

"""
Generate train-test pairs using PurgedKFold for UnitRange.

Reference: Original code
"""
function generate_train_test_pairs(purged_kfold::PurgedKFold, rows::UnitRange)
    start_time = timestamp(purged_kfold.times)[1]
    data = DataFrame(t = [start_time + Day(i - 1) for i in rows], k = rows)
    data = TimeArray(data, timestamp = :t)
    return collect(purged_kfold_split(purged_kfold, data))
end
