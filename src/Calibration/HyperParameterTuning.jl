include("PurgedKFoldCV.jl")

function MLJBase.train_test_pairs(purgedkfold::PurgedKFold, rows)
    return collect(purgedKFoldSplit(purgedkfold, rows))
end

function MLJBase.train_test_pairs(purgedkfold::PurgedKFold, rows::UnitRange)
    starttime =timestamp(purgedkfold.times)[1]
    data = DataFrame(t = [starttime + Day(i-1)  for i in rows], k = rows)
    data = TimeArray(data,timestamp = :t)
    return collect(purgedKFoldSplit(purgedkfold, data))
end

