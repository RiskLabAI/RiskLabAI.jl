using PurgedKFoldCV
using DataFrames
using TimeSeries

"""
    generateTrainTestPairs(purgedKFold::PurgedKFold, rows::AbstractVector)

Generate train-test pairs using the PurgedKFold cross-validator.

# Arguments
- `purgedKFold::PurgedKFold`: the PurgedKFold cross-validator.
- `rows::AbstractVector`: the rows of data to be used.

# Returns
- `Array{Tuple{AbstractVector, AbstractVector}}`: an array of tuples where each tuple contains indices of training and testing data.

# Mathematical Formula
This function uses the PurgedKFold cross-validator. It is used in financial applications where it is necessary to ensure that there is no information leakage from the training data to the testing data, due to the temporally-ordered nature of financial time series.

In this method, the input data is split into K equally-sized partitions. For each partition, the training data consists of all partitions that occur before the test partition and all partitions that occur after it. The test data consists of only the test partition. Additionally, the data immediately before and after the test partition is purged to prevent any information leakage.

.. math::
    \\text{{Training data}} = \\text{{data}}[1 : (k-1) * n/K - p] \\cup \\text{{data}}[(k+1) * n/K + p : n]
    \\text{{Testing data}} = \\text{{data}}[k * n/K - p : (k+1) * n/K + p]
    \\text{{where }} n \\text{{ is the total number of samples, }} K \\text{{ is the number of partitions, and }} p \\text{{ is the purge length.}}
"""
function generateTrainTestPairs(
        purgedKFold::PurgedKFold,
        rows::AbstractVector
    )
    return collect(purged_kfold_split(purgedKFold, rows))
end

"""
    generateTrainTestPairs(purgedKFold::PurgedKFold, rows::UnitRange)

Generate train-test pairs using the PurgedKFold cross-validator for UnitRange.

# Arguments
- `purgedKFold::PurgedKFold`: the PurgedKFold cross-validator.
- `rows::UnitRange`: the range of rows to be used.

# Returns
- `Array{Tuple{AbstractVector, AbstractVector}}`: an array of tuples where each tuple contains indices of training and testing data.

# Mathematical Formula
This function uses the PurgedKFold cross-validator. It is used in financial applications where it is necessary to ensure that there is no information leakage from the training data to the testing data, due to the temporally-ordered nature of financial time series.

In this method, the input data is split into K equally-sized partitions. For each partition, the training data consists of all partitions that occur before the test partition and all partitions that occur after it. The test data consists of only the test partition. Additionally, the data immediately before and after the test partition is purged to prevent any information leakage.

.. math::
    \\text{{Training data}} = \\text{{data}}[1 : (k-1) * n/K - p] \\cup \\text{{data}}[(k+1) * n/K + p : n]
    \\text{{Testing data}} = \\text{{data}}[k * n/K - p : (k+1) * n/K + p]
    \\text{{where }} n \\text{{ is the total number of samples, }} K \\text{{ is the number of partitions, and }} p \\text{{ is the purge length.}}
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
