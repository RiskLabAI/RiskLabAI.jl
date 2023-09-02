using TimeSeries

"""
    PurgedKFoldCombinatorial

A mutable struct for performing combinatorial k-fold cross-validation 
while purging observations that overlap test-label intervals. This 
struct facilitates backtesting paths in a combinatorial manner.

:param nSplits: Number of combinatorial splits.
:type nSplits: Int
:param nTestSplits: Number of test splits in the sample.
:type nTestSplits: Int
:param times: Entire observation times.
:type times: TimeArray
:param percentEmbargo: Embargo size percentage divided by 100.
:type percentEmbargo: Float64
:param nBacktestPaths: Number of combinatorial backtest paths.
:type nBacktestPaths: Int
:param backtestPaths: Array storing combinatorial backtest paths.
:type backtestPaths: Array

- `timestamp(TimeArray)`: Time when the observation started.
- `values(TimeArray)[:, 1]`: Time when the observation ended.

.. math::
    \\text{Number of backtest paths} = \\frac{\\binom{nSplits}{nSplits - nTestSplits} \\times nTestSplits}{nSplits}
"""
mutable struct PurgedKFoldCombinatorial
    nSplits::Int
    nTestSplits::Int
    times::TimeArray
    percentEmbargo::Float64
    nBacktestPaths::Int
    backtestPaths::Array

    function PurgedKFoldCombinatorial(
        nSplits::Int = 3, 
        nTestSplits::Int = 2, 
        times::TimeArray = nothing, 
        percentEmbargo::Float64 = 0.0
    )
        if isa(times, TimeArray)
            new(
                nSplits, 
                nTestSplits, 
                times, 
                percentEmbargo, 
                backtestPathsNumber(nSplits, nTestSplits), 
                []
            )
        else
            error("The times parameter must be a TimeArray.")
        end
    end
end
