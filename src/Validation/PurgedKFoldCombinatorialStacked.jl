using TimeSeries, DataFrames, Combinatorics

"""
    PurgedKFoldCombinatorialStacked(
        nSplits::Int,
        nTestSplits::Int,
        times::Dict{String, TimeArray},
        percentEmbargo::Float64
    ) -> PurgedKFoldCombinatorialStacked

A mutable struct for purged k-fold combinatorial cross-validation with support for multiple assets.

This struct defines the configuration for purged k-fold combinatorial cross-validation. 
The training set is purged of observations that overlap with the test-label intervals in 
a combinatorial manner for multiple assets.

:param nSplits: The number of combinatorial splits.
:type nSplits: Int
:param nTestSplits: Number of test splits in the sample.
:type nTestSplits: Int
:param times: Dictionary of entire observation times for multiple assets.
:type times: Dict{String, TimeArray}
:param percentEmbargo: Embargo size as a percentage (divided by 100).
:type percentEmbargo: Float64

:returns: A new PurgedKFoldCombinatorialStacked instance.
:rtype: PurgedKFoldCombinatorialStacked

.. math::
    \\text{Number of Backtest Paths} = C(\\text{nSplits}, \\text{nTestSplits})
"""

mutable struct PurgedKFoldCombinatorialStacked
    nSplits::Int               # The number of combinatorial splits
    nTestSplits::Int           # Number of test splits in the sample
    times::Dict{String, TimeArray}  # Dictionary of entire observation times for multiple assets
    percentEmbargo::Float64    # Embargo size percentage divided by 100
    nBacktestPaths::Int        # Number of combinatorial backtest paths
    backtestPaths::Dict{String, Vector{Dict{String, Vector{Int}}}}  # Combinatorial backtest paths

    function PurgedKFoldCombinatorialStacked(
        nSplits::Int,
        nTestSplits::Int,
        times::Dict{String, TimeArray},
        percentEmbargo::Float64
    )
        if times isa Dict{String, TimeArray}
            new(nSplits, nTestSplits, times, percentEmbargo, backtestPathsNumber(nSplits, nTestSplits), Dict{String, Vector{Dict{String, Vector{Int}}}}())
        else
            error("The times parameter should be a Dictionary of TimeArrays.")
        end
    end
end

# Helper function to calculate the number of combinatorial backtest paths.
# Assuming this function is already defined or will be defined in your code.
function backtestPathsNumber(nSplits::Int, nTestSplits::Int) -> Int
    return binomial(nSplits, nTestSplits)
end
