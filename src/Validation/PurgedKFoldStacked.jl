"""
    PurgedKFoldStacked(
        numSplits::Int64,
        observationTimes::Dict{Symbol, TimeArray},
        percentEmbargo::Float64
    )

A mutable struct for implementing a purged k-fold cross-validation strategy that supports multiple assets with labels that span intervals.

- **Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 110, snippet 7.4

:param numSplits: The number of k-fold splits.
:type numSplits: Int64
:param observationTimes: Dictionary containing the entire observation times for multiple assets.
:type observationTimes: Dict{Symbol, TimeArray}
:param percentEmbargo: The percentage size for the embargo, divided by 100.
:type percentEmbargo: Float64

**Note:**

- `observationTimes` should contain the TimeArray for each asset where:
    - `timestamp(TimeArray)`: Time when the observation started.
    - `values(TimeArray)[:, 1]`: Time when the observation ended.
"""
mutable struct PurgedKFoldStacked
    numSplits::Int64
    observationTimes::Dict{Symbol, TimeArray}
    percentEmbargo::Float64

    """
        PurgedKFoldStacked(
            numSplits::Int = 3,
            observationTimes::Dict{Symbol, TimeArray} = nothing,
            percentEmbargo::Float64 = 0.0
        )
    """
    function PurgedKFoldStacked(
        numSplits::Int = 3,
        observationTimes::Dict{Symbol, TimeArray} = nothing,
        percentEmbargo::Float64 = 0.0
    )
        if observationTimes isa Dict{Symbol, TimeArray}
            new(numSplits, observationTimes, percentEmbargo)
        else
            error("The observationTimes parameter should be a Dictionary of TimeArrays.")
        end
    end
end
