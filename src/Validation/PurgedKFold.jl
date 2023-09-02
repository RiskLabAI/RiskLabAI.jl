using TimeSeries  # Required package for TimeArray

"""
    PurgedKFold

Custom structure for cross-validation with purging of overlapping observations.

.. math::

    Given \\( n \\) as the number of splits, \\( T \\) as the TimeArray of times, and \\( p \\) as the percentage of embargo:

    PurgedKFold(n, T, p)

**Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 109, snippet 7.3
"""
mutable struct PurgedKFold
    nSplits::Int64
    times::TimeArray
    percentEmbargo::Float64

    """
        PurgedKFold(
            nSplits::Int = 3,
            times::TimeArray = nothing,
            percentEmbargo::Float64 = 0.0
        )::PurgedKFold

    Create a new PurgedKFold instance.

    :param nSplits: Number of splits.
    :type nSplits: Int
    :param times: TimeArray of data times.
    :type times: TimeArray
    :param percentEmbargo: Percentage of embargo.
    :type percentEmbargo: Float64

    :returns: A new PurgedKFold instance.
    :rtype: PurgedKFold
    """
    function PurgedKFold(
        nSplits::Int = 3,
        times::TimeArray = nothing,
        percentEmbargo::Float64 = 0.0
    )::PurgedKFold
        # Ensure the times parameter is a TimeArray
        times isa TimeArray || error("The times parameter should be a TimeArray.")
        
        new(nSplits, times, percentEmbargo)
    end
end
