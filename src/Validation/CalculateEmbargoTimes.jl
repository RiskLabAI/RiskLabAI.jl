using TimeSeries  # Required package for TimeArray

"""
    calculateEmbargoTimes(
        times::Array{DateTime},
        percentEmbargo::Float64
    )::TimeArray

Calculate the embargo time for each bar based on a given percentage.

This function calculates the embargo time for each bar by adding a certain percentage of the total bars as embargo time to each bar.

.. math::

    \\[ \\text{step} = \\text{round}(\\text{length}(\\text{times}) \\times \\text{percentEmbargo}) \\]

:param times: Array of bar times.
:type times: Array{DateTime}
:param percentEmbargo: Percentage of embargo.
:type percentEmbargo: Float64

:returns: Array of embargo times.
:rtype: TimeArray

**Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 108, snippet 7.2
"""
function calculateEmbargoTimes(
    times::Array{DateTime},
    percentEmbargo::Float64
)::TimeArray
    # Calculate the step size based on the given percentage
    step = round(Int, length(times) * percentEmbargo)

    # Initialize the TimeArray for the embargo times
    if step == 0
        embargoTimes = TimeArray((Times = times, Timestamp = times), timestamp = :Timestamp)
    else
        mainEmbargo = TimeArray((Times = times[step + 1:end], Timestamp = times[1:end - step]), timestamp = :Timestamp)
        
        # Create a TimeArray for the tail times, repeated with the last time value
        tailTimes = TimeArray((Times = repeat([times[end]], step), Timestamp = times[end - step + 1:end]), timestamp = :Timestamp)
        
        # Concatenate the main and tail TimeArrays
        embargoTimes = vcat(mainEmbargo, tailTimes)
    end

    return embargoTimes
end
