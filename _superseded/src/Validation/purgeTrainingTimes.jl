using TimeSeries  # Include required packages for TimeArray

"""
    purgeTrainingTimes(
        data::TimeArray,
        test::TimeArray
    )::TimeArray

Remove test observations from the training set.

This function removes observations from the training data that fall within the time periods covered by the test data.

.. math::

    Let \\( T \\) be the set of training times, and \\( t_{start}, t_{end} \\) be the start and end times in the test set. Then,

    \\[
    T' = T \\setminus \\{ t | t_{start} \leq t \leq t_{end} \\}
    \\]

where \\( T' \\) is the updated set of training times.

:param data: Time series data for training.
:type data: TimeArray
:param test: Time series data for testing.
:type test: TimeArray

:returns: Training data with test observations removed.
:rtype: TimeArray

**Reference:** De Prado, M. (2018) *Advances in Financial Machine Learning*. Methodology: page 106, snippet 7.1
"""
function purgeTrainingTimes(
    data::TimeArray,
    test::TimeArray
)::TimeArray
    # Deep copy of the data to avoid altering the original dataset
    trainingTimes = deepcopy(data)
    
    # Pre-compute timestamp values for trainingTimes to avoid repeated calculations
    trainingTimestamps = timestamp(trainingTimes)
    trainingValues = values(trainingTimes)
    
    # Loop through each (startTime, endTime) pair in the test set
    for (startTime, endTime) in zip(timestamp(test), values(test)[:, 1])
        startWithinTestTimes = filter(t -> t >= startTime && t <= endTime, trainingTimestamps)
        endWithinTestTimes = filter(t -> trainingValues[findfirst(==(t), trainingTimestamps), 1] >= startTime && trainingValues[findfirst(==(t), trainingTimestamps), 1] <= endTime, trainingTimestamps)
        envelopeTestTimes = filter(t -> t <= startTime && trainingValues[findfirst(==(t), trainingTimestamps), 1] >= endTime, trainingTimestamps)
        
        # Use setdiff to find the times that are not within any of the test intervals
        filteredTimes = setdiff(trainingTimestamps, union(startWithinTestTimes, endWithinTestTimes, envelopeTestTimes))
        
        # Update the training set
        trainingTimes = trainingTimes[filteredTimes]
        trainingTimestamps = timestamp(trainingTimes)  # Update for the next iteration
    end
    
    return trainingTimes
end
