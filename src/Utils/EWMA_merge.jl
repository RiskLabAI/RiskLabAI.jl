"""
Calculate the Exponentially Weighted Moving Average (EWMA) of a given input array.

This function calculates the EWMA of a given input array using the specified window size.

Parameters:
- arrInput::Vector{Float64}: Input array for which EWMA needs to be calculated.
- window::Int: Window size for the EWMA calculation.

Returns:
- ewmaArray::Vector{Float64}: Array containing the EWMA values.

Formula:
.. math::
    \\text{EWMA}_t = \\frac{\\sum_{i=0}^{t} (1 - \\alpha)^i \\times \\text{arrInput}[t - i]}{\\sum_{i=0}^{t} (1 - \\alpha)^i}

Where:
- \\text{EWMA}_t: EWMA value at time t.
- \\alpha: Smoothing factor, calculated as \\frac{2}{\\text{window} + 1}.
- \\text{arrInput}[t]: Value of input array at time t.

Reference: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
"""
function calculateEWMA(
    arrInput::Vector{Float64},
    window::Int
)::Vector{Float64}
    arrayLength = length(arrInput)
    ewmaArray = similar(arrInput, Float64)
    
    alpha = 2 / (window + 1)
    weight = 1.0
    ewmaOld = arrInput[1]
    ewmaArray[1] = ewmaOld
    
    for i in 2:arrayLength
        weight += (1 - alpha) ^ i
        ewmaOld = ewmaOld * (1 - alpha) + arrInput[i]
        ewmaArray[i] = ewmaOld / weight
    end
    
    return ewmaArray
end

