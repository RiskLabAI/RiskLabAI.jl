"""
Compute the exponentially weighted moving average (EWMA), EWMA variance, and EWMA standard deviations.

Reference: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
"""
function ewma(arrIn::Vector, window::Int)
    arrLength = length(arrIn)
    ewmaArr = zeros(arrLength)

    alpha = 2 / (window + 1)
    weight = 1

    ewmaOld = arrIn[1]
    ewmaArr[1] = ewmaOld

    for i in 2:arrLength
        weight += (1 - alpha) ^ i
        ewmaOld = ewmaOld * (1 - alpha) + arrIn[i]
        ewmaArr[i] = ewmaOld / weight
    end

    ewmaArr
end
