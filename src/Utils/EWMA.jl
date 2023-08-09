"""
Compute the exponentially weighted moving average (EWMA), EWMA variance, and EWMA standard deviations.

Reference: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
"""
function ewma(arr_in::Vector, window::Int)
    arr_length = length(arr_in)
    ewma_arr = zeros(arr_length)

    alpha = 2 / (window + 1)
    weight = 1

    ewma_old = arr_in[1]
    ewma_arr[1] = ewma_old

    for i in 2:arr_length
        weight += (1 - alpha) ^ i
        ewma_old = ewma_old * (1 - alpha) + arr_in[i]
        ewma_arr[i] = ewma_old / weight
    end

    ewma_arr
end
