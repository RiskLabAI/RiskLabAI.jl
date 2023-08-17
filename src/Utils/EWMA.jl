"""
Compute the exponentially weighted moving average (EWMA), EWMA variance, and EWMA standard deviations.

Reference: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
"""
function calculate_ewma(
        arr_in::Vector{Float64},
        window::Int
    )::Vector{Float64}
    arr_length = length(arr_in)
    ewma_arr = zeros(Float64, arr_length)

    alpha = 2 / (window + 1)
    weight = 1.0

    ewma_old = arr_in[1]
    ewma_arr[1] = ewma_old

    for i in 2:arr_length
        weight += (1 - alpha) ^ i
        ewma_old = ewma_old * (1 - alpha) + arr_in[i]
        ewma_arr[i] = ewma_old / weight
    end

    return ewma_arr
end
