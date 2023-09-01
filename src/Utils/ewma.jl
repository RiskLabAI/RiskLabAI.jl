"""
function: EWMA: y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) / (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).
reference: n/a
methodology: n/a
"""
function ewma(
    array::Vector,
    window::Int 
)::Vector{Float64}
    
    arrayLength = length(array)
    result = zeros(Float64, arrayLength)

    α::Float64 = 2 / (window + 1)
    weight::Float64 = 1.0
    thisTerm::Float64 = array[1]
    result[1] = thisTerm
    for i ∈ 2: arrayLength
        weight += (1 - α) ^ (i - 1)
        thisTerm = thisTerm * (1 - α) + array[i]
        result[i] = thisTerm / weight
    end
    
    return result
end