module RollingFunctions

export rolling, rolling
    
function rolling(mapping, vector1, vector2, windowSpan)
    @assert length(vector1) == length(vector2)
    n = length(vector1)
    result = [] 
    for i ∈ 1:n
        rightIndex, leftIndex = i, i - (windowSpan - 1)
        value = leftIndex ≥ 1 ? mapping(vector1[leftIndex:rightIndex], vector2[leftIndex:rightIndex]) : missing
        push!(result, value)
    end

    result
end

function rolling(mapping, vector, windowSpan)
    n = length(vector)
    result = [] 
    for i ∈ 1:n
        rightIndex, leftIndex = i, i - (windowSpan - 1)
        value = leftIndex ≥ 1 ? mapping(vector[leftIndex:rightIndex]) : missing
        push!(result, value)
    end

    result
end

end

