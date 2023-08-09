module RollingFunctions

export rolling

"""
Calculate rolling computations over two input vectors with a specified window span.

Parameters:
- mapping: A function that computes a value from the input vectors.
- vector1: The first input vector.
- vector2: The second input vector.
- windowSpan: The size of the rolling window.

Returns:
An array of computed values with rolling window.
"""
function rolling(mapping, vector1, vector2, windowSpan)
    @assert length(vector1) == length(vector2)
    n = length(vector1)
    result = []
    for i in 1:n
        rightIndex, leftIndex = i, i - (windowSpan - 1)
        value = leftIndex ≥ 1 ? mapping(vector1[leftIndex:rightIndex], vector2[leftIndex:rightIndex]) : missing
        push!(result, value)
    end

    result
end

"""
Calculate rolling computations over a single input vector with a specified window span.

Parameters:
- mapping: A function that computes a value from the input vector.
- vector: The input vector.
- windowSpan: The size of the rolling window.

Returns:
An array of computed values with rolling window.
"""
function rolling(mapping, vector, windowSpan)
    n = length(vector)
    result = []
    for i in 1:n
        rightIndex, leftIndex = i, i - (windowSpan - 1)
        value = leftIndex ≥ 1 ? mapping(vector[leftIndex:rightIndex]) : missing
        push!(result, value)
    end

    result
end

end
