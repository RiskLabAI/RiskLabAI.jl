module RollingFunctions

export rolling
"""
Calculate rolling computations over two input vectors with a specified window span.

This function applies a given mapping function to rolling windows of specified size over two input vectors.
The mapping function is applied to each rolling window, and the computed values are returned in an array.

Parameters:
- mapping::Function: The mapping function that computes a value from input vectors.
- vector1::Vector: The first input vector.
- vector2::Vector: The second input vector.
- windowSpan::Int: The size of the rolling window.

Returns:
- result::Vector: An array of computed values using rolling windows.
"""
function rolling_computations(
    mapping::Function,
    vector1::Vector,
    vector2::Vector,
    windowSpan::Int
)::Vector
    @assert length(vector1) == length(vector2)
    n = length(vector1)
    result = Vector{Any}(undef, n)  # Preallocate result array

    for i in 1:n
        rightIndex, leftIndex = i, i - (windowSpan - 1)
        if leftIndex ≥ 1
            windowSlice1 = view(vector1, leftIndex:rightIndex)
            windowSlice2 = view(vector2, leftIndex:rightIndex)
            value = mapping(windowSlice1, windowSlice2)
            result[i] = value
        else
            result[i] = missing
        end
    end

    return result
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
function rolling(
    mapping,
    vector,
    windowSpan
)
   
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
