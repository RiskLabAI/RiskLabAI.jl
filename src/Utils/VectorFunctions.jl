module VectorFunctions

export differences, vectorToMatrix

"""
Calculate the differences between consecutive elements of a vector.

Parameters:
- vector: The input vector.

Returns:
A new vector with the differences between consecutive elements. The first element is marked as missing.
"""
function differences(vector::Vector)::Vector
    result = [missing, (vector |> diff)...]
    result
end

"""
Convert a vector into a column matrix.

Parameters:
- vector: The input vector.

Returns:
A column matrix where each element of the vector corresponds to a row in the matrix.
"""
function vectorToMatrix(vector::Vector)::Matrix
    reshape(vector, length(vector), 1)
end

end
