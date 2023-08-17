module VectorFunctions

export calculate_differences, vector_to_matrix

"""
Calculate the differences between consecutive elements of a vector.

This function computes the differences between consecutive elements of the input vector.
The resulting vector has the same length as the input vector, with the first element marked as missing.

Parameters:
- vector::Vector: The input vector.

Returns:
- result::Vector: A new vector with the differences between consecutive elements.

Formula:
.. math::
    \\text{result}[i] = \\text{vector}[i] - \\text{vector}[i-1]

Where:
- \\text{result}[i]: The difference between consecutive elements at index i.
- \\text{vector}[i]: The element at index i of the input vector.
"""
function calculate_differences(vector::Vector)::Vector
    result = [missing; diff(vector)]
    result
end

"""
Convert a vector into a column matrix.

This function converts a given vector into a column matrix.
Each element of the vector corresponds to a row in the resulting matrix.

Parameters:
- vector::Vector: The input vector.

Returns:
- matrix::Matrix: A column matrix where each element of the vector corresponds to a row in the matrix.
"""
function vector_to_matrix(vector::Vector)::Matrix
    reshape(vector, length(vector), 1)
end

end
