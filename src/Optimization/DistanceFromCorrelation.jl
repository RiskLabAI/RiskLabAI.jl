using LinearAlgebra
using DataFrames

"""
    distanceFromCorrelation(correlation::AbstractMatrix)::AbstractMatrix

Calculate the distance matrix from a correlation matrix.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 16.4, Page 241

# Arguments
- `correlation::AbstractMatrix`: A correlation matrix.

# Returns
- `AbstractMatrix`: A distance matrix derived from the correlation matrix.

# Formula
The distance matrix is calculated as follows:

.. math::
    \text{distance} = \sqrt{1 - \text{correlation}}
"""
function distanceFromCorrelation(correlation::AbstractMatrix)::AbstractMatrix
    return sqrt.(1 .- correlation)
end