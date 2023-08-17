using PyCall
using LinearAlgebra
using DataFrames
using Statistics

@pyimport sklearn.datasets as Datasets

"""
    calculateEigenVectors

Calculate eigen vectors and perform orthogonal features transformation.

Parameters:
- `dotProduct::Matrix{<: Number}`: Input dot product matrix.
- `explainedVarianceThreshold::Float64`: Threshold for variance filtering.

Returns:
- `DataFrame`: DataFrame containing eigen values, vectors, and cumulative variance.
"""
function calculateEigenVectors(
    dotProduct::Matrix{<: Number},
    explainedVarianceThreshold::Float64
)::DataFrame

    eigDecomposition = eigen(dotProduct)
    eigenValues = Float64.(eigDecomposition.values)
    eigenVectors = [eigenVector for eigenVector in eachcol(eigDecomposition.vectors)]

    eigenDataFrame = DataFrame(Index=["PC $i" for i in 1:length(eigenValues)], EigenValue=eigenValues, EigenVector=eigenVectors)

    sort!(eigenDataFrame, [:EigenValue], rev=true)

    cumulativeVariance = cumsum(eigenDataFrame.EigenValue) / sum(eigenDataFrame.EigenValue)

    eigenDataFrame.CumulativeVariance = cumulativeVariance
    index = searchsortedfirst(cumulativeVariance, explainedVarianceThreshold)

    return eigenDataFrame[1:index, :]
end

"""
    standardize

Standardize the input matrix.

Parameters:
- `X`: Features matrix / dataframe.

Returns:
- `Matrix`: Standardized features matrix.
"""
function standardize(X)
    (X .- mean(X, dims=1)) ./ std(X, dims=1)
end

"""
    orthogonalFeatures

Perform orthogonal features transformation.

Parameters:
- `X::Matrix{<: Number}`: Features matrix.
- `varianceThreshold::Float64`: Threshold for variance filtering.

Returns:
- `Tuple{Matrix, DataFrame}`: Transformed features matrix P and eigenDataFrame.
"""
function orthogonalFeatures(
    X::Matrix{<: Number},
    varianceThreshold::Float64=0.95
)::Tuple{Matrix, DataFrame}
    Z = standardize(X)
    dotProduct = Z' * Z
    eigenDataFrame = calculateEigenVectors(dotProduct, varianceThreshold)

    W = reduce(hcat, eigenDataFrame.EigenVector)
    P = Z * W
    return P, eigenDataFrame
end
