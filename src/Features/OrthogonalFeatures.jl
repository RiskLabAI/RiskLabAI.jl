
using PyCall
using LinearAlgebra
using DataFrames
using Statistics

@pyimport sklearn.datasets as Datasets

"""
function: Implementation of Orthogonal Features (Compute Eigen Vectors)
reference: De Prado, M. (2018) Advances In Financial Machine Learning
methodology: page 119 Orthogonal Features section snippet 8.5
"""
function eigenVectors(
    dotProduct::Matrix{<: Number}, # input dot product matrix 
    explainedVarianceThreshold::Float64 # threshold for variance filtering
)::DataFrame

    F = eigen(dotProduct)
    eigenValues = Float64.(F.values)
    eigenVectors = [eigenVector for eigenVector ∈ eachcol(F.vectors)]

    # #2) only positive eVals
    eigenDataFrame = DataFrame(Index=["PC $i" for i ∈ 1:length(eigenValues)], EigenValue=eigenValues, EigenVector=eigenVectors)

    eigenDataFrame = sort(eigenDataFrame, [:EigenValue], rev=true)

    #3) reduce dimension, form PCs
    cumulativeVariance = cumsum(eigenDataFrame.EigenValue) / sum(eigenDataFrame.EigenValue)

    eigenDataFrame.CumulativeVariance = cumulativeVariance
    index = searchsortedfirst(cumulativeVariance, explainedVarianceThreshold)

    eigenDataFrame = eigenDataFrame[1:index, :]

    return eigenDataFrame

end

"""
function: Implementation of Features Matrix Standardization 
reference: n/a
methodology: n/a
"""
function standardize(
    X # features matrix / dataframe
) 
    (X .- mean(X, dims=1)) ./ std(X, dims=1)
end

"""
function: Implementation of Orthogonal Features
reference: De Prado, M. (2018) Advances In Financial Machine Learning
methodology: page 119 Orthogonal Features section snippet 8.5
"""
function orthogonalFeatures(
    X::Matrix{<: Number}; # features matrix
    varianceThreshold::Float64=0.95 # threshold for variance filtering
)::Tuple{Matrix, DataFrame}
    # Given a X, compute orthofeatures P
    Z = standardize(X)
    dotProduct = Z' * Z
    eigenDataFrame = eigenVectors(dotProduct, varianceThreshold)

    W = reduce(hcat, eigenDataFrame.EigenVector)
    P = Z * W
    return P, eigenDataFrame
end

