using DataFrames, LinearAlgebra, Statistics, Distributions, KernelDensity, BlockArrays, Optim, MultivariateStats

"""
    marcenkoPasturPdf(var::Float64, ratio::Float64, points::Int)

Implements the Marcenko–Pastur PDF in Julia.

- `var`: Scalar variable for which the PDF is calculated.
- `ratio`: Ratio of the sample size to the number of variables.
- `points`: Number of points at which the PDF is evaluated.

Returns a DataFrame with the eigenvalues and their corresponding PDF values.

.. math::
    λ_{min} = var * (1 - √(1 / ratio))^2
    λ_{max} = var * (1 + √(1 / ratio))^2
    diffλ = ((λ_{max} .- eigenValues) .* (eigenValues .- λ_{min}))
    pdf = ratio ./ (2 * π * var * eigenValues) .* diffλ
"""
function marcenkoPasturPdf(
    var::Float64,
    ratio::Float64,
    points::Int
)::DataFrame
    λmin = var * (1 - sqrt(1 / ratio))^2
    λmax = var * (1 + sqrt(1 / ratio))^2
    eigenValues = range(λmin, stop=λmax, length=points)
    diffλ = ((λmax .- eigenValues) .* (eigenValues .- λmin))
    diffλ[diffλ .< -1E-3] .= 0.
    pdf = ratio ./ (2 * π * var * eigenValues) .* diffλ
    return DataFrame(index=eigenValues, values=pdf)
end
using LinearAlgebra
using Distributions
using KernelDensity
using DataFrames

"""
    principalComponentAnalysis(matrix::AbstractMatrix{<:Real})

Compute the principal components of a Hermitian matrix.

**Formulae**:
Given a matrix :math:`A`, the principal components are computed as:

.. math::

    A = U \Sigma V^*

Where :math:`U` is the left singular vectors (eigenvectors), :math:`\Sigma` is a diagonal matrix of eigenvalues.

# Arguments
- `matrix::AbstractMatrix{<:Real}`: The input Hermitian matrix.

# Returns
- `::Diagonal{<:Real}`: Diagonal matrix of sorted eigenvalues.
- `::AbstractMatrix{<:Real}`: Matrix of sorted eigenvectors.
"""
function principalComponentAnalysis(
    matrix::AbstractMatrix{<:Real}
)::Tuple{Diagonal{<:Real}, AbstractMatrix{<:Real}}
    eigenValues, eigenVectors = eigen(matrix)
    indices = sortperm(eigenValues, rev=true)
    eigenValues = eigenValues[indices]
    eigenVectors = eigenVectors[:, indices]
    return Diagonal(eigenValues), eigenVectors
end

"""
    fitKernelDensity(
        observations::AbstractVector{<:Real};
        bandwidth::Float64=0.25,
        kernel::Distributions=Normal,
        valuesForEvaluating::Union{AbstractVector{<:Real}, Nothing}=nothing
    )

Fit a kernel density estimator to the provided observations.

# Arguments
- `observations::AbstractVector{<:Real}`: Observations to estimate density from.
- `bandwidth::Float64=0.25`: Bandwidth for the kernel density estimation.
- `kernel::Distributions=Normal`: Kernel distribution to use.
- `valuesForEvaluating::Union{AbstractVector{<:Real}, Nothing}=nothing`: Values where the density will be evaluated.

# Returns
- `::DataFrame`: A DataFrame with columns 'index' (evaluation points) and 'values' (density values).
"""
function fitKernelDensity(
    observations::AbstractVector{<:Real};
    bandwidth::Float64=0.25,
    kernel::Distributions=Normal(),
    valuesForEvaluating::Union{AbstractVector{<:Real}, Nothing}=nothing
)::DataFrame
    densityEstimate = kde(observations, kernel=kernel, bandwidth=bandwidth)
    if isnothing(valuesForEvaluating)
        valuesForEvaluating = reverse(unique(observations))
    end
    densityValues = KernelDensity.pdf(densityEstimate, valuesForEvaluating)
    return DataFrame(index=valuesForEvaluating, values=densityValues)
end

"""
    generateCovarianceWithSignal(
        numberColumns::Int,
        numberFactors::Int
    )

Generate a random covariance matrix and add a signal to it.

# Arguments
- `numberColumns::Int`: Number of columns for the generated matrix.
- `numberFactors::Int`: Number of factors for the generated matrix.

# Returns
- `::AbstractMatrix{<:Real}`: A covariance matrix with added signal.
"""
function generateCovarianceWithSignal(
    numberColumns::Int,
    numberFactors::Int
)::AbstractMatrix{<:Real}
    dataMatrix = rand(Normal(), numberColumns, numberFactors)
    covarianceMatrix = dataMatrix * dataMatrix'
    covarianceMatrix += Diagonal(rand(Uniform(), numberColumns))
    return covarianceMatrix
end
using LinearAlgebra
using Distributions
using KernelDensity

"""
    covarianceToCorrelation(cov::Matrix{Float64})::Matrix{Float64}

Derive the correlation matrix from a covariance matrix.

# Arguments
- `cov::Matrix{Float64}`: A covariance matrix.

# Returns
- `::Matrix{Float64}`: A correlation matrix.

# Mathematical Formula
The correlation matrix is derived from a covariance matrix using the formula:

.. math::

    \text{corr}_{i,j} = \frac{\text{cov}_{i,j}}{\sqrt{\text{var}_i \cdot \text{var}_j}}

where:
- :math:`\text{corr}_{i,j}` is the correlation between asset i and asset j.
- :math:`\text{cov}_{i,j}` is the covariance between asset i and asset j.
- :math:`\text{var}_i` is the variance of asset i.
- :math:`\text{var}_j` is the variance of asset j.
"""
function covarianceToCorrelation(cov::Matrix{Float64})::Matrix{Float64}
    stdDev = sqrt.(diag(cov))
    corrMatrix = cov ./ (stdDev * stdDev')
    corrMatrix[corrMatrix .< -1] .= -1
    corrMatrix[corrMatrix .> 1] .= 1
    return corrMatrix
end

"""
    fitMarcenkoPasturToCovarianceMatrix(var::Float64, eigenValues::Vector{Float64},
                                         ratio::Float64, bandwidth::Float64, points::Int = 1000)::Float64

Fit the Marcenko–Pastur PDF to a random covariance matrix that contains signal.
The objective of the fit is to find the value of σ² that minimizes the sum of the squared differences between the analytical PDF and the KDE of the eigenvalues.

# Arguments
- `var::Float64`: The variance of the residuals.
- `eigenValues::Vector{Float64}`: The eigenvalues of the covariance matrix.
- `ratio::Float64`: The aspect ratio (T/N) of the covariance matrix.
- `bandwidth::Float64`: The bandwidth of the KDE.
- `points::Int`: The number of points to evaluate the PDF. Default is 1000.

# Returns
- `::Float64`: The sum of the squared differences between the analytical PDF and the KDE of the eigenvalues.
"""
function fitMarcenkoPasturToCovarianceMatrix(
        var::Float64,
        eigenValues::Vector{Float64},
        ratio::Float64,
        bandwidth::Float64,
        points::Int = 1000)::Float64
    
    pdf0 = pdfMarcenkoPastur(var, ratio, points)
    pdf1 = kde(eigenValues, bandwidth = bandwidth, kernel = Distributions.Normal, points = pdf0.index)
    sse = sum((pdf1.values .- pdf0.values).^2)
    return sse
end

using LinearAlgebra
using Optim

"""
    findMaxEigenvalues(
        eigenValues::Vector{Float64},
        ratio::Float64,
        bandWidth::Float64
    )::Tuple{Float64, Float64}

Find the maximum random eigenvalues by fitting Marcenko's distribution.

# Arguments
- `eigenValues::Vector{Float64}`: The eigenvalues of the covariance matrix.
- `ratio::Float64`: The aspect ratio (T/N) of the covariance matrix.
- `bandWidth::Float64`: The bandwidth of the KDE.

# Returns
- `::Tuple{Float64, Float64}`: A tuple containing the maximum eigenvalue and the estimated variance.
"""
function findMaxEigenvalues(
        eigenValues::Vector{Float64},
        ratio::Float64,
        bandWidth::Float64
    )::Tuple{Float64, Float64}

    out = optimize(var -> errorPDFs(var, eigenValues, ratio, bandWidth), 1E-5, 1-1E-5)
    if Optim.converged(out)
        variance = Optim.minimizer(out)
    else
        variance = 1.0
    end
    λmax = variance * (1 + sqrt(1 / ratio))^2
    return λmax, variance
end

"""
    constantResidualEigenvalueMethod(
        eigenValues::Matrix{Float64},
        eigenVectors::Matrix{Float64},
        numberFactors::Int
    )::Matrix{Float64}

Constant Residual Eigenvalue Method.

# Arguments
- `eigenValues::Matrix{Float64}`: The eigenvalues of the covariance matrix.
- `eigenVectors::Matrix{Float64}`: The eigenvectors of the covariance matrix.
- `numberFactors::Int`: The number of factors to retain.

# Returns
- `::Matrix{Float64}`: The denoised correlation matrix.
"""
function constantResidualEigenvalueMethod(
        eigenValues::Matrix{Float64},
        eigenVectors::Matrix{Float64},
        numberFactors::Int
    )::Matrix{Float64}

    λ = copy(diagm(eigenValues))
    λ[numberFactors:end] .= sum(λ[numberFactors:end]) / (length(λ) - numberFactors)
    λDiag = Diagonal(λ)
    covariance = eigenVectors * λDiag * transpose(eigenVectors)
    correlation = covarianceToCorrelation(covariance)
    return correlation
end

"""
    denoisedCorrelationShrinkage(
        eigenValues::Matrix{Float64},
        eigenVectors::Matrix{Float64},
        numberFactors::Int,
        α::Float64 = 0
    )::Matrix{Float64}

Denoising by Targeted Shrinkage.

# Arguments
- `eigenValues::Matrix{Float64}`: The eigenvalues of the covariance matrix.
- `eigenVectors::Matrix{Float64}`: The eigenvectors of the covariance matrix.
- `numberFactors::Int`: The number of factors to retain.
- `α::Float64`: The shrinkage intensity parameter, defaults to 0.

# Returns
- `::Matrix{Float64}`: The denoised correlation matrix.
"""
function denoisedCorrelationShrinkage(
        eigenValues::Matrix{Float64},
        eigenVectors::Matrix{Float64},
        numberFactors::Int,
        α::Float64 = 0.0
    )::Matrix{Float64}

    eigenValuesL = eigenValues[1:numberFactors, 1:numberFactors]
    eigenVectorsL = eigenVectors[:, 1:numberFactors]
    eigenValuesR = eigenValues[numberFactors:end, numberFactors:end]
    eigenVectorsR = eigenVectors[:, numberFactors:end]
    corr0 = eigenVectorsL * eigenValuesL * transpose(eigenVectorsL)
    corr1 = eigenVectorsR * eigenValuesR * transpose(eigenVectorsR)
    corr2 = corr0 + α * corr1 + (1 - α) * diagm(diag(corr1))
    return corr2
end

"""
    formBlockMatrix(numberBlocks::Int, sizeBlock::Int, corrBlock::Float64)::Matrix{Float64}

Generates a block-diagonal covariance matrix.

# Arguments
- `numberBlocks::Int`: Number of blocks.
- `sizeBlock::Int`: Size of each block.
- `corrBlock::Float64`: Correlation within each block.

# Returns
- `Matrix{Float64}`: Block-diagonal matrix.

# Reference
De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons. Snippet 2.7, Page 33
"""
function formBlockMatrix(numberBlocks::Int, sizeBlock::Int, corrBlock::Float64)::Matrix{Float64}
    block = ones(sizeBlock, sizeBlock) * corrBlock
    block[diagind(block)] .= 1.0
    corr = BlockArray{Float64}(undef_blocks, repeat([sizeBlock], numberBlocks), repeat([sizeBlock], numberBlocks))
    for i in 1:numberBlocks
        for j in 1:numberBlocks
            if i == j
                setblock!(corr, block, i, j)
            else
                setblock!(corr, zeros(sizeBlock, sizeBlock), i, j)
            end
        end
    end
    return Matrix(corr)
end

"""
    generateTrueMatrix(numberBlocks::Int, sizeBlock::Int, corrBlock::Float64)::Tuple{Vector{Float64}, Matrix{Float64}}

Generates the true matrix with shuffled columns, mean vector, and covariance matrix.

# Arguments
- `numberBlocks::Int`: Number of blocks.
- `sizeBlock::Int`: Size of each block.
- `corrBlock::Float64`: Correlation within each block.

# Returns
- `Tuple{Vector{Float64}, Matrix{Float64}}`: Mean vector and covariance matrix.

# Reference
De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons. Snippet 2.7, Page 33
"""
function generateTrueMatrix(numberBlocks::Int, sizeBlock::Int, corrBlock::Float64)::Tuple{Vector{Float64}, Matrix{Float64}}
    corr = formBlockMatrix(numberBlocks, sizeBlock, corrBlock)
    columns = shuffle(collect(1:numberBlocks * sizeBlock))
    corr = corr[columns, columns]
    std0 = rand(Uniform(0.05, 0.2), size(corr)[1])
    cov0 = correlationToCovariance(corr, std0)
    mu0 = rand.(Normal.(std0, std0), 1)
    return mu0, cov0
end

"""
    correlationToCovariance(corr::Matrix{Float64}, std::Vector{Float64})::Matrix{Float64}

Converts a correlation matrix to a covariance matrix.

# Arguments
- `corr::Matrix{Float64}`: Correlation matrix.
- `std::Vector{Float64}`: Standard deviations.

# Returns
- `Matrix{Float64}`: Covariance matrix.

# Reference
De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons. Snippet 2.7, Page 33
"""
function correlationToCovariance(corr::Matrix{Float64}, std::Vector{Float64})::Matrix{Float64}
    cov = corr .* (std .* std')
    return cov
end
