"""
Function: Implements the Marcenko–Pastur PDF in Julia.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.1, Page 25
"""
function pdfMarcenkoPastur(var, ratio, points)
    λmin = var * (1 - sqrt(1 / ratio))^2
    λmax = var * (1 + sqrt(1 / ratio))^2
    eigenValues = range(λmin, stop=λmax, length=points)
    diffλ = ((λmax .- eigenValues) .* (eigenValues .- λmin))
    diffλ[diffλ .< -1E-3] .= 0.
    pdf = ratio ./ (2 * pi * var * eigenValues) .* diffλ
    return DataFrame(index=eigenValues, values=pdf)
end

"""
Function: Get eVal, eVec from a Hermitian matrix.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.2, Page 25
"""
function pca(matrix) # Hermitian matrix
    eigenValues, eigenVectors = eigen(matrix)
    indices = sortperm(eigenValues, rev=true)
    eigenValues, eigenVectors = eigenValues[indices], eigenVectors[:, indices]
    eigenValues = Diagonal(eigenValues)
    return eigenValues, eigenVectors
end

"""
Function: Fit kernel density.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.2, Page 25
"""
function kde(observations; bandwidth=0.25, kernel=Distributions.Normal, valuesForEvaluating=nothing)
    density = kde(observations, kernel=kernel, bandwidth=bandwidth)
    if valuesForEvaluating == nothing
        valuesForEvaluating = reshape(reverse(unique(observations)), :, 1)
    end
    density = KernelDensity.pdf(density, valuesForEvaluating[:])
    return DataFrame(index=vec(valuesForEvaluating), values=density)
end

"""
Function: Add signal to a random covariance matrix.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.3, Page 27
"""
function addSignalToCovarianceMatrix(numberColumns, numberFactors)
    data = rand(Normal(), numberColumns, numberFactors)
    covData = data * data'
    covData += Diagonal(rand(Uniform(), numberColumns))
    return covData
end

"""
Function: Derive the correlation matrix from a covariance matrix.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.3, Page 27
"""
function covarianceToCorrelation(cov)
    std = sqrt.(diag(cov))
    corr = cov ./ (std .* std')
    corr[corr .< -1] .= -1
    corr[corr .> 1] .= 1
    return corr
end

"""
Function: Fits the Marcenko–Pastur PDF to a random covariance
matrix that contains signal. The objective of the fit is to find the value of σ² that
minimizes the sum of the squared differences between the analytical PDF and.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.4, Page 27
"""
function fitMarcenkoPasturToCovarianceMatrix(var, eigenValues, ratio, bandWidth, points = 1000)
    pdf0 = pdfMarcenkoPastur(var, ratio, points)
    pdf1 = kde(eigenValues, bandwidth = bandWidth, kernel = Distributions.Normal, valuesForEvaluating = pdf0.index)
    sse = sum((pdf1.values .- pdf0.values).^2)
    return sse
end

"""
Function: Find the maximum random eigenvalues by fitting Marcenko's distribution.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.4, Page 27
"""
function findMaxEigenvalues(eigenValues, ratio, bandWidth)
    out = optimize(var -> errorPDFs(var, eigenValues, ratio, bandWidth), 1E-5, 1-1E-5)
    if Optim.converged(out) == true
        var = Optim.minimizer(out)
    else
        var = 1
    end
    λmax = var * (1 + (1 / ratio)^0.5)^2
    return λmax, var
end

"""
Function: Constant Residual Eigenvalue Method
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.5, Page 29
"""
function constantResidualEigenvalueMethod(eigenValues, eigenVectors, numberFactors)
    λ = copy(diag(eigenValues))
    λ[numberFactors:end] .= sum(λ[numberFactors:end]) / (size(λ)[1] - numberFactors)
    λDiag = Diagonal(λ)
    cov = eigenVectors * λDiag * eigenVectors'
    corr2 = covarianceToCorrelation(cov)
    return corr2
end

"""
Function: Denoising by Targeted Shrinkage
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.6, Page 31
"""
function denoisedCorrelationShrinkage(eigenValues, eigenVectors, numberFactors, α = 0)
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
Function: Generate a block-diagonal covariance matrix and a vector of means.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.7, Page 33
"""
function formBlockMatrix(numberBlocks, sizeBlock, corrBlock)
    block = ones(sizeBlock, sizeBlock) * corrBlock
    block[Statistics.diagind(block)] .= 1.0
    corr = BlockArray{Float64}(undefBlocks, repeat([sizeBlock], numberBlocks), repeat([sizeBlock], numberBlocks))
    for i in range(1, stop = numberBlocks)
        for j in range(1, stop = numberBlocks)
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
Function: Generate the true matrix with shuffled columns, mean vector, and covariance matrix.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.7, Page 33
"""
function generateTrueMatrix(numberBlocks, sizeBlock, corrBlock)
    corr = formBlockMatrix(numberBlocks, sizeBlock, corrBlock)
    columns =  shuffle(collect(1:numberBlocks * sizeBlock))
    corr = corr[columns, columns]
    std0 = rand(Uniform(0.05, 0.2), size(corr)[1])
    cov0 = correlationToCovariance(corr, std0)
    mu0 = rand.(Normal.(std0, std0), 1)
    return mu0, cov0
end

"""
Function: Derive the covariance matrix from a correlation matrix
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.7, Page 33
"""
function correlationToCovariance(corr, std)
    cov = corr .* (std .* std')
    return cov
end

"""
Function: Generate the empirical covariance matrix using simulations.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.8, Page 33
"""
function generateEmpiricalCovariance(mu0, cov0, observations, shrink = false)
    data = transpose(rand(MvNormal(vcat(mu0...), cov0), observations))
    mu1 = vec(reshape(mean(data, dims = 1), (500, 1)))
    if shrink
        cov1 = LedoitWolf().fit(data).covariance_
    else
        cov1 = cov(data)
    end
    return mu1, cov1
end

"""
Function: Denoising of the empirical covariance matrix.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.9, Page 34
"""
function denoiseCovariance(cov0, ratio, bandWidth)
    corr0 = covarianceToCorrelation(cov0)
    eigenValues0, eigenVectors0 = principalComponentAnalysis(corr0)
    λmax0, var0 = findMaxEigenvalues(diag(eigenValues0), ratio, bandWidth)
    numberFactors0 = size(eigenValues0)[1] - searchsortedfirst(reverse(diag(eigenValues0)), λmax0) + 1
    corr1 = constantResidualEigenvalueMethod(eigenValues0, eigenVectors0, numberFactors0)
    cov1 = correlationToCovariance(corr1, diag(cov0).^0.5)
    return cov1
end

"""
Function: Perform Monte Carlo simulation to optimize portfolio weights.
Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 2.10, Page 34
"""
function monteCarloOptimizePortfolio(cov, mu = nothing)
    inverse = inv(cov)
    ones1 = ones(size(inverse)[1], 1)
    if isnothing(mu)
        mu = ones1
    end
    weight = inverse * mu
    weight /= transpose(ones1) * weight
    return weight
end
