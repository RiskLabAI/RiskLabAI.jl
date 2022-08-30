"""----------------------------------------------------------------------
    function:  implements the Marcenko–Pastur PDF in python
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.1, Page 25
----------------------------------------------------------------------"""
function pdfMarcenkoPastur(var, # variance of observations
                           ratio, # T/N
                           points) # points for lambda
    λmin = var*(1 - sqrt(1/ratio))^2 # minimum expected eigenvalue
    λmax = var*(1 + sqrt(1/ratio))^2 # maximum expected eigenvalue
    eigenValues = range(λmin, stop = λmax, length = points) # range for eigen values
    diffλ = ((λmax .- eigenValues).*(eigenValues .- λmin)) # numerical error
    diffλ[diffλ .< -1E-3] .= 0. # numerical error
    pdf = ratio./(2*pi*var*eigenValues).*diffλ # probability density function
    # pdf = ratio./(2*pi*var*eigenValues).*sqrt.(((λmax .- eigenValues).*(eigenValues .- λmin))) # probability density function
    return DataFrames.DataFrame(index = eigenValues, values = pdf)
end

"""----------------------------------------------------------------------
    function: Get eVal,eVec from a Hermitian matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.2, Page 25
----------------------------------------------------------------------"""
function PCA(matrix) # Hermitian matrix
    eigenValues, eigenVectors = LinearAlgebra.eigen(matrix) # compute eigenValues, eigenVectors from matrix
    indices = sortperm(eigenValues, rev = true) # arguments for sorting eigenValues desc
    eigenValues, eigenVectors = eigenValues[indices], eigenVectors[:, indices] # sort eigenValues, eigenVectors
    eigenValues = Diagonal(eigenValues) # diagonal matrix with eigenValues
    return eigenValues, eigenVectors
end

"""----------------------------------------------------------------------
    function: Fit kernel density
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.2, Page 25
----------------------------------------------------------------------"""
function KDE(observations; # Series of observations
             bandWidth = 0.25, 
             kernel = Distributions.Normal, # type of kernel
             valuesForEvaluating = nothing) # array of values on which the fit KDE will be evaluated   
    #=
    if length(size(observations)) == 1
        observations = reshape(observations, 1, :)
    end
    =#
    density = kde(observations, kernel = kernel, bandwidth = bandWidth) # kernel density
    if valuesForEvaluating == nothing
        valuesForEvaluating = reshape(reverse(unique(observations)), :, 1) # reshape valuesForEvaluating to vector
    end
    #=
    if length(size(valuesForEvaluating)) == 1
        valuesForEvaluating = reshape(valuesForEvaluating, 1, :)
    end
    =#
    #k = kde(observations, xeval = valuesForEvaluating[:], h = bandWidth)
    
    density = KernelDensity.pdf(density, valuesForEvaluating[:]) # probability density function
    return DataFrames.DataFrame(index = vec(valuesForEvaluating), values = density)
end

"""----------------------------------------------------------------------
    function: ADD SIGNAL TO A RANDOM COVARIANCE MATRIX
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.3, Page 27
----------------------------------------------------------------------"""
function randomCov(numberColumns, # number of columns
                   numberFactors) # number of factors
    data = rand(Normal(), numberColumns, numberFactors) # random data
    covData = data*data' # covariance of data
    covData += Diagonal(rand(Uniform(), numberColumns)) # add noise to the matrix
    return covData
end

"""----------------------------------------------------------------------
    function: Derive the correlation matrix from a covariance matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.3, Page 27
----------------------------------------------------------------------"""
function covToCorr(cov) # covariance matrix
    std = sqrt.((diag(cov))) # standard deviations
    corr = cov./(std.*std') # create correlation matrix
    corr[corr .< -1] .= -1 # numerical error
    corr[corr .> 1] .= 1 # numerical error
    return corr
end

"""----------------------------------------------------------------------
    function: Fits the Marcenko–Pastur PDF to a random covariance
              matrix that contains signal. The objective of the fit is to find the value of σ2 that
              minimizes the sum of the squared differences between the analytical PDF and
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.4, Page 27
----------------------------------------------------------------------"""
function errorPDFs(var, # variance
                   eigenValues, # eigenvalues
                   ratio, # T/N
                   bandWidth; # band width for kernel
                   points = 1000) # points for pdfMarcenkoPastur
   pdf0 = pdfMarcenkoPastur(var, ratio, points) # theoretical pdf
   pdf1 = KDE(eigenValues, bandWidth = bandWidth, kernel = Distributions.Normal, valuesForEvaluating = pdf0.index) # empirical pdf
   sse = sum((pdf1.values .- pdf0.values).^2) # sum of squares of errors
   return sse 
end 

"""----------------------------------------------------------------------
    function: Find max random eigenValues by fitting Marcenko’s dist
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.4, Page 27
----------------------------------------------------------------------"""
function findMaxEval(eigenValues, # eigenvalues
                     ratio, # T/N
                     bandWidth) # band width for kernel
    out = optimize(var->errorPDFs(var, eigenValues, ratio, bandWidth), 1E-5, 1-1E-5) # minimize pdferrors
    if Optim.converged(out) == true
        var = Optim.minimizer(out) # variance that minimizes pdferrors
    else
        var = 1
    end
    λmax = var*(1 + (1/ratio)^.5)^2 # max random eigenvalue
    return λmax, var
end
   
"""----------------------------------------------------------------------
    function: Constant Residual Eigenvalue Method
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.5, Page 29
----------------------------------------------------------------------"""
function denoisedCorr(eigenValues, # eigenvalues
                      eigenVectors,  # eigenvectors
                      numberFactors) # number of factors
    λ = copy(diag(eigenValues)) # copy eigenvalues
    λ[numberFactors:end] .= sum(λ[numberFactors:end])/(size(λ)[1] - numberFactors)
    λdiag = Diagonal(λ) # diagonal matrix with λ
    cov = eigenVectors * λdiag * eigenVectors' # covariance matrix
    corr2 = covToCorr(cov) # correlation matrix
    return corr2
end

"""----------------------------------------------------------------------
    function: DENOISING BY TARGETED SHRINKAGE
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.6, Page 31
----------------------------------------------------------------------"""
function denoisedCorrShrinkage(eigenValues, # eigen values
                               eigenVectors, # eigen vectors
                               numberFactors; # number of factors
                               α = 0) # parameter for shrinkage
   eigenValuesL = eigenValues[1:numberFactors, 1:numberFactors] # divide eigenValues
   eigenVectorsL = eigenVectors[:, 1:numberFactors] # divide eigenVectors
   eigenValuesR = eigenValues[numberFactors:end, numberFactors:end] # divide eigenValues
   eigenVectorsR = eigenVectors[:, numberFactors:end] # divide eigenVectors
   corr0 = eigenVectorsL * eigenValuesL * transpose(eigenVectorsL) # correlation matrix 1
   corr1 = eigenVectorsR * eigenValuesR * transpose(eigenVectorsR) # correlation matrix 2
   corr2 = corr0 + α*corr1 + (1 - α)*diagm(diag(corr1)) # correlation matrix
   return corr2 
end

"""----------------------------------------------------------------------
    function: GENERATING A BLOCK-DIAGONAL COVARIANCE MATRIX AND A VECTOR Of MEANS
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.7, Page 33
----------------------------------------------------------------------"""
function formBlockMatrix(numberBlocks, # number of blocks
                         sizeBlock, # size of block
                         corrBlock) # correlation in block
    block = ones(sizeBlock, sizeBlock)*corrBlock # ones matrix
    block[Statistics.diagind(block)] .= 1. # set diag to 1
    corr = BlockArray{Float64}(undef_blocks, repeat([sizeBlock], numberBlocks), repeat([sizeBlock], numberBlocks)) # corr matrix
    # change block array values
    for i in range(1,stop = numberBlocks)
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
function formTrueMatrix(numberBlocks, # number of blocks
                        sizeBlock, # size of block
                        corrBlock) # correlation in block
    corr = formBlockMatrix(numberBlocks, sizeBlock, corrBlock) # corr matrix
    columns =  Random.shuffle(collect(1:numberBlocks*sizeBlock)) # shuffle columns
    corr = corr[columns, columns] # shuffled corr matrix
    std0 = rand(Uniform(.05, 0.2), size(corr)[1]) # standard deviations
    cov0 = corrToCov(corr, std0) # cov matrix
    mu0 = rand.(Normal.(std0, std0), 1) # mu vector
    return mu0, cov0
end

"""----------------------------------------------------------------------
    function: Derive the covarinace matrix from a correlation matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.7, Page 33
----------------------------------------------------------------------"""
function corrToCov(corr, # correlation matrix
                   std) # standard deviations
    cov = corr.*(std.*std') # covarinance matrix
    return cov
end

"""----------------------------------------------------------------------
    function: GENERATE THE EMPIRICAL COVARIANCE MATRIX
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.8, Page 33
----------------------------------------------------------------------"""
function simCovMu(mu0, # mean vector
                  cov0, # covariance matrix
                  observations, # number of observations
                  shrink = false) # shrinkage
    data = transpose(rand(MvNormal(vcat(mu0...), cov0), observations)) # generate data
    mu1 = vec(reshape(mean(data, dims = 1), (500,1))) # mean data
    if shrink
        cov1 = LedoitWolf().fit(data).covariance_ # ledoitwolf
    else
        cov1 = cov(data)
    end
    return mu1, cov1
end

"""----------------------------------------------------------------------
    function: DENOISING OF THE EMPIRICAL COVARIANCE MATRIX
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.9, Page 34
----------------------------------------------------------------------"""
function deNoiseCov(cov0, # covarinace matrix
                    ratio, # T/N
                    bandWidth) # band width
    corr0 = covToCorr(cov0) # correlation matrix
    eigenValues0, eigenVectors0 = PCA(corr0) # eigen values and vectors from pca
    λmax0, var0 = findMaxEval(diag(eigenValues0), ratio, bandWidth) # find maximum eigen value
    numberFactors0 = size(eigenValues0)[1] - searchsortedfirst(reverse(diag(eigenValues0)), λmax0) + 1 # compute number of factors
    corr1 = denoisedCorr(eigenValues0, eigenVectors0, numberFactors0) # denoise correlation matrix
    cov1 = corrToCov(corr1, diag(cov0).^.5) # covariance matrix from corr
    return cov1
end

"""----------------------------------------------------------------------
    function: Monte Carlo simulation
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.10, Page 34
----------------------------------------------------------------------"""
function optPort(cov, # covariance matrix
                 mu = nothing) # mean vector
    inverse = inv(cov) # inverse of cov 
    ones1 = ones(size(inverse)[1], 1) # ones matrix for mu
    if isnothing(mu) 
        mu = ones1
    end
    weight = inverse*mu # compute weights
    weight /= transpose(ones1)*weight # normalize weights
    return weight
end
