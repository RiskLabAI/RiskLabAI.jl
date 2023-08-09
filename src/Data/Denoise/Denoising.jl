"""
    Function: Implements the Marcenko–Pastur PDF in python.
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.1, Page 25
"""
function pdf_marcenko_pastur(var, ratio, points)
    λmin = var * (1 - sqrt(1 / ratio))^2
    λmax = var * (1 + sqrt(1 / ratio))^2
    eigen_values = range(λmin, stop=λmax, length=points)
    diffλ = ((λmax .- eigen_values) .* (eigen_values .- λmin))
    diffλ[diffλ .< -1E-3] .= 0.
    pdf = ratio ./ (2 * pi * var * eigen_values) .* diffλ
    return DataFrames.DataFrame(index=eigen_values, values=pdf)
end

"""
    Function: Get eVal, eVec from a Hermitian matrix.
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.2, Page 25
"""
function pca(matrix) # Hermitian matrix
    eigen_values, eigen_vectors = LinearAlgebra.eigen(matrix)
    indices = sortperm(eigen_values, rev=true)
    eigen_values, eigen_vectors = eigen_values[indices], eigen_vectors[:, indices]
    eigen_values = Diagonal(eigen_values)
    return eigen_values, eigen_vectors
end

"""
    Function: Fit kernel density.
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.2, Page 25
"""
function kde(observations; band_width=0.25, kernel=Distributions.Normal, values_for_evaluating=nothing)
    density = kde(observations, kernel=kernel, bandwidth=band_width)
    if values_for_evaluating == nothing
        values_for_evaluating = reshape(reverse(unique(observations)), :, 1)
    end
    density = KernelDensity.pdf(density, values_for_evaluating[:])
    return DataFrames.DataFrame(index=vec(values_for_evaluating), values=density)
end

"""
    Function: Add signal to a random covariance matrix.
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.3, Page 27
"""
function add_signal_to_covariance_matrix(number_columns, number_factors)
    data = rand(Normal(), number_columns, number_factors)
    cov_data = data * data'
    cov_data += Diagonal(rand(Uniform(), number_columns))
    return cov_data
end

"""
    Function: Derive the correlation matrix from a covariance matrix.
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.3, Page 27
"""
function covariance_to_correlation(cov)
    std = sqrt.((diag(cov)))
    corr = cov ./ (std .* std')
    corr[corr .< -1] .= -1
    corr[corr .> 1] .= 1
    return corr
end

"""
    Function: Fits the Marcenko–Pastur PDF to a random covariance
              matrix that contains signal. The objective of the fit is to find the value of σ2 that
              minimizes the sum of the squared differences between the analytical PDF and.
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.4, Page 27
"""
function fit_marcenko_pastur_to_covariance_matrix(var, eigen_values, ratio, band_width, points = 1000)
    pdf0 = pdf_marcenko_pastur(var, ratio, points)
    pdf1 = kde(eigen_values, band_width = band_width, kernel = Distributions.Normal, values_for_evaluating = pdf0.index)
    sse = sum((pdf1.values .- pdf0.values).^2)
    return sse
end

"""
    Function: Find the maximum random eigenvalues by fitting Marcenko's distribution.
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.4, Page 27
"""
function find_max_eigenvalues(eigen_values, ratio, band_width)
    out = optimize(var -> error_PDFs(var, eigen_values, ratio, band_width), 1E-5, 1-1E-5)
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
function constant_residual_eigenvalue_method(eigen_values, eigen_vectors, number_factors)
    λ = copy(diag(eigen_values))
    λ[number_factors:end] .= sum(λ[number_factors:end]) / (size(λ)[1] - number_factors)
    λ_diag = Diagonal(λ)
    cov = eigen_vectors * λ_diag * eigen_vectors'
    corr2 = covariance_to_correlation(cov)
    return corr2
end

"""
    Function: Denoising by Targeted Shrinkage
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.6, Page 31
"""
function denoised_corr_shrinkage(eigen_values, eigen_vectors, number_factors, α = 0)
    eigen_values_l = eigen_values[1:number_factors, 1:number_factors]
    eigen_vectors_l = eigen_vectors[:, 1:number_factors]
    eigen_values_r = eigen_values[number_factors:end, number_factors:end]
    eigen_vectors_r = eigen_vectors[:, number_factors:end]
    corr0 = eigen_vectors_l * eigen_values_l * transpose(eigen_vectors_l)
    corr1 = eigen_vectors_r * eigen_values_r * transpose(eigen_vectors_r)
    corr2 = corr0 + α * corr1 + (1 - α) * diagm(diag(corr1))
    return corr2
end

"""
    Function: Generate a block-diagonal covariance matrix and a vector of means.
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.7, Page 33
"""
function form_block_matrix(number_blocks, size_block, corr_block)
    block = ones(size_block, size_block) * corr_block
    block[Statistics.diagind(block)] .= 1.0
    corr = BlockArray{Float64}(undef_blocks, repeat([size_block], number_blocks), repeat([size_block], number_blocks))
    for i in range(1, stop = number_blocks)
        for j in range(1, stop = number_blocks)
            if i == j
                setblock!(corr, block, i, j)
            else
                setblock!(corr, zeros(size_block, size_block), i, j)
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
function generate_true_matrix(number_blocks, size_block, corr_block)
    corr = form_block_matrix(number_blocks, size_block, corr_block)
    columns =  Random.shuffle(collect(1:number_blocks * size_block))
    corr = corr[columns, columns]
    std0 = rand(Uniform(0.05, 0.2), size(corr)[1])
    cov0 = correlation_to_covariance(corr, std0)
    mu0 = rand.(Normal.(std0, std0), 1)
    return mu0, cov0
end

"""
    Function: Derive the covariance matrix from a correlation matrix
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.7, Page 33
"""
function correlation_to_covariance(corr, std)
    cov = corr .* (std .* std')
    return cov
end

"""
    Function: Generate the empirical covariance matrix using simulations.
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.8, Page 33
"""
function generate_empirical_covariance(mu0, cov0, observations, shrink = false)
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
function denoise_covariance(cov0, ratio, band_width)
    corr0 = covariance_to_correlation(cov0)
    eigen_values0, eigen_vectors0 = principal_component_analysis(corr0)
    λmax0, var0 = find_max_eigenvalues(diag(eigen_values0), ratio, band_width)
    number_factors0 = size(eigen_values0)[1] - searchsortedfirst(reverse(diag(eigen_values0)), λmax0) + 1
    corr1 = constant_residual_eigenvalue_method(eigen_values0, eigen_vectors0, number_factors0)
    cov1 = correlation_to_covariance(corr1, diag(cov0).^0.5)
    return cov1
end

"""
    Function: Perform Monte Carlo simulation to optimize portfolio weights.
    Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Snippet 2.10, Page 34
"""
function monte_carlo_optimize_portfolio(cov, mu = nothing)
    inverse = inv(cov)
    ones1 = ones(size(inverse)[1], 1)
    if isnothing(mu)
        mu = ones1
    end
    weight = inverse * mu
    weight /= transpose(ones1) * weight
    return weight
end
