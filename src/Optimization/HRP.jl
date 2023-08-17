using DataFrames, Statistics, LinearAlgebra, Random, Distributions

"""
    recursiveBisection(cov::Matrix, sortedItems::Vector{Int})

Calculate the weights of assets using recursive bisection.

This function computes the Hierarchical Risk Parity (HRP) allocations of a set of assets
based on their covariance matrix using the recursive bisection technique.

# Arguments
- `cov::Matrix`: Covariance matrix of the assets.
- `sortedItems::Vector{Int}`: Vector of integers representing the indices of the sorted assets.

# Returns
- `DataFrame`: A DataFrame with the asset indices as rows and their corresponding weights.

# References
- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
  Methodology: Snippet 16.2, Page 230.
"""
function recursiveBisection(
        cov::Matrix,
        sortedItems::Vector{Int}
    )
    # Initialize weight DataFrame
    ω = DataFrame(index = sortedItems, weight = ones(length(sortedItems)))
    clusteredItems = [sortedItems]  # Initial cluster
    
    while length(clusteredItems) > 0
        clusteredItems = [i[j:k] for i in clusteredItems for (j, k) in ((1, div(length(i), 2)), (div(length(i), 2) + 1, length(i))) if length(i) > 1]  # Bisection
        
        for i in 1:2:length(clusteredItems)  # Iterate over even indices
            cluster0 = clusteredItems[i]  # Cluster 1
            cluster1 = clusteredItems[i + 1]  # Cluster 2
            
            clusterVariance0 = clusterVariance(cov, cluster0)  # Variance of cluster 1
            clusterVariance1 = clusterVariance(cov, cluster1)  # Variance of cluster 2
            α = 1 - clusterVariance0 / (clusterVariance0 + clusterVariance1)  # Set alpha
            
            ω[in.(ω.index, Ref(cluster0)), :weight] .*= α  # Weight 1
            ω[in.(ω.index, Ref(cluster1)), :weight] .*= 1 - α  # Weight 2
        end
    end
    
    return ω
end


"""
    clusterVariance(cov::Matrix, clusteredItems::Vector{Int})

Calculate the variance of a cluster.

# Arguments
- `cov::Matrix`: Covariance matrix of the assets.
- `clusteredItems::Vector{Int}`: Vector of integers representing the indices of the clustered assets.

# Returns
- `Float64`: The variance of the cluster.

# References
- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
  Methodology: Snippet 16.4, Page 240.
"""
function clusterVariance(
        cov::Matrix,
        clusteredItems::Vector{Int}
    )

    covSlice = cov[clusteredItems, clusteredItems]  # Matrix slice
    ω = inverseVariancePortfolio(covSlice)  # Weight from inverse variance 
    return dot(ω, covSlice * ω)  # Compute variance
end


"""
    inverseVariancePortfolio(cov::Matrix)

Calculate inverse variance weights.

# Arguments
- `cov::Matrix`: Covariance matrix of the assets.

# Returns
- `Vector{Float64}`: The inverse variance portfolio weights.

# References
- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
  Methodology: Snippet 16.4, Page 240.
"""
function inverseVariancePortfolio(
        cov::Matrix
    )
    # Compute the inverse-variance portfolio
    ω = 1 ./ diagm(cov)  # Inverse of diagonal of covariance matrix
    return ω ./ sum(ω)  # Normalize
end


"""
    generateRandomData(nObservations::Int, lengthSample::Int, size0::Int, size1::Int, mu0::Float64, sigma0::Float64, sigma1::Float64)

Generate random data for Monte Carlo simulation.

# Arguments
- `nObservations::Int`: Number of observations.
- `lengthSample::Int`: Starting point for selecting random observations.
- `size0::Int`: Size of uncorrelated data.
- `size1::Int`: Size of correlated data.
- `mu0::Float64`: Mean for uncorrelated data.
- `sigma0::Float64`: Standard deviation for uncorrelated data.
- `sigma1::Float64`: Standard deviation for correlated data.

# Returns
- `Matrix`: A matrix with the random data.
- `Vector{Int}`: A vector of integers representing the indices of the selected random columns.

# References
- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
  Methodology: Snippet 16.5, Page 242.
"""
function generateRandomData(
        nObservations::Int,
        lengthSample::Int,
        size0::Int,
        size1::Int,
        mu0::Float64,
        sigma0::Float64,
        sigma1::Float64
    )
    data1 = rand(Normal(mu0, sigma0), nObservations, size0)  # Generate random uncorrelated data
    
    columns = sample(1:size0, size1, replace=false)  # Select random columns without replacement
    data2 = data1[:, columns] + rand(Normal(0, sigma0 * sigma1), nObservations, length(columns))  # Create correlation between the variables
    data = hcat(data1, data2)  # Merge data sets
    
    point = rand(lengthSample:nObservations, 2)  # Randomly select elements
    data[append!(point, [columns[1], size0])] = [-0.5, -0.5, 2, 2]  # Add common random shock
    
    point = rand(lengthSample:nObservations, 2)  # Randomly select elements
    data[point, columns[end]] = [-0.5, 2]  # Add specific random shock

    return data, columns
end
using LinearAlgebra, Statistics, DataFrames, Clustering, Distributions, Distances

"""
    hierarchicalRiskParity(cov::Matrix, corr::Matrix)

Calculate hierarchical risk parity (HRP) portfolio weights.

# Arguments
- `cov::Matrix`: Covariance matrix.
- `corr::Matrix`: Correlation matrix.

# Returns
- `::Vector{Float64}`: Array of weights.

# References
De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Methodology: Snippet 16.5, Page 243.
"""
function hierarchicalRiskParity(
        cov::Matrix,
        corr::Matrix
    )
    distance = Distances.pairwise(Euclidean(), corr)
    linkageMatrix = Clustering.hclust(distance, linkage=:single)
    sortedItems = Clustering.order(linkageMatrix)
    hrp = recursiveBisection(cov, sortedItems)
    return sort(hrp).weight
end

"""
    hrpMonteCarlo(; nIterations::Int, nObservations::Int, size0::Int, size1::Int, mu0::Float64, sigma0::Float64, sigma1::Float64, lengthSample::Int, testSize::Int)

Perform Monte Carlo simulation for out-of-sample comparison.

# Arguments
- `nIterations::Int`: Number of iterations.
- `nObservations::Int`: Number of observations.
- `size0::Int`: Size of uncorrelated data.
- `size1::Int`: Size of correlated data.
- `mu0::Float64`: Mean for uncorrelated data.
- `sigma0::Float64`: Standard deviation for uncorrelated data.
- `sigma1::Float64`: Standard deviation for correlated data.
- `lengthSample::Int`: Length for in-sample.
- `testSize::Int`: Number of observations in the test set.

# Returns
- `::DataFrame`: DataFrame of results with standard deviation and variance of each method.

# References
De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Methodology: Snippet 16.5, Page 243.
"""
function hrpMonteCarlo(;
        nIterations::Int = 5000,
        nObservations::Int = 520,
        size0::Int = 5,
        size1::Int = 5,
        mu0::Float64 = 0,
        sigma0::Float64 = 0.01,
        sigma1::Float64 = 0.25,
        lengthSample::Int = 260,
        testSize::Int = 20
    )
    methods = [inverseVariancePortfolio, hierarchicalRiskParity]
    results, numIter = Dict(String.(Symbol.(methods)) .=> [[]]), 0
    pointers = range(lengthSample + 1, stop = nObservations, step = testSize)
    
    while numIter < nIterations
        data, columns = generalRandomData(nObservations, lengthSample, size0, size1, mu0, sigma0, sigma1)
        returns = Dict(String.(Symbol.(methods)) .=> [[]])
        
        for pointer in pointers
            inSample = data[pointer - lengthSample:pointer - 1, :]
            covariance = cov(inSample)
            correlation = cor(inSample)
            outSample = data[pointer:pointer + testSize - 1, :]
            
            for func in methods
                weights = func(covariance, correlation)
                ret = outSample * weights
                returns[String(Symbol(func))] = vcat(returns[String(Symbol(func))], ret)
            end
        end
        
        for func in methods
            ret = returns[String(Symbol(func))]
            cumprodReturn = cumprod(ret .+ 1)
            results[String(Symbol(func))] = vcat(results[String(Symbol(func))], [cumprodReturn[end] - 1])
        end
        
        numIter += 1
    end
    
    results = DataFrame(results)
    stdResults, varResults = std.(eachcol(results)), var.(eachcol(results))
    return hcat(stdResults, varResults, varResults ./ varResults[2] .- 1)
end
