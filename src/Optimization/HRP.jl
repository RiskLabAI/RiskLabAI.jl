"""
Calculate the weights of assets using recursive bisection.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 16.2, Page 230
"""
function recursiveBisection(
        cov::Matrix,  # Covariance matrix
        sortedItems
    )
    # Compute HRP allocations
    ω = DataFrames.DataFrame(index = sortedItems, weight = repeat([1.], length(sortedItems)))  # Initialize weight DataFrame
    clusteredItems = [sortedItems]  # Initial cluster
    
    while length(clusteredItems) > 0
        clusteredItems = [i[j:k] for i in clusteredItems for (j, k) in ((1, div(length(i), 2)), (div(length(i), 2) + 1, length(i))) if length(i) > 1]  # Bisection
        
        for i in range(1, stop = length(clusteredItems), step = 2)
            clusteredItems0 = clusteredItems[i]  # Cluster 1
            clusteredItems1 = clusteredItems[i + 1]  # Cluster 2
            
            clusterVariance0 = clusterVariance(cov, clusteredItems0)  # Variance of cluster 1
            clusterVariance1 = clusterVariance(cov, clusteredItems1)  # Variance of cluster 2
            α = 1 - clusterVariance0 / (clusterVariance0 + clusterVariance1)  # Set alpha
            
            ω[in.(ω.index, (clusteredItems0,)), :weight] .*= α  # Weight 1
            ω[in.(ω.index, (clusteredItems1,)), :weight] .*= 1 - α  # Weight 2
        end
    end
    
    return ω
end


"""
Calculate the variance of a cluster.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 16.4, Page 240
"""
function clusterVariance(
        cov::Matrix,  # Covariance matrix
        clusteredItems  # Clustered items 
    )

    # Compute variance per cluster
    covSlice = cov[clusteredItems .+ 1, clusteredItems .+ 1]  # Matrix slice
    ω = inverseVariancePortfolio(covSlice)  # Weight from inverse variance 
    clusterVariance = (transpose(ω) * covSlice) * ω  # Compute variance
    return clusterVariance
end


"""
Calculate inverse variance weights.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 16.4, Page 240
"""
function inverseVariancePortfolio(
        cov,  # Covariance matrix
        kwargs...
    )
    # Compute the inverse-variance portfolio
    ω = 1 ./ Statistics.diag(cov)  # Inverse of diagonal of covariance matrix
    ω /= sum(ω)  # Normalize
    return ω
end


"""
Generate random data for Monte Carlo simulation.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 16.5, Page 242
"""
function generalRandomData(
        nObservations,  # Number of observations
        lengthSample,  # Starting point for selecting random observations
        size0,  # Size of uncorrelated data
        size1,  # Size of correlated data
        mu0,  # Mean for uncorrelated data
        sigma0,  # Standard deviation for uncorrelated data
        sigma1  # Standard deviation for correlated data
    )
    data1 = rand(Normal(mu0, sigma0), nObservations, size0)  # Generate random uncorrelated data
    
    columns = rand(1:size0, size1)  # Select random columns
    data2 = data1[:, columns] + rand(Normal(0, sigma0 * sigma1), nObservations, length(columns))  # Create correlation between the variables
    data = hcat(data1, data2)  # Merge data sets
    
    point = rand(lengthSample:nObservations, 2)  # Randomly select elements
    data[append!(point, [columns[1], size0])] = [-0.5, -0.5, 2, 2]  # Add common random shock
    
    point = rand(lengthSample:nObservations, 2)  # Randomly select elements
    data[point, columns[end]] = [-0.5, 2]  # Add specific random shock

    return data, columns
end


"""
Calculate hierarchical risk parity (HRP) portfolio weights.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 16.5, Page 243
"""
function hierarchicalRiskParity(
        cov::Matrix,  # Covariance matrix
        corr::Matrix  # Correlation matrix
    )
    # Construct a hierarchical portfolio
    distance = distanceCorr(corr)  # Distance matrix
    linkageMatrix = sch.linkage(distance, "single")  # Linkage matrix
    sortedItems = quasiDiagonal(linkageMatrix)  # Sorted items
    hrp = recursiveBisection(cov, sortedItems)  # DataFrame of weights
    return sort(hrp).weight  # Array of weights
end


"""
Perform Monte Carlo simulation for out-of-sample comparison.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
Methodology: Snippet 16.5, Page 243
"""
function hrpMC(;
        nIterations = 5e3,  # Number of iterations
        nObservations = 520,  # Number of observations
        size0 = 5,  # Size of uncorrelated data
        size1 = 5,  # Size of correlated data
        mu0 = 0,  # Mean for uncorrelated data
        sigma0 = 1e-2,  # Standard deviation for uncorrelated data
        sigma1 = 0.25,  # Standard deviation for correlated data
        lengthSample = 260,  # Length for in-sample
        testSize = 20  # Number of observations in the test set
    )
    methods = [inverseVariancePortfolio, hierarchicalRiskParity]  # Methods
    results, numIter = Dict(String.(Symbol.(methods)) .=> [[]]), 0  # Initialize results and number of iterations
    pointers = range(lengthSample + 1, stop = nObservations, step = testSize)  # Pointers for in-sample and out-sample
    
    while numIter < nIterations
        data, columns = generalRandomData(nObservations, lengthSample, size0, size1, mu0, sigma0, sigma1)  # Prepare data for one experiment
        returns = Dict(String.(Symbol.(methods)) .=> [[]])  # Initialize returns
        
        # Compute portfolios in-sample
        for pointer in pointers
            inSample = data[pointer - lengthSample:pointer - 1, :]  # In-sample
            covariance = cov(inSample)  # Covariance of sample
            correlation = cor(inSample)  # Correlation of sample
            
            # Compute performance out-of-sample
            outSample = data[pointer:pointer + testSize - 1, :]  # Out-of-sample 
            
            for func in methods
                weights = func(covariance, correlation)  # Calculate portfolio weights
                ret = outSample * weights  # Calculate returns
                returns[String(Symbol(func))] = vcat(returns[String(Symbol(func))], ret)  # Update returns dictionary
            end
        end
        
        # Evaluate and store results
        for func in methods
            ret = returns[String(Symbol(func))]  # Returns for each method
            cumprodReturn = cumprod(ret .+ 1)  # Cumulative product of returns
            results[String(Symbol(func))] = vcat(results[String(Symbol(func))], [cumprodReturn[end] - 1])  # Update results
        end
        
        numIter += 1  # Next iteration
    end
    
    # Report results
    results = DataFrames.DataFrame(results)  # DataFrame of results
    stdResults, varResults = std.(eachcol(results)), var.(eachcol(results))  # Standard deviation and variance of each method
    return hcat(stdResults, varResults, varResults ./ varResults[2] .- 1)
end
