"""----------------------------------------------------------------------
    function: Distance from corr matrix
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 16.4, Page 241
----------------------------------------------------------------------"""
function distanceCorr(
    correlation::AbstractMatrix # correlation matrix
) 
    return (1 .- correlation)^.5
end


"""----------------------------------------------------------------------
    function: The output is a sorted list of original items to reshape corr matrix
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 16.2, Page 229
----------------------------------------------------------------------"""
function quasiDiagonal(linkageMatrix) # linkage matrix
    # Sort clustered items by distance
    linkageMatrix = Int.(floor.(linkageMatrix)) # int each element
    sortedItems = DataFrames.DataFrame(index = [1, 2], value = [linkageMatrix[end,1], linkageMatrix[end, 2]]) # initialize sorted array
    nItems = linkageMatrix[end, 4] # number of original items
    while maximum(sortedItems.value) >= nItems 
        sortedItems.index = range(0,stop = size(sortedItems)[1]*2 - 1, step = 2) # make space
        dataframe = sortedItems[sortedItems.value .>= nItems, :]  # find clusters
        index = dataframe.index # index
        value = dataframe.value .- nItems # value
        sortedItems[in.(sortedItems.index, (index,)),:value] = linkageMatrix[value .+ 1,1]  # item 1
        dataframe = DataFrames.DataFrame(index = index .+ 1, value = linkageMatrix[value .+ 1, 2])
        sortedItems = vcat(sortedItems, dataframe) # item 2
        sort!(sortedItems, by = x->x[1]) # re-sort
        sortedItems.index = range(0, length = size(sortedItems)[1]) # re-index
    end
    return sortedItems.value
end


"""----------------------------------------------------------------------
    function: The output is a dataframe including weights of assets
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 16.2, Page 230
----------------------------------------------------------------------"""
function recursiveBisection(
        cov::Matrix, # covariance matrix
        sortedItems
) # sorted items from quasiDiagonal
    # Compute HRP alloc
    ω = DataFrames.DataFrame(index = sortedItems, weight = repeat([1.], length(sortedItems))) # initialize weight dataframe
    clusteredItems = [sortedItems] # initial cluster

    while length(clusteredItems) > 0 
        clusteredItems=[i[j:k] for i in clusteredItems for (j, k) in ((1, div(length(i), 2)), (div(length(i), 2) + 1, length(i))) if length(i) > 1] # bi-section
        for i in range(1, stop = length(clusteredItems), step = 2)
            clusteredItems0 = clusteredItems[i] # cluster 1
            clusteredItems1 = clusteredItems[i + 1] # cluster 2
            clusterVariance0 = clusterVariance(cov, clusteredItems0) # variance of cluster 1
            clusterVariance1 = clusterVariance(cov, clusteredItems1) # variance of cluster 2
            α = 1 - clusterVariance0/(clusterVariance0 + clusterVariance1) # set alpha

            ω[in.(ω.index, (clusteredItems0,)), :weight] .*= α # weight 1
            ω[in.(ω.index, (clusteredItems1,)), :weight] .*= 1 - α # weight 2
        end
    end
    return ω
end
        
"""----------------------------------------------------------------------
function: Compute variance of cluster
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.4, Page 240
----------------------------------------------------------------------""" 
function clusterVariance(
        cov::Matrix, # covariance matrix
        clusteredItems # clustered items 
)
    # Compute variance per cluster
    covSlice = cov[clusteredItems .+ 1, clusteredItems .+ 1] # matrix slice
    ω = inverseVariancePortfolio(covSlice) # weight from inverse variance 
    clusterVariance = (transpose(ω)*covSlice)*ω #compute variance
    return clusterVariance
end

"""----------------------------------------------------------------------
function: inverse variance weights
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.4, Page 240
----------------------------------------------------------------------""" 
function inverseVariancePortfolio(cov, # covariance matrix
             kwargs...)
    # Compute the inverse-variance portfolio
    ω = 1 ./Statistics.diag(cov) # inverse of diag of cov matrix
    ω /= sum(ω) # divide by sum(ivp)
    return ω
end   
        
"""----------------------------------------------------------------------
function: random data for MC simulation
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.5, Page 242
----------------------------------------------------------------------"""   
function generalRandomData(nObservations, # number of observation
                           lengthSample, # starting point for selecting random observation
                           size0, # size of uncorrelated data
                           size1,  # size of correlated data
                           mu0,  # mu for uncorrelated data
                           sigma0, # sigma for uncorrelated data
                           sigma1) # sigma for correlated data
    data1 = rand(Normal(mu0, sigma0), nObservations, size0) # generate random uncorrelated data
    
    columns = rand(1:size0, size1) # select random number of columns
    data2 = data1[:, columns] + rand(Normal(0, sigma0*sigma1), nObservations, length(columns)) # create correlation between the variables
    data = hcat(data1, data2) # merge data sets
    point = rand(lengthSample:nObservations, 2) # randomly select elements
    data[append!(point, [columns[1], size0])] = [-.5, -.5, 2, 2] # add common random shock
    point = rand(lengthSample:nObservations, 2) # randomly select elements
    data[point,columns[end]] = [-.5, 2] # add specific random shock

    return data, columns

end
        
"""----------------------------------------------------------------------
function: HRP method
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.5, Page 243
----------------------------------------------------------------------"""  
function hierarchicalRiskParity(cov::Matrix, # covariance matrix
             corr::Matrix) # correlation matrix
    # Construct a hierarchical portfolio
    distance = distanceCorr(corr) # distance matrix
    linkageMatrix = sch.linkage(distance,"single") # linkage matrix
    sortedItems = quasiDiagonal(linkageMatrix) # sorted items
    hrp = recursiveBisection(cov, sortedItems) # dataframe of weights
    return sort(hrp).weight # array of weights
end

"""----------------------------------------------------------------------
function: MC simulation for out of sample comparison
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.5, Page 243
----------------------------------------------------------------------"""  
function hrpMC(;nIterations = 5e3, # number of iterations
                nObservations = 520, # number of observation
                size0 = 5, # size of uncorrelated data
                size1 = 5, # size of correlated data
                mu0 = 0, # mu for uncorrelated data
                sigma0 = 1e-2, # sigma for uncorrelated data
                sigma1 = .25, # sigma for correlated data
                lengthSample = 260, # length for in sample
                testSize = 20) # observation for test set
    methods = [inverseVariancePortfolio, hierarchicalRiskParity] # methods
    results, numIter = Dict(String.(Symbol.(methods)) .=> [[]]), 0 # initialize results and number of iteration
    pointers = range(lengthSample + 1, stop = nObservations, step = testSize) # pointers for inSample and outSample
    while numIter < nIterations
        # println(numIter)
        data, columns = generalRandomData(nObservations, lengthSample, size0, size1, mu0, sigma0, sigma1)  # Prepare data for one experiment
        returns = Dict(String.(Symbol.(methods)) .=> [[]]) # initialize returns
        # Compute portfolios in-sample
        for pointer in pointers
            inSample = data[pointer - lengthSample:pointer - 1, :] # in sample
            covariance = cov(inSample) # covariance of sample
            correlation = cor(inSample) # correlation of sample
            # Compute performance out-of-sample
            outSample = data[pointer:pointer + testSize - 1, :] # out of sample 
            for func in methods
                weights = func(covariance, correlation) # call methods
                ret = outSample*weights # returns
                returns[String(Symbol(func))] = vcat(returns[String(Symbol(func))], ret) # update returns dictionary
            end
        end
        # Evaluate and store results
        for func in methods
            ret = returns[String(Symbol(func))] # return column of each method
            cumprodReturn = cumprod(ret.+1) # cumprod of returns
            results[String(Symbol(func))] = vcat(results[String(Symbol(func))], [cumprodReturn[end] - 1])  # update results
        end
        numIter += 1 # next iteration
    end
    # Report results
    results = DataFrames.DataFrame(results) # dataframe of results
    stdResults, varResults = std.(eachcol(results)), var.(eachcol(results)) # std and var of each method
    return hcat(stdResults, varResults, varResults./varResults[2] .- 1)
end