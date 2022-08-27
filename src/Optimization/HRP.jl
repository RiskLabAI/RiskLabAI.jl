"""----------------------------------------------------------------------
    function: Generates random data
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 16.4, Page 241
----------------------------------------------------------------------"""
function randomData(numberObservations, #number of observation
                    size0, # size uncorrelated data
                    size1, # size correlated data
                    sigma1) # sigma for uncorrelated data
    data1 = rand(Normal(0, 1), numberObservations, size0) # random normal data
    columns = rand(1:size0, size1) # select random column
    data2 = data1[:, columns] + rand(Normal(0, sigma1), numberObservations, length(columns)) # add random shock into columns
    data = hcat(data1, data2) # merge two data
    dataframe = DataFrames.DataFrame(data, :auto) # dataframe of data
    DataFrames.rename!(dataframe, Symbol.(names(dataframe)) .=> Symbol.(collect(1:size(data)[2]))) # rename dataframe
    return dataframe, columns
end


"""----------------------------------------------------------------------
    function: Distance from corr matrix
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 16.4, Page 241
----------------------------------------------------------------------"""
function distanceCorr(correlation) # correlation matrix
    distance = (1 .- correlation) # each element of corr minus 1
    distance = distance.^0.5 # sqrt of distance
    return distance
end


"""----------------------------------------------------------------------
    function: The output is a sorted list of original items to reshape corr matrix
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 16.2, Page 229
----------------------------------------------------------------------"""
function quasiDiagonal(link) # linkage matrix
    # Sort clustered items by distance
    link = Int.(floor.(link)) # int each element
    sortedItems = DataFrames.DataFrame(index = [1, 2], value = [link[end,1], link[end, 2]]) # initialize sorted array
    numberItems = link[end, 4] # number of original items
    while maximum(sortedItems.value) >= numberItems 
        sortedItems.index = range(0,stop = size(sortedItems)[1]*2 - 1, step = 2) # make space
        dataframe = sortedItems[sortedItems.value .>= numberItems, :]  # find clusters
        index = dataframe.index # index
        value = dataframe.value .- numberItems # value
        sortedItems[in.(sortedItems.index, (index,)),:value] = link[value .+ 1,1]  # item 1
        dataframe = DataFrames.DataFrame(index = index .+ 1, value = link[value .+ 1, 2])
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
function recursiveBisection(cov, # covariance matrix
                            sortedItems) # sorted items from quasiDiagonal
    # Compute HRP alloc
    ω = DataFrames.DataFrame(index = sortedItems, weight = repeat([1.], length(sortedItems))) # initialize weight dataframe
    clusteredItems = [sortedItems] # initial cluster

    while length(clusteredItems) > 0 
        clusteredItems=[i[j:k] for i in clusteredItems for (j, k) in ((1, div(length(i), 2)), (div(length(i), 2) + 1, length(i))) if length(i) > 1] # bi-section
        for i in range(1, stop = length(clusteredItems), step = 2)
            clusteredItems0 = clusteredItems[i] # cluster 1
            clusteredItems1 = clusteredItems[i + 1] # cluster 2
            clusterVariance0 = varianceCluster(cov, clusteredItems0) # variance of cluster 1
            clusterVariance1 = varianceCluster(cov, clusteredItems1) # variance of cluster 2
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
function varianceCluster(cov, # covariance matrix
                         clusteredItems) # clustered items 
    # Compute variance per cluster
    covSlice = cov[clusteredItems .+ 1, clusteredItems .+ 1] # matrix slice
    ω = IVP(covSlice) # weight from inverse variance 
    clusterVariance = (transpose(ω)*covSlice)*ω #compute variance
    return clusterVariance
end

"""----------------------------------------------------------------------
function: inverse variance weights
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.4, Page 240
----------------------------------------------------------------------""" 
function IVP(cov, # covariance matrix
             kwargs...)
    # Compute the inverse-variance portfolio
    ivp = 1 ./Statistics.diag(cov) # inverse of diag of cov matrix
    ivp /= sum(ivp) # divide by sum(ivp)
    return ivp
end   
        
"""----------------------------------------------------------------------
function: random data for MC simulation
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.5, Page 242
----------------------------------------------------------------------"""   
function generalRandomData(numberObservations, # number of observation
                           lengthSample, # starting point for selecting random observation
                           size0, # size of uncorrelated data
                           size1,  # size of correlated data
                           mu0,  # mu for uncorrelated data
                           sigma0, # sigma for uncorrelated data
                           sigma1) # sigma for correlated data
    data1 = rand(Normal(mu0, sigma0), numberObservations, size0) # generate random uncorrelated data
    
    columns = rand(1:size0, size1) # select random number of columns
    data2 = data1[:, columns] + rand(Normal(0, sigma0*sigma1), numberObservations, length(columns)) # create correlation between the variables
    data = hcat(data1, data2) # merge data sets
    point = rand(lengthSample:numberObservations, 2) # randomly select elements
    data[append!(point, [columns[1], size0])] = [-.5, -.5, 2, 2] # add common random shock
    point = rand(lengthSample:numberObservations, 2) # randomly select elements
    data[point,columns[end]] = [-.5, 2] # add specific random shock
    #df = DataFrames.DataFrame(data, :auto)
    #DataFrames.rename!(df, Symbol.(names(df)).=>Symbol.(collect(1:size(data)[2])))
    return data, columns
end
        
"""----------------------------------------------------------------------
function: HRP method
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.5, Page 243
----------------------------------------------------------------------"""  
function HRP(cov::Matrix, # covariance matrix
             corr::Matrix) # correlation matrix
    # Construct a hierarchical portfolio
    distance = distanceCorr(corr) # distance matrix
    link = sch.linkage(distance,"single") # linkage matrix
    sortedItems = quasiDiagonal(link) # sorted items
    hrp = recursiveBisection(cov, sortedItems) # dataframe of weights
    return sort(hrp).weight # array of weights
end

"""----------------------------------------------------------------------
function: MC simulation for out of sample comparison
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 16.5, Page 243
----------------------------------------------------------------------"""  
function hrpMC(;numberIters = 5e3, # number of iterations
                numberObservations = 520, # number of observation
                size0 = 5, # size of uncorrelated data
                size1 = 5, # size of correlated data
                mu0 = 0, # mu for uncorrelated data
                sigma0 = 1e-2, # sigma for uncorrelated data
                sigma1 = .25, # sigma for correlated data
                lengthSample = 260, # length for in sample
                testSize = 20) # observation for test set
    methods = [IVP, HRP] # methods
    results, numIter = Dict(String.(Symbol.(methods)) .=> [[]]), 0 # initialize results and number of iteration
    pointers = range(lengthSample + 1, stop = numberObservations, step = testSize) # pointers for inSample and outSample
    while numIter < numberIters
        # println(numIter)
        data, columns = generalRandomData(numberObservations, lengthSample, size0, size1, mu0, sigma0, sigma1)  # Prepare data for one experiment
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