#using KernelEstimator
using Distributions
using LinearAlgebra
using DataFrames
using Statistics
using BlockArrays
using Shuffle
using Random
using Clustering
using Dates
using MarketData
using CSV
using StatsBase
using BlockDiagonals
@pyimport sklearn.metrics as Metrics

"""
    function: Calculate returns
    reference: n/a
    methodology: n/a
"""
function percentChange(prices::DataFrames.DataFrame)
    returns = DataFrames.DataFrame() # empty dataframe of returns
    for sym in names(prices)[2:end]
        data = prices[!, Symbol(sym)] # prices of each name
        ret = Array{Float64}(undef,length(data)) # returns of each name
        ret[1] = NaN
        for i in 2:length(data)
            ret[i] = (data[i]/data[i-1]) - 1 # calculate returns of each name
        end
        returns[!, Symbol(sym)] = ret # append returns into dataframe
    end
    return returns
end

"""
    function: Clustering
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.1, Page 56
"""
function clusterKMeansBase(correlation; 
                           numberClusters = 10, 
                           iterations = 10)
    distance = sqrt.((1 .- correlation)/2) # distance matrix
    silh, kmeansOut = [NaN], [NaN] # initial value for silh, kmeans
    for init ∈ 1:iterations
        for i ∈ 2:numberClusters
            kmeans_ = kmeans(distance, i) # clustering distance with maximum cluster i
            silh_ = Metrics.silhouette_samples(distance, assignments(kmeans_)) # silh score of clustering
            statistic = (mean(silh_)/std(silh_), mean(silh)/std(silh)) # calculate t-statistic
            if isnan(statistic[2]) || statistic[1]>statistic[2]
                silh, kmeansOut = silh_, kmeans_ # replace better clustering
            end
        end
    end
    indexSorted = sortperm(assignments(kmeansOut)) # sort arguments based on clustering
    correlationSorted = correlation[indexSorted, indexSorted] # new corr matrix based on clustering
    # dictionary of clustering
    clusters = Dict("$i"=> filter(p->assignments(kmeansOut)[p] == i, indexSorted) for i in unique(assignments(kmeansOut)))
    silh = DataFrames.DataFrame(silh = silh) # dataframe of silh scores
    return correlationSorted, clusters, silh, indexSorted
end

"""
    function: make new clustering
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.2, Page 58
"""
function makeNewOutputs(correlation, # corr dataframe
                        clusters,   # cluster 1
                        clusters2)  # cluster 2
    assets = names(correlation)# name of the columns of corr dataframe 
    # merge two clusters
    clustersNew = Dict()
    for i in keys(clusters)
        clustersNew[length(keys(clustersNew)) + 1] = clusters[i] 
    end
    for i in keys(clusters2)
        clustersNew[length(keys(clustersNew)) + 1] = clusters2[i]
    end
    indexNew = [j for i in keys(clustersNew) for j in clustersNew[i]] # sorted index of assets
    correlationNew = correlation[indexin(indexNew, assets), indexin(indexNew, assets)] # new corr matrix
    distance = sqrt.((1 .- Matrix(correlation))/2) # distance matrix
    labelsKmeans = zeros(size(distance)[2]) # initial labels
    for i in keys(clustersNew)
        index = indexin(clustersNew[i], assets) 
        labelsKmeans[index] .= i # label for clusters
    end
    silhNew = DataFrames.DataFrame(index = assets, silh = Metrics.silhouette_samples(distance, labelsKmeans)) # silh series
    return correlationNew,clustersNew,silhNew
end

"""
    function: clustering (ONC)
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.2, Page 58
"""
function clusterKMeansTop(correlation; # corr dataframe  
                          numberClusters = nothing, # number of clusters
                          iterations = 10) # number of iterations
    if isnothing(numberClusters)
        numberClusters = size(correlation)[2] - 1 # set number of cluster
    end
    assets = names(correlation) # names of columns
    # clustering
    correlationSorted, clusters, silh, indexSorted = clusterKMeansBase(Matrix(correlation), numberClusters = min(numberClusters, size(correlation)[2] - 1), iterations = 10)
    correlationSorted = DataFrames.DataFrame(correlationSorted, :auto) # dataframe of correlationSorted
    DataFrames.rename!(correlationSorted, Symbol.(names(correlationSorted)) .=> assets[indexSorted]) # rename columns of the dataframe of correlationSorted
    clusters = Dict("$i" => assets[clusters[i]] for i in keys(clusters)) # dictionary of clusters
    # calcultae t-statistic of each cluster
    clusterTstats = Dict("$i" => mean(silh[indexin(clusters[i], assets), :silh])/std(silh[indexin(clusters[i], assets), :silh]) for i in keys(clusters))
    tStatMean = sum(values(clusterTstats))/length(clusterTstats) # mean of t-statistics
    redoClusters = [i for i in keys(clusterTstats) if clusterTstats[i]<tStatMean] # select clusters which have t-stat lower than mean
    if length(redoClusters) <= 1
        return correlationSorted, clusters, silh
    else
        keysRedo = [j for i in redoClusters for j in clusters[i]] # select keys of redocluster
        correlationTemp = correlation[indexin(keysRedo, assets), indexin(keysRedo, assets)] # slice corr for redoclusters
        assets_ = names(correlationTemp) # names of dataframe
        tStatMean = mean([clusterTstats[i] for i in redoClusters]) # mean of t-stats redoclusters
        # call again clusterKMeansTop
        correlationSorted2, clusters2, silh2 = clusterKMeansTop(correlationTemp, numberClusters = min(numberClusters, size(correlationTemp)[2] - 1), iterations = iterations)
        # Make new outputs, if necessary
        correlationNew, clustersNew, silhNew = makeNewOutputs(correlation, Dict("$i" => clusters[i] for i in keys(clusters) if i ∉ redoClusters), clusters2)
        # mean of t-stats new output
        newTstatMean = mean([mean(silhNew[indexin(clustersNew[i], silhNew.index), :silh])/
                             std(silhNew[indexin(clustersNew[i], silhNew.index), :silh]) 
                             for i in keys(clustersNew)])
        if newTstatMean <= tStatMean
            return correlationSorted, clusters, silh
        else
            return correlationNew, clustersNew, silhNew
        end
    end
end

"""
    function: Compute sub cov matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.3, Page 61
"""
function randomCovarianceSub(numberObservations, # number of observations
                             numberColumns, # number of cols
                             σ, # sigma for normal distribution
                             domain) # range for rand
    # Sub correlation matrix
    if numberColumns == 1
        return ones(1, 1)
    end
    data = rand(domain, Distributions.Normal(), numberObservations) # generate data
    data = repeat(data, 1, numberColumns) # repeat data
    data += rand(domain, Distributions.Normal(0, σ), size(data)) # add data
    covariance = cov(data) # covariance of data
    return covariance
end

"""
    function: Compute random block cov matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.3, Page 61
"""
function randomBlockCovariance(numberColumns, # number of cols
                               numberBlocks; # number of blocks
                               blockSizeMin = 1, # minimum size of block
                               σ = 1., # sigma for normal distribution
                               domain = nothing) # range for rand
    # Generate a block random correlation matrix
    parts = sort(StatsBase.sample(domain,  1:numberColumns - (blockSizeMin - 1)*numberBlocks - 1, numberBlocks - 1, replace = false))
    append!(parts, numberColumns - (blockSizeMin - 1)*numberBlocks)
    parts = append!([parts[1]], diff(parts)) .-1 .+ blockSizeMin
    covariance = nothing
    for column in parts
        thisCovariance = randomCovarianceSub(Int(max(column*(column + 1)/2., 100)), column, σ, domain) # sub covariance
        if isnothing(covariance)
            covariance = copy(thisCovariance) #copy covariance
        else
            covariance = BlockDiagonal([covariance, thisCovariance])  # block diagram covariance matrix
        end
    end
    return covariance
end

"""
    function: Compute random block corr matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.3, Page 61
"""
function randomBlockCorrelation(numberColumns,  # number of cols
                                numberBlocks;  # number of blocks
                                randomState = nothing,  # for rand data
                                blockSizeMin = 1) # minimum size of block
    # set seed
    domain = MersenneTwister(randomState)
    # generate 2 random block diagram cov matrix
    covariance1 = randomBlockCovariance(numberColumns, numberBlocks, blockSizeMin = blockSizeMin, σ = .5, domain = domain)
    covariance2 = randomBlockCovariance(numberColumns, 1, blockSizeMin = blockSizeMin, σ = 1., domain = domain) # add noise
    covariance1 += covariance2 # add 2 cov matrix
    correlation = covToCorr(covariance1) # corr matrix
    correlation = DataFrames.DataFrame(correlation, :auto) # dataframe of corr matrix
    return correlation
end

"""
    function: Derive the correlation matrix from a covariance matrix
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 2.3, Page 27
"""
function covToCorr(covariance) # covariance matrix
    std = sqrt.((diag(covariance))) # standard deviations
    correlation = covariance./(std.*std') # create correlation matrix
    correlation[correlation .< -1] .= -1 # numerical error
    correlation[correlation .> 1] .= 1 # numerical error
    return correlation
end
