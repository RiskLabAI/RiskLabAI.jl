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
Calculate percent changes of prices.

This function calculates the percent changes of prices for each asset.

Parameters:
- prices (DataFrame): DataFrame containing asset prices.

Returns:
- DataFrame: DataFrame containing percent changes of prices.
"""
function percentChange(prices::DataFrame)
    returns = DataFrame()
    for sym in names(prices)[2:end]
        data = prices[!, Symbol(sym)]
        ret = [NaN; diff(data) ./ data[1:end-1]]
        push!(returns, Symbol(sym) => ret)
    end
    return returns
end

"""
Cluster assets using KMeans.

This function performs clustering on assets using the KMeans algorithm based on the methodology presented in
De Prado (2020), Advances in Financial Machine Learning, Snippet 4.1, Page 56.

Parameters:
- correlation (DataFrame): Correlation matrix.
- numberClusters (Int, optional): Number of clusters. Default is 10.
- iterations (Int, optional): Number of iterations. Default is 10.

Returns:
- DataFrame: Clustered correlation matrix.
- Dict: Clusters of assets.
- DataFrame: Silhouette scores for each asset.
"""
function clusterKMeansBase(correlation, numberClusters = 10, iterations = 10)
    distance = sqrt.((1 .- correlation) / 2)
    silh, kmeansOut = NaN, NaN
    for init ∈ 1:iterations
        for i ∈ 2:numberClusters
            kmeans_ = kmeans(distance, i)
            silh_ = Metrics.silhouette_samples(distance, assignments(kmeans_))
            statistic = (mean(silh_) / std(silh_), mean(silh) / std(silh))
            if isnan(statistic[2]) || statistic[1] > statistic[2]
                silh, kmeansOut = silh_, kmeans_
            end
        end
    end
    indexSorted = sortperm(assignments(kmeansOut))
    correlationSorted = correlation[indexSorted, indexSorted]
    clusters = Dict("$i" => filter(p -> assignments(kmeansOut)[p] == i, indexSorted) for i in unique(assignments(kmeansOut)))
    silh = DataFrame(silh = silh)
    return correlationSorted, clusters, silh, indexSorted
end

"""
Merge two sets of clusters into new clusters.

This function merges two sets of clusters and generates a new clustered correlation matrix.

Parameters:
- correlation (DataFrame): Original correlation matrix.
- clusters (Dict): First set of clusters.
- clusters2 (Dict): Second set of clusters.

Returns:
- DataFrame: Clustered correlation matrix.
- Dict: Merged clusters of assets.
- DataFrame: Silhouette scores for each asset in the merged clusters.
"""
function makeNewOutputs(correlation, clusters, clusters2)
    assets = names(correlation)
    clustersNew = Dict()
    for i in keys(clusters)
        clustersNew[length(keys(clustersNew)) + 1] = clusters[i]
    end
    for i in keys(clusters2)
        clustersNew[length(keys(clustersNew)) + 1] = clusters2[i]
    end
    indexNew = [j for i in keys(clustersNew) for j in clustersNew[i]]
    correlationNew = correlation[indexin(indexNew, assets), indexin(indexNew, assets)]
    distance = sqrt.((1 .- Matrix(correlation)) / 2)
    labelsKmeans = zeros(size(distance)[2])
    for i in keys(clustersNew)
        index = indexin(clustersNew[i], assets)
        labelsKmeans[index] .= i
    end
    silhNew = DataFrame(index = assets, silh = Metrics.silhouette_samples(distance, labelsKmeans))
    return correlationNew, clustersNew, silhNew
end

"""
Perform clustering using KMeans and ONC.

This function performs clustering using the KMeans algorithm and the ONC (Organized Nearest Clusters) method based on the methodology presented in
De Prado (2020), Advances in Financial Machine Learning, Snippet 4.2, Page 58.

Parameters:
- correlation (DataFrame): Correlation matrix.
- numberClusters (Int, optional): Number of clusters. Default is determined automatically.
- iterations (Int, optional): Number of iterations. Default is 10.

Returns:
- DataFrame: Clustered correlation matrix.
- Dict: Clusters of assets.
- DataFrame: Silhouette scores for each asset.
"""
function clusterKMeansTop(correlation, numberClusters = nothing, iterations = 10)
    if isnothing(numberClusters)
        numberClusters = size(correlation)[2] - 1
    end
    assets = names(correlation)
    correlationSorted, clusters, silh, indexSorted = clusterKMeansBase(Matrix(correlation),
        numberClusters = min(numberClusters, size(correlation)[2] - 1), iterations = 10)
    correlationSorted = DataFrame(correlationSorted, :auto)
    DataFrames.rename!(correlationSorted, Symbol.(names(correlationSorted)) .=> assets[indexSorted])
    clusters = Dict("$i" => assets[clusters[i]] for i in keys(clusters))
    clusterTstats = Dict("$i" => mean(silh[indexin(clusters[i], assets), :silh]) / std(silh[indexin(clusters[i], assets), :silh]) for i in keys(clusters))
    tStatMean = sum(values(clusterTstats)) / length(clusterTstats)
    redoClusters = [i for i in keys(clusterTstats) if clusterTstats[i] < tStatMean]
    if length(redoClusters) <= 1
        return correlationSorted, clusters, silh
    else
        keysRedo = [j for i in redoClusters for j in clusters[i]]
        correlationTemp = correlation[indexin(keysRedo, assets), indexin(keysRedo, assets)]
        assets_ = names(correlationTemp)
        tStatMean = mean([clusterTstats[i] for i in redoClusters])
        correlationSorted2, clusters2, silh2 = clusterKMeansTop(correlationTemp, numberClusters = min(numberClusters, size(correlationTemp)[2] - 1), iterations = iterations)
        correlationNew, clustersNew, silhNew = makeNewOutputs(correlation, Dict("$i" => clusters[i] for i in keys(clusters) if i ∉ redoClusters), clusters2)
        newTstatMean = mean([mean(silhNew[indexin(clustersNew[i], silhNew.index), :silh]) /
            std(silhNew[indexin(clustersNew[i], silhNew.index), :silh]) for i in keys(clustersNew)])
        if newTstatMean <= tStatMean
            return correlationSorted, clusters, silh
        else
            return correlationNew, clustersNew, silhNew
        end
    end
end

"""
Compute a random covariance submatrix.

This function computes a random covariance submatrix with specified parameters.

Parameters:
- numberObservations (Int): Number of observations.
- numberColumns (Int): Number of columns.
- σ (Float64): Sigma value for normal distribution.
- domain (StepRangeLen, optional): Range for random data generation.

Returns:
- Matrix: Random covariance submatrix.
"""
function randomCovarianceSub(numberObservations, numberColumns, σ, domain)
    if numberColumns == 1
        return ones(1, 1)
    end
    data = rand(domain, Distributions.Normal(), numberObservations)
    data = repeat(data, 1, numberColumns)
    data += rand(domain, Distributions.Normal(0, σ), size(data))
    covariance = cov(data)
    return covariance
end

"""
Compute a random block covariance matrix.

This function generates a random block covariance matrix with specified parameters.

Parameters:
- numberColumns (Int): Number of columns.
- numberBlocks (Int): Number of blocks.
- blockSizeMin (Int, optional): Minimum size of a block. Default is 1.
- σ (Float64, optional): Sigma value for normal distribution. Default is 1.
- domain (StepRangeLen, optional): Range for random data generation.

Returns:
- BlockArray: Random block covariance matrix.
"""
function randomBlockCovariance(numberColumns, numberBlocks; blockSizeMin = 1, σ = 1.0, domain = nothing)
    parts = sort(sample(domain, 1:numberColumns - (blockSizeMin - 1) * numberBlocks - 1, numberBlocks - 1, replace = false))
    append!(parts, numberColumns - (blockSizeMin - 1) * numberBlocks)
    parts = append!([parts[1]], diff(parts)) .- 1 .+ blockSizeMin
    covariance = nothing
    for column in parts
        thisCovariance = randomCovarianceSub(Int(max(column * (column + 1) / 2.0, 100)), column, σ, domain)
        if isnothing(covariance)
            covariance = copy(thisCovariance)
        else
            covariance = BlockDiagonal([covariance, thisCovariance])
        end
    end
    return covariance
end

"""
Compute a random block correlation matrix.

This function generates a random block correlation matrix with specified parameters.

Parameters:
- numberColumns (Int): Number of columns.
- numberBlocks (Int): Number of blocks.
- randomState (Int, optional): Seed for random data generation.
- blockSizeMin (Int, optional): Minimum size of a block. Default is 1.

Returns:
- DataFrame: Random block correlation matrix.
"""
function randomBlockCorrelation(numberColumns, numberBlocks; randomState = nothing, blockSizeMin = 1)
    domain = MersenneTwister(randomState)
    covariance1 = randomBlockCovariance(numberColumns, numberBlocks, blockSizeMin = blockSizeMin, σ = 0.5, domain = domain)
    covariance2 = randomBlockCovariance(numberColumns, 1, blockSizeMin = blockSizeMin, σ = 1.0, domain = domain)
    covariance1 += covariance2
    correlation = covToCorr(covariance1)
    correlation = DataFrame(correlation, :auto)
    return correlation
end

"""
Derive the correlation matrix from a covariance matrix.

This function converts a covariance matrix into a correlation matrix.

Parameters:
- covariance (Matrix): Covariance matrix.

Returns:
- Matrix: Correlation matrix.
"""
function covToCorr(covariance)
    std = sqrt.(diag(covariance))
    correlation = covariance ./ (std .* std')
    correlation[correlation .< -1] .= -1
    correlation[correlation .> 1] .= 1
    return correlation
end
