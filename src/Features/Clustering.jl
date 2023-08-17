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
using PyCall

Metrics = pyimport("sklearn.metrics")

"""
    calculatePercentChange(prices::DataFrame)::DataFrame

Calculate percent changes of prices.

This function calculates the percent changes of prices for each asset.

# Parameters:
- `prices::DataFrame`: DataFrame containing asset prices.

# Returns:
- `DataFrame`: DataFrame containing percent changes of prices.
"""
function calculatePercentChange(prices::DataFrame)::DataFrame
    returns = DataFrame()
    for symbol in names(prices)[2:end]
        data = prices[!, Symbol(symbol)]
        ret = [NaN; diff(data) ./ data[1:end-1]]
        returns[!, Symbol(symbol)] = ret
    end
    return returns
end

"""
    clusterKMeansBase(
        correlation::DataFrame,
        numberClusters::Int=10,
        iterations::Int=10
    )::Tuple{DataFrame, Dict{String, Vector{Int}}, DataFrame, Vector{Int}}

Cluster assets using KMeans.

This function performs clustering on assets using the KMeans algorithm based on the methodology presented in
De Prado (2020), Advances in Financial Machine Learning, Snippet 4.1, Page 56.

# Parameters:
- `correlation::DataFrame`: Correlation matrix.
- `numberClusters::Int=10`: Number of clusters. Default is 10.
- `iterations::Int=10`: Number of iterations. Default is 10.

# Returns:
- `DataFrame`: Clustered correlation matrix.
- `Dict{String, Vector{Int}}`: Clusters of assets.
- `DataFrame`: Silhouette scores for each asset.
- `Vector{Int}`: Index of sorted correlation.
"""
function clusterKMeansBase(
    correlation::DataFrame,
    numberClusters::Int=10,
    iterations::Int=10
)::Tuple{DataFrame, Dict{String, Vector{Int}}, DataFrame, Vector{Int}}
    distance = sqrt.((1 .- correlation) / 2)
    silh, kmeansOut = NaN, NaN
    for init in 1:iterations
        for i in 2:numberClusters
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
    mergeClusters(
        correlation::DataFrame,
        clusters::Dict{String, Vector{Int}},
        clusters2::Dict{String, Vector{Int}}
    )::Tuple{DataFrame, Dict{String, Vector{Int}}, DataFrame}

Merge two sets of clusters into new clusters.

This function merges two sets of clusters and generates a new clustered correlation matrix.

# Parameters:
- `correlation::DataFrame`: Original correlation matrix.
- `clusters::Dict{String, Vector{Int}}`: First set of clusters.
- `clusters2::Dict{String, Vector{Int}}`: Second set of clusters.

# Returns:
- `DataFrame`: Clustered correlation matrix.
- `Dict{String, Vector{Int}}`: Merged clusters of assets.
- `DataFrame`: Silhouette scores for each asset in the merged clusters.
"""
function mergeClusters(
    correlation::DataFrame,
    clusters::Dict{String, Vector{Int}},
    clusters2::Dict{String, Vector{Int}}
)::Tuple{DataFrame, Dict{String, Vector{Int}}, DataFrame}
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

using Distributions
using DataFrames
using LinearAlgebra
using Clustering
using BlockDiagonals
using StatsBase
using Random
using PyCall

const Metrics = pyimport("sklearn.metrics")

"""
    percentChange(prices::DataFrame) -> DataFrame

Calculate percent changes of prices.

This function calculates the percent changes of prices for each asset.

- `prices` (DataFrame): DataFrame containing asset prices.

Returns: DataFrame containing percent changes of prices.
"""
function percentChange(prices::DataFrame)
    returns = DataFrame()
    for sym in names(prices)[2:end]
        data = prices[!, Symbol(sym)]
        ret = [NaN; diff(data) ./ data[1:end-1]]
        returns[!, Symbol(sym)] = ret
    end
    return returns
end

"""
    clusterKMeansBase(
        correlation::AbstractMatrix,
        numberClusters::Int = 10,
        iterations::Int = 10
    ) -> Tuple{AbstractMatrix, Dict{String, Vector{Int}}, DataFrame, Vector{Int}}

Cluster assets using KMeans.

This function performs clustering on assets using the KMeans algorithm.

- `correlation` (AbstractMatrix): Correlation matrix.
- `numberClusters` (Int, optional): Number of clusters. Default is 10.
- `iterations` (Int, optional): Number of iterations. Default is 10.

Returns:
- Clustered correlation matrix.
- Clusters of assets.
- Silhouette scores for each asset.
- Index of sorted assets.
"""
function clusterKMeansBase(
    correlation::AbstractMatrix,
    numberClusters::Int = 10,
    iterations::Int = 10
)
    distance = sqrt.((1 .- correlation) / 2)
    silh, kmeansOut = NaN, NaN
    for init in 1:iterations
        for i in 2:numberClusters
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
    makeNewOutputs(
        correlation::AbstractMatrix,
        clusters::Dict{String, Vector{Int}},
        clusters2::Dict{String, Vector{Int}}
    ) -> Tuple{AbstractMatrix, Dict{String, Vector{Int}}, DataFrame}

Merge two sets of clusters into new clusters.

This function merges two sets of clusters and generates a new clustered correlation matrix.

- `correlation` (AbstractMatrix): Original correlation matrix.
- `clusters` (Dict): First set of clusters.
- `clusters2` (Dict): Second set of clusters.

Returns:
- Clustered correlation matrix.
- Merged clusters of assets.
- Silhouette scores for each asset in the merged clusters.
"""
function makeNewOutputs(
    correlation::AbstractMatrix,
    clusters::Dict{String, Vector{Int}},
    clusters2::Dict{String, Vector{Int}}
)
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
    clusterKMeansTop(
        correlation::AbstractMatrix,
        numberClusters::Int = nothing,
        iterations::Int = 10
    ) -> Tuple{AbstractMatrix, Dict{String, Vector{Int}}, DataFrame}

Perform clustering using KMeans and ONC.

This function performs clustering using the KMeans algorithm and the ONC (Organized Nearest Clusters) method.

- `correlation` (AbstractMatrix): Correlation matrix.
- `numberClusters` (Int, optional): Number of clusters. Default is determined automatically.
- `iterations` (Int, optional): Number of iterations. Default is 10.

Returns:
- Clustered correlation matrix.
- Clusters of assets.
- Silhouette scores for each asset.
"""
function clusterKMeansTop(
    correlation::AbstractMatrix,
    numberClusters::Int = nothing,
    iterations::Int = 10
)
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

using Random
using LinearAlgebra
using DataFrames

"""
    randomBlockCorrelation(
        numberColumns::Int,
        numberBlocks::Int;
        randomState::Union{Int, Nothing} = nothing,
        blockSizeMin::Int = 1
    ) -> DataFrame

Generate a random block correlation matrix.

This function generates a random block correlation matrix with specified parameters.

- `numberColumns` (Int): Number of columns.
- `numberBlocks` (Int): Number of blocks.
- `randomState` (Union{Int, Nothing}, optional): Seed for random data generation. Default is `nothing`.
- `blockSizeMin` (Int, optional): Minimum size of a block. Default is 1.

Returns:
- DataFrame: Random block correlation matrix.
"""
function randomBlockCorrelation(
    numberColumns::Int,
    numberBlocks::Int;
    randomState::Union{Int, Nothing} = nothing,
    blockSizeMin::Int = 1
)
    domain = MersenneTwister(randomState)
    covariance1 = randomBlockCovariance(numberColumns, numberBlocks, blockSizeMin = blockSizeMin, σ = 0.5, domain = domain)
    covariance2 = randomBlockCovariance(numberColumns, 1, blockSizeMin = blockSizeMin, σ = 1.0, domain = domain)
    covariance1 += covariance2
    correlation = covToCorr(covariance1)
    correlation = DataFrame(correlation, :auto)
    return correlation
end

"""
    covToCorr(covariance::AbstractMatrix) -> AbstractMatrix

Convert a covariance matrix to a correlation matrix.

This function converts a covariance matrix into a correlation matrix.

- `covariance` (Matrix): Covariance matrix.

Returns:
- Matrix: Correlation matrix.
"""
function covToCorr(covariance::AbstractMatrix)
    stdDeviations = sqrt.(diag(covariance))
    correlation = covariance ./ (stdDeviations * stdDeviations')
    correlation[correlation .< -1] .= -1
    correlation[correlation .> 1] .= 1
    return correlation
end
