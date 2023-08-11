using Distributions
using Statistics
using DataFrames
using GLM
using LinearAlgebra
using PlotlyJS

"""
Function to calculate returns.

This function calculates the percentage change in prices.

Args:
    prices::DataFrame: DataFrame containing price data for various assets.

Returns:
    DataFrame: DataFrame containing calculated returns for each asset.
"""
function percentChange(prices::DataFrame)
    returns = DataFrame() # Empty DataFrame for returns
    for sym in names(prices)[2:end]
        data = prices[!, Symbol(sym)] # Prices of each asset
        ret = Array{Float64}(undef, length(data)) # Returns of each asset
        ret[1] = NaN
        for i in 2:length(data)
            ret[i] = (data[i] / data[i - 1]) - 1 # Calculate returns of each asset
        end
        returns[!, Symbol(sym)] = ret # Append returns to DataFrame
    end
    return returns
end

"""
Function to perform clustering.

This function implements the clustering methodology from De Prado's book "Advances in Financial Machine Learning" (Snippet 4.1, Page 56).

Args:
    correlation::DataFrame: Correlation matrix.
    numberClusters::Int: Number of clusters.
    iterations::Int: Number of iterations.

Returns:
    Tuple{DataFrame, Dict, DataFrame}: Resulting correlation matrix, clusters, and silhouette scores.
"""
function clusterKMeansBase(
        correlation; 
        numberClusters = 10, 
        iterations = 10
    )
    distance = sqrt.((1 .- correlation) / 2) # Distance matrix
    silh, kmeansOut = [NaN], [NaN] # Initial values for silhouette and kmeans
    for init ∈ 1:iterations
        for i ∈ 2:numberClusters
            kmeans_ = kmeans(distance, i) # Cluster distances with maximum cluster size i
            silh_ = silhouette_samples(distance, assignments(kmeans_)) # Silhouette score of clustering
            statistic = (mean(silh_) / std(silh_), mean(silh) / std(silh)) # Calculate t-statistic
            if isnan(statistic[2]) || statistic[1] > statistic[2]
                silh, kmeansOut = silh_, kmeans_ # Replace with better clustering
            end
        end
    end
    indexSorted = sortperm(assignments(kmeansOut)) # Sort arguments based on clustering
    correlationSorted = correlation[indexSorted, indexSorted] # New correlation matrix based on clustering
    # Dictionary of clusters
    clusters = Dict("$i" => filter(p -> assignments(kmeansOut)[p] == i, indexSorted) for i in unique(assignments(kmeansOut)))
    silh = DataFrame(silh = silh) # DataFrame of silhouette scores
    return correlationSorted, clusters, silh, indexSorted
end

"""
Function to make new clustering.

This function combines two sets of clusters into a new set of clusters.

Args:
    correlation::DataFrame: Original correlation matrix.
    clusters::Dict: First set of clusters.
    clusters2::Dict: Second set of clusters.

Returns:
    Tuple{DataFrame, Dict, DataFrame}: Resulting correlation matrix, combined clusters, and silhouette scores.
"""
function makeNewOutputs(
        correlation,
        clusters,
        clusters2
    )
    assets = names(correlation) # Names of the columns in the correlation matrix
    # Merge two sets of clusters
    clustersNew = Dict()
    for i in keys(clusters)
        clustersNew[length(keys(clustersNew)) + 1] = clusters[i] 
    end
    for i in keys(clusters2)
        clustersNew[length(keys(clustersNew)) + 1] = clusters2[i]
    end
    indexNew = [j for i in keys(clustersNew) for j in clustersNew[i]] # Sorted index of assets
    correlationNew = correlation[indexin(indexNew, assets), indexin(indexNew, assets)] # New correlation matrix
    distance = sqrt.((1 .- Matrix(correlation)) / 2) # Distance matrix
    labelsKmeans = zeros(size(distance)[2]) # Initial labels
    for i in keys(clustersNew)
        index = indexin(clustersNew[i], assets) 
        labelsKmeans[index] .= i # Labels for clusters
    end
    silhNew = DataFrame(index = assets, silh = silhouette_samples(distance, labelsKmeans)) # Silhouette series
    return correlationNew, clustersNew, silhNew
end

"""
Function to perform clustering (ONC).

This function refines clustering results using the ONC methodology.

Args:
    correlation::DataFrame: Correlation matrix.
    numberClusters::Int: Number of clusters.
    iterations::Int: Number of iterations.

Returns:
    Tuple{DataFrame, Dict, DataFrame}: Resulting correlation matrix, clusters, and silhouette scores.
"""
function clusterKMeansTop(
        correlation; 
        numberClusters = nothing, 
        iterations = 10
    )
    if isnothing(numberClusters)
        numberClusters = size(correlation)[2] - 1 # Set number of clusters
    end
    assets = names(correlation) # Names of columns
    # Clustering
    correlationSorted, clusters, silh, indexSorted = clusterKMeansBase(Matrix(correlation), numberClusters = min(numberClusters, size(correlation)[2] - 1), iterations = 10)
    correlationSorted = DataFrame(correlationSorted, :auto) # DataFrame of sorted correlation matrix
    rename!(correlationSorted, Symbol.(names(correlationSorted)) .=> assets[indexSorted]) # Rename columns of sorted correlation matrix
    clusters = Dict("$i" => assets[clusters[i]] for i in keys(clusters)) # Dictionary of clusters
    # Calculate t-statistic for each cluster
    clusterTstats = Dict("$i" => mean(silh[indexin(clusters[i], assets), :silh]) / std(silh[indexin(clusters[i], assets), :silh]) for i in keys(clusters))
    tStatMean = sum(values(clusterTstats)) / length(clusterTstats) # Mean of t-statistics
    redoClusters = [i for i in keys(clusterTstats) if clusterTstats[i] < tStatMean] # Select clusters with t-statistics lower than mean
    if length(redoClusters) <= 1
        return correlationSorted, clusters, silh
    else
        keysRedo = [j for i in redoClusters for j in clusters[i]] # Select keys of redo clusters
        correlationTemp = correlation[indexin(keysRedo, assets), indexin(keysRedo, assets)] # Slice correlation for redo clusters
        assets_ = names(correlationTemp) # Names of DataFrame
        tStatMean = mean([clusterTstats[i] for i in redoClusters]) # Mean of t-stats for redo clusters
        # Call clusterKMeansTop again
        correlationSorted2, clusters2, silh2 = clusterKMeansTop(correlationTemp, numberClusters = min(numberClusters, size(correlationTemp)[2] - 1), iterations = iterations)
        # Make new outputs if necessary
        correlationNew, clustersNew, silhNew = makeNewOutputs(correlation, Dict("$i" => clusters[i] for i in keys(clusters) if i ∉ redoClusters), clusters2)
        # Mean of t-stats for new output
        newTstatMean = mean([mean(silhNew[indexin(clustersNew[i], silhNew.index), :silh]) /
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
Function to compute a sub covariance matrix.

This function generates a sub covariance matrix.

Args:
    numberObservations::Int: Number of observations.
    numberColumns::Int: Number of columns.
    σ::Float64: Standard deviation.
    domain: Range for random data.

Returns:
    Matrix{Float64}: Sub covariance matrix.
"""
function randomCovarianceSub(
        numberObservations, 
        numberColumns,
        σ,
        domain
    )
    # Sub covariance matrix
    if numberColumns == 1
        return ones(1, 1)
    end
    data = rand(domain, Distributions.Normal(), numberObservations) # Generate data
    data = repeat(data, 1, numberColumns) # Repeat data
    data += rand(domain, Distributions.Normal(0, σ), size(data)) # Add noise
    covariance = cov(data) # Covariance of data
    return covariance
end

"""
Function to compute a random block covariance matrix.

This function generates a random block covariance matrix.

Args:
    numberColumns::Int: Number of columns.
    numberBlocks::Int: Number of blocks.
    blockSizeMin::Int: Minimum block size.
    σ::Float64: Standard deviation.
    domain: Range for random data.

Returns:
    AbstractMatrix: Random block covariance matrix.
"""
function randomBlockCovariance(
        numberColumns,
        numberBlocks;
        blockSizeMin = 1,
        σ = 1.0,
        domain = nothing
    )
    # Generate a block random covariance matrix
    parts = sort(StatsBase.sample(domain, 1:numberColumns - (blockSizeMin - 1) * numberBlocks - 1, numberBlocks - 1, replace = false))
    append!(parts, numberColumns - (blockSizeMin - 1) * numberBlocks)
    parts = append!([parts[1]], diff(parts)) .- 1 .+ blockSizeMin
    covariance = nothing
    for column in parts
        thisCovariance = randomCovarianceSub(Int(max(column * (column + 1) / 2.0, 100)), column, σ, domain) # Sub covariance
        if isnothing(covariance)
            covariance = copy(thisCovariance) # Copy covariance
        else
            covariance = BlockDiagonal([covariance, thisCovariance])  # Block diagonal covariance matrix
        end
    end
    return covariance
end

"""
Function to compute a random block correlation matrix.

This function generates a random block correlation matrix.

Args:
    numberColumns::Int: Number of columns.
    numberBlocks::Int: Number of blocks.
    randomState: Random seed.
    blockSizeMin::Int: Minimum block size.

Returns:
    DataFrame: Random block correlation matrix.
"""
function randomBlockCorrelation(
        numberColumns,
        numberBlocks,
        randomState = nothing,
        blockSizeMin = 1
    )
    # Set seed
    domain = MersenneTwister(randomState)
    # Generate two random block diagonal covariance matrices
    covariance1 = randomBlockCovariance(numberColumns, numberBlocks, blockSizeMin = blockSizeMin, σ = 0.5, domain = domain)
    covariance2 = randomBlockCovariance(numberColumns, 1, blockSizeMin = blockSizeMin, σ = 1.0, domain = domain) # Add noise
    covariance1 += covariance2 # Add two covariance matrices
    correlation = covToCorr(covariance1) # Correlation matrix
    correlation = DataFrame(correlation, :auto) # DataFrame of correlation matrix
    return correlation
end

"""
Function to derive the correlation matrix from a covariance matrix.

This function calculates the correlation matrix from a given covariance matrix.

Args:
    covariance::Matrix: Covariance matrix.

Returns:
    Matrix{Float64}: Correlation matrix.
"""
function covToCorr(covariance)
    std = sqrt.((diag(covariance))) # Standard deviations
    correlation = covariance ./ (std .* std') # Create correlation matrix
    correlation[correlation .< -1] .= -1 # Handle numerical errors
    correlation[correlation .> 1] .= 1  # Handle numerical errors
    return correlation
end
