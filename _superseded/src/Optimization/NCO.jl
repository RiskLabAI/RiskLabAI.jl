using DataFrames
using Statistics
using Clustering
using LinearAlgebra

"""
    calculateReturns(prices::DataFrame)

Calculate returns from prices DataFrame.

# Arguments
- `prices::DataFrame`: A DataFrame with price data.

# Returns
- `returns::DataFrame`: A DataFrame with return data.

"""
function calculateReturns(prices::DataFrame)
    returns = DataFrame()

    for sym in names(prices)[2:end]
        data = prices[!, Symbol(sym)]
        ret = diff(data) ./ data[1:end-1]
        returns[!, Symbol(sym)] = vcat(NaN, ret)
    end

    return returns
end

"""
    clusterKMeansBase(correlation::Matrix; numberClusters=10, iterations=10)

Perform clustering using K-Means.

# Arguments
- `correlation::Matrix`: A correlation matrix.
- `numberClusters::Int=10`: Number of clusters.
- `iterations::Int=10`: Number of iterations.

# Returns
- `correlationNew::Matrix`: A re-ordered correlation matrix.
- `clusters::Dict`: Cluster assignment for each observation.
- `silh::DataFrame`: Silhouette scores for each observation.
- `indexNew::Array{Int}`: The order of observations based on cluster assignment.

"""
function clusterKMeansBase(correlation::Matrix; numberClusters=10, iterations=10)
    distance = sqrt.((1 .- correlation) / 2)
    silh, kmeansOut = [NaN], [NaN]

    for init in 1:iterations
        for i in 2:numberClusters
            kmeans = kmeans(distance, i)
            silhSample = silhouette_samples(distance, assignments(kmeans))
            statistic = (mean(silhSample) / std(silhSample), mean(silh) / std(silh))

            if isnan(statistic[2]) || statistic[1] > statistic[2]
                silh, kmeansOut = silhSample, kmeans
            end
        end
    end

    indexNew = sortperm(assignments(kmeansOut))
    correlationNew = correlation[indexNew, indexNew]
    clusters = Dict("$i" => filter(p -> assignments(kmeansOut)[p] == i, indexNew) for i in unique(assignments(kmeansOut)))
    silh = DataFrame(silh = silh)

    return correlationNew, clusters, silh, indexNew
end

"""
    optPortNCO(covariance::Matrix; μ=nothing, numberClusters=nothing)

Optimal portfolio construction using NCO algorithm.

# Arguments
- `covariance::Matrix`: A covariance matrix.
- `μ::Vector=Vector{Float64}()`: Expected returns vector.
- `numberClusters::Int=Nothing`: Number of clusters.

# Returns
- `ωNCO::DataFrame`: Portfolio weights using the NCO algorithm.

"""
function optPortNCO(covariance::Matrix; μ=nothing, numberClusters=nothing)
    function covToCorr(cov::Matrix)
        diagInv = diagm(1 ./ sqrt.(diag(cov)))
        return diagInv * cov * diagInv
    end

    function optPort(cov::Matrix, μ=nothing)
        # Add your optimal portfolio calculation code here
        return [0.0, 0.0, 0.0]  # Replace this line with the actual calculation
    end

    correlation = covToCorr(covariance)

    if isnothing(numberClusters)
        numberClusters = Int(size(correlation, 1) / 2)
    end

    correlation, clusters, _, _ = clusterKMeansBase(correlation, numberClusters=numberClusters, iterations=10)
    ωIntraCluster = DataFrame(repeat([0.0], size(covariance, 1), length(keys(clusters))), :auto)
    rename!(ωIntraCluster, Symbol.(names(ωIntraCluster)) .=> Symbol.(keys(clusters)))

    for i in keys(clusters)
        covarianceIntraCluster = covariance[clusters[i], clusters[i]]
        μIntraCluster = isnothing(μ) ? nothing : μ[clusters[i]]
        ωIntraCluster[clusters[i], i] .= optPort(covarianceIntraCluster, μIntraCluster)
    end

    covarianceInterCluster = transpose(Matrix(ωIntraCluster)) * (covariance * Matrix(ωIntraCluster))
    μInterCluster = isnothing(μ) ? nothing : transpose(Matrix(ωIntraCluster)) * μ
    ωInterCluster = DataFrame(weight = vec(optPort(covarianceInterCluster, μInterCluster)))
    ωNCO = DataFrame(weights = vec(sum(Matrix(ωIntraCluster) .* Matrix(ωInterCluster)', dims=2)))

    return ωNCO
end
