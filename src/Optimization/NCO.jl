"""
Calculate returns from prices DataFrame.

Reference: n/a
Methodology: n/a
"""
function calculateReturns(
        prices::DataFrames.DataFrame
    )

    returns = DataFrames.DataFrame()
    for sym in names(prices)[2:end]
        data = prices[!, Symbol(sym)]
        ret = Array{Float64}(undef, length(data))
        ret[1] = NaN
        for i in 2:length(data)
            ret[i] = (data[i] / data[i - 1]) - 1
        end
        returns[!, Symbol(sym)] = ret
    end
    return returns
end

"""
Perform clustering using K-Means.

Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons. Snippet 4.1, Page 56
"""
function clusterKMeansBase(
        correlation;
        numberClusters = 10,
        iterations = 10
    )

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
    clusters = Dict("$i"=> filter(p -> assignments(kmeansOut)[p] == i, indexNew) for i in unique(assignments(kmeansOut)))
    silh = DataFrames.DataFrame(silh = silh)
    return correlationNew, clusters, silh, indexNew
end

"""
Optimal portfolio construction using NCO algorithm.

Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons. Snippet 7.6, Page 100
"""
function optPortNCO(
        covariance;
        μ = nothing,
        numberClusters = nothing
    )
        
    correlation = covToCorr(covariance)
    if isnothing(numberClusters)
        numberClusters = Int(size(correlation)[1] / 2)
    end
    correlation, clusters, _, _ = clusterKMeansBase(correlation, numberClusters = numberClusters, iterations = 10)
    ωIntraCluster = DataFrames.DataFrame(repeat([0.], size(covariance)[1], length(keys(clusters))), :auto)
    DataFrames.rename!(ωIntraCluster, Symbol.(names(ωIntraCluster)) .=> Symbol.(keys(clusters)))
    for i in keys(clusters)
        covarianceIntraCluster = covariance[clusters[i], clusters[i]]
        if isnothing(μ)
            μIntraCluster = nothing
        else
            μIntraCluster = μ[clusters[i]]
        end
        ωIntraCluster[clusters[i], i] .= optPort(covarianceIntraCluster, μIntraCluster)
    end
    covarianceInterCluster = transpose(Matrix(ωIntraCluster)) * (covariance * Matrix(ωIntraCluster))
    if isnothing(μ)
        μInterCluster = nothing
    else
        μInterCluster = transpose(Matrix(ωIntraCluster)) * μ
    end
    ωInterCluster = DataFrames.DataFrame(weight = vec(optPort(covarianceInterCluster, μInterCluster)))
    ωNCO = DataFrames.DataFrame(weights = vec(sum(Matrix(ωIntraCluster) .* Matrix(ωInterCluster)', dims = 2)))
    return ωNCO
end
