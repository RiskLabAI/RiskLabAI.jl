"""
Calculate returns from prices DataFrame.

Reference: n/a
Methodology: n/a
"""
function calculate_returns(prices::DataFrames.DataFrame)
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
function cluster_kmeans_base(correlation; number_clusters = 10, iterations = 10)
    distance = sqrt.((1 .- correlation) / 2)
    silh, kmeans_out = [NaN], [NaN]
    for init in 1:iterations
        for i in 2:number_clusters
            kmeans_ = kmeans(distance, i)
            silh_ = silhouette_samples(distance, assignments(kmeans_))
            statistic = (mean(silh_) / std(silh_), mean(silh) / std(silh))
            if isnan(statistic[2]) || statistic[1] > statistic[2]
                silh, kmeans_out = silh_, kmeans_
            end
        end
    end
    index_new = sortperm(assignments(kmeans_out))
    correlation_new = correlation[index_new, index_new]
    clusters = Dict("$i"=> filter(p -> assignments(kmeans_out)[p] == i, index_new) for i in unique(assignments(kmeans_out)))
    silh = DataFrames.DataFrame(silh = silh)
    return correlation_new, clusters, silh, index_new
end

"""
Optimal portfolio construction using NCO algorithm.

Reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons. Snippet 7.6, Page 100
"""
function opt_port_nco(covariance; μ = nothing, number_clusters = nothing)
    correlation = cov_to_corr(covariance)
    if isnothing(number_clusters)
        number_clusters = Int(size(correlation)[1] / 2)
    end
    correlation, clusters, _, _ = cluster_kmeans_base(correlation, number_clusters = number_clusters, iterations = 10)
    ω_intra_cluster = DataFrames.DataFrame(repeat([0.], size(covariance)[1], length(keys(clusters))), :auto)
    DataFrames.rename!(ω_intra_cluster, Symbol.(names(ω_intra_cluster)) .=> Symbol.(keys(clusters)))
    for i in keys(clusters)
        covariance_intra_cluster = covariance[clusters[i], clusters[i]]
        if isnothing(μ)
            μ_intra_cluster = nothing
        else
            μ_intra_cluster = μ[clusters[i]]
        end
        ω_intra_cluster[clusters[i], i] .= opt_port(covariance_intra_cluster, μ_intra_cluster)
    end
    covariance_inter_cluster = transpose(Matrix(ω_intra_cluster)) * (covariance * Matrix(ω_intra_cluster))
    if isnothing(μ)
        μ_inter_cluster = nothing
    else
        μ_inter_cluster = transpose(Matrix(ω_intra_cluster)) * μ
    end
    ω_inter_cluster = DataFrames.DataFrame(weight = vec(opt_port(covariance_inter_cluster, μ_inter_cluster)))
    ω_nco = DataFrames.DataFrame(weights = vec(sum(Matrix(ω_intra_cluster) .* Matrix(ω_inter_cluster)', dims = 2)))
    return ω_nco
end
