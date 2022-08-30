"""----------------------------------------------------------------------
    function: Calculate returns
    reference: n/a
    methodology: n/a
----------------------------------------------------------------------"""
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

"""----------------------------------------------------------------------
    function: Clustering
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 4.1, Page 56
----------------------------------------------------------------------"""
function clusterKMeansBase(correlation; # corr matrix
                           numberClusters = 10, # maximum number of clusters
                           iterations = 10) # iterations
    distance = sqrt.((1 .- correlation)/2) # distance matrix
    silh, kmeansOut = [NaN], [NaN] # initial value for silh, kmeans
    for init ∈ 1:iterations
        for i ∈ 2:numberClusters
            kmeans_ = kmeans(distance, i) # clustering distance with maximum cluster i
            silh_ = silhouette_samples(distance, assignments(kmeans_)) # silh score of clustering
            statistic = (mean(silh_)/std(silh_), mean(silh)/std(silh)) # calculate t-statistic
            if isnan(statistic[2]) || statistic[1]>statistic[2]
                silh, kmeansOut = silh_, kmeans_ # replace better clustering
            end
        end
    end
    indexNew = sortperm(assignments(kmeansOut)) # sort arguments based on clustering
    correlationNew = correlation[indexNew, indexNew] # new corr matrix based on clustering
    # dictionary of clustering
    clusters = Dict("$i"=> filter(p->assignments(kmeansOut)[p] == i, indexNew) for i in unique(assignments(kmeansOut)))
    silh = DataFrames.DataFrame(silh = silh) # dataframe of silh scores
    return correlationNew, clusters, silh, indexNew
end

"""----------------------------------------------------------------------
    function: NCO algorithm
    reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
    methodology: Snipet 7.6, Page 100
----------------------------------------------------------------------"""
function optPortNCO(covariance; # covariance matrix
                    μ = nothing, # mean vector
                    numberClusters = nothing) # maximum number of clusters
    correlation = covToCorr(covariance) # correlation matrix
    if isnothing(numberClusters)
        numberClusters = Int(size(correlation)[1]/2) # set maximum number of clusters
    end
    # clustering
    correlation, clusters, _, _ = clusterKMeansBase(correlation, numberClusters = numberClusters, iterations = 10)
    ωIntraCluster = DataFrames.DataFrame(repeat([0.], size(covariance)[1], length(keys(clusters))), :auto) # dataframe of weights of intraclustering
    DataFrames.rename!(ωIntraCluster, Symbol.(names(ωIntraCluster)) .=> Symbol.(keys(clusters))) # rename dataframe
    for i in keys(clusters)
        covarianceIntraCluster = covariance[clusters[i], clusters[i]] # slice cov matrix
        if isnothing(μ)
            μIntraCluster = nothing # set mu to nothing
        else
            μIntraCluster = μ[clusters[i]] # slice mu 
        end
        ωIntraCluster[clusters[i], i] .= optPort(covarianceIntraCluster, μIntraCluster) # calculate weights of intraclustering
    end
    covarianceInterCluster = transpose(Matrix(ωIntraCluster))*(covariance*Matrix(ωIntraCluster)) # reduced covariance matrix
    if isnothing(μ)
        μInterCluster = nothing # set mu to nothing
    else
        μInterCluster = transpose(Matrix(ωIntraCluster))*μ # set mu for clusters
    end
    ωInterCluster = DataFrames.DataFrame(weight = vec(optPort(covarianceInterCluster, μInterCluster))) # dataframe of weights of interclustering
    ωNCO = DataFrames.DataFrame(weights = vec(sum(Matrix(ωIntraCluster).*Matrix(ωInterCluster)', dims = 2))) # calculate weights
    return ωNCO
end


