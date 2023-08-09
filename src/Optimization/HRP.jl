"""
Generate random data for MC simulation.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.4, Page 241
"""
function generateRandomData(numberObservations, size0, size1, sigma1)
    data1 = rand(Normal(0, 1), numberObservations, size0)
    columns = rand(1:size0, size1)
    data2 = data1[:, columns] + rand(Normal(0, sigma1), numberObservations, length(columns))
    data = hcat(data1, data2)
    dataframe = DataFrames.DataFrame(data, :auto)
    DataFrames.rename!(dataframe, Symbol.(names(dataframe)) .=> Symbol.(collect(1:size(data)[2])))
    return dataframe, columns
end

"""
Compute distance from correlation matrix.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.4, Page 241
"""
function distanceCorr(correlation)
    distance = (1 .- correlation)
    distance = distance.^0.5
    return distance
end

"""
Quasi-diagonalization of a linkage matrix.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.2, Page 229
"""
function quasiDiagonal(link)
    link = Int.(floor.(link))
    sortedItems = DataFrames.DataFrame(index = [1, 2], value = [link[end,1], link[end, 2]])
    numberItems = link[end, 4]
    while maximum(sortedItems.value) >= numberItems 
        sortedItems.index = range(0, stop = size(sortedItems)[1]*2 - 1, step = 2)
        dataframe = sortedItems[sortedItems.value .>= numberItems, :]
        index = dataframe.index
        value = dataframe.value .- numberItems
        sortedItems[in.(sortedItems.index, (index,)), :value] = link[value .+ 1,1]
        dataframe = DataFrames.DataFrame(index = index .+ 1, value = link[value .+ 1, 2])
        sortedItems = vcat(sortedItems, dataframe)
        sort!(sortedItems, by = x -> x[1])
        sortedItems.index = range(0, length = size(sortedItems)[1])
    end
    return sortedItems.value
end

"""
Hierarchical Risk Parity portfolio construction.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.2, Page 230
"""
function recursiveBisection(cov, sortedItems)
    ω = DataFrames.DataFrame(index = sortedItems, weight = repeat([1.], length(sortedItems)))
    clusteredItems = [sortedItems]

    while length(clusteredItems) > 0
        clusteredItems = [i[j:k] for i in clusteredItems for (j, k) in ((1, div(length(i), 2)), (div(length(i), 2) + 1, length(i))) if length(i) > 1]
        for i in range(1, stop = length(clusteredItems), step = 2)
            clusteredItems0 = clusteredItems[i]
            clusteredItems1 = clusteredItems[i + 1]
            clusterVariance0 = varianceCluster(cov, clusteredItems0)
            clusterVariance1 = varianceCluster(cov, clusteredItems1)
            α = 1 - clusterVariance0 / (clusterVariance0 + clusterVariance1)
            ω[in.(ω.index, (clusteredItems0,)), :weight] .*= α
            ω[in.(ω.index, (clusteredItems1,)), :weight] .*= 1 - α
        end
    end
    return ω
end

"""
Compute the variance of a cluster.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.4, Page 240
""" 
function varianceCluster(cov, clusteredItems)
    covSlice = cov[clusteredItems .+ 1, clusteredItems .+ 1]
    ω = IVP(covSlice)
    clusterVariance = (transpose(ω) * covSlice) * ω
    return clusterVariance
end

"""
Inverse variance portfolio weights.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.4, Page 240
""" 
function IVP(cov, kwargs...)
    ivp = 1 ./ Statistics.diag(cov)
    ivp /= sum(ivp)
    return ivp
end

"""
Generate random data for Monte Carlo simulation.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.5, Page 242
"""   
function generalRandomData(numberObservations, lengthSample, size0, size1, mu0, sigma0, sigma1)
    data1 = rand(Normal(mu0, sigma0), numberObservations, size0)
    columns = rand(1:size0, size1)
    data2 = data1[:, columns] + rand(Normal(0, sigma0 * sigma1), numberObservations, length(columns))
    data = hcat(data1, data2)
    point = rand(lengthSample:numberObservations, 2)
    data[append!(point, [columns[1], size0])] = [-.5, -.5, 2, 2]
    point = rand(lengthSample:numberObservations, 2)
    data[point, columns[end]] = [-.5, 2]
    return data, columns
end

"""
Hierarchical Risk Parity (HRP) method.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.5, Page 243
"""  
function hrp(cov::Matrix, corr::Matrix)
    distance = distanceCorr(corr)
    link = sch.linkage(distance, "single")
    sortedItems = quasiDiagonal(link)
    hrp = recursiveBisection(cov, sortedItems)
    return sort(hrp).weight
end

"""
Monte Carlo simulation for out-of-sample comparison.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.5, Page 243
"""  
function hrpMc(; numberIters = 5e3, numberObservations = 520, size0 = 5, size1 = 5, mu0 = 0, sigma0 = 1e-2, sigma1 = .25, lengthSample = 260, testSize = 20)
    methods = [IVP, hrp]
    results, numIter = Dict(String.(Symbol.(methods)) .=> [[]]), 0
    pointers = range(lengthSample + 1, stop = numberObservations        
end