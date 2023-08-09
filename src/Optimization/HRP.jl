"""
Generate random data for MC simulation.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.4, Page 241
"""
function generate_random_data(number_observations, size0, size1, sigma1)
    data1 = rand(Normal(0, 1), number_observations, size0)
    columns = rand(1:size0, size1)
    data2 = data1[:, columns] + rand(Normal(0, sigma1), number_observations, length(columns))
    data = hcat(data1, data2)
    dataframe = DataFrames.DataFrame(data, :auto)
    DataFrames.rename!(dataframe, Symbol.(names(dataframe)) .=> Symbol.(collect(1:size(data)[2])))
    return dataframe, columns
end

"""
Compute distance from correlation matrix.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.4, Page 241
"""
function distance_corr(correlation)
    distance = (1 .- correlation)
    distance = distance.^0.5
    return distance
end

"""
Quasi-diagonalization of a linkage matrix.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.2, Page 229
"""
function quasi_diagonal(link)
    link = Int.(floor.(link))
    sorted_items = DataFrames.DataFrame(index = [1, 2], value = [link[end,1], link[end, 2]])
    number_items = link[end, 4]
    while maximum(sorted_items.value) >= number_items 
        sorted_items.index = range(0, stop = size(sorted_items)[1]*2 - 1, step = 2)
        dataframe = sorted_items[sorted_items.value .>= number_items, :]
        index = dataframe.index
        value = dataframe.value .- number_items
        sorted_items[in.(sorted_items.index, (index,)), :value] = link[value .+ 1,1]
        dataframe = DataFrames.DataFrame(index = index .+ 1, value = link[value .+ 1, 2])
        sorted_items = vcat(sorted_items, dataframe)
        sort!(sorted_items, by = x -> x[1])
        sorted_items.index = range(0, length = size(sorted_items)[1])
    end
    return sorted_items.value
end

"""
Hierarchical Risk Parity portfolio construction.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.2, Page 230
"""
function recursive_bisection(cov, sorted_items)
    ω = DataFrames.DataFrame(index = sorted_items, weight = repeat([1.], length(sorted_items)))
    clustered_items = [sorted_items]

    while length(clustered_items) > 0
        clustered_items = [i[j:k] for i in clustered_items for (j, k) in ((1, div(length(i), 2)), (div(length(i), 2) + 1, length(i))) if length(i) > 1]
        for i in range(1, stop = length(clustered_items), step = 2)
            clustered_items0 = clustered_items[i]
            clustered_items1 = clustered_items[i + 1]
            cluster_variance0 = variance_cluster(cov, clustered_items0)
            cluster_variance1 = variance_cluster(cov, clustered_items1)
            α = 1 - cluster_variance0 / (cluster_variance0 + cluster_variance1)
            ω[in.(ω.index, (clustered_items0,)), :weight] .*= α
            ω[in.(ω.index, (clustered_items1,)), :weight] .*= 1 - α
        end
    end
    return ω
end

"""
Compute the variance of a cluster.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.4, Page 240
""" 
function variance_cluster(cov, clustered_items)
    cov_slice = cov[clustered_items .+ 1, clustered_items .+ 1]
    ω = IVP(cov_slice)
    cluster_variance = (transpose(ω) * cov_slice) * ω
    return cluster_variance
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
function general_random_data(number_observations, length_sample, size0, size1, mu0, sigma0, sigma1)
    data1 = rand(Normal(mu0, sigma0), number_observations, size0)
    columns = rand(1:size0, size1)
    data2 = data1[:, columns] + rand(Normal(0, sigma0 * sigma1), number_observations, length(columns))
    data = hcat(data1, data2)
    point = rand(length_sample:number_observations, 2)
    data[append!(point, [columns[1], size0])] = [-.5, -.5, 2, 2]
    point = rand(length_sample:number_observations, 2)
    data[point, columns[end]] = [-.5, 2]
    return data, columns
end
        
"""
Hierarchical Risk Parity (HRP) method.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.5, Page 243
"""  
function hrp(cov::Matrix, corr::Matrix)
    distance = distance_corr(corr)
    link = sch.linkage(distance, "single")
    sorted_items = quasi_diagonal(link)
    hrp = recursive_bisection(cov, sorted_items)
    return sort(hrp).weight
end

"""
Monte Carlo simulation for out-of-sample comparison.

Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Snippet 16.5, Page 243
"""  
function hrp_mc(; number_iters = 5e3, number_observations = 520, size0 = 5, size1 = 5, mu0 = 0, sigma0 = 1e-2, sigma1 = .25, length_sample = 260, test_size = 20)
    methods = [IVP, hrp]
    results, num_iter = Dict(String.(Symbol.(methods)) .=> [[]]), 0
    pointers = range(length_sample + 1, stop = number_observations, step = test_size)
    while num_iter < number_iters
        data, columns = general_random_data(number_observations, length_sample, size0, size1, mu0, sigma0, sigma1)
        returns = Dict(String.(Symbol.(methods)) .=> [[]])
        for pointer in pointers
            in_sample = data[pointer - length_sample:pointer - 1, :]
            covariance = cov(in_sample)
            correlation = cor(in_sample)
            out_sample = data[pointer:pointer + test_size - 1, :]
            for func in methods
                weights = func(covariance, correlation)
                ret = out_sample * weights
                returns[String(Symbol(func))] = vcat(returns[String(Symbol(func))], ret)
            end
        end
        for func in methods
            ret = returns[String(Symbol(func))]
            cumprod_return = cumprod(ret .+ 1)
            results[String(Symbol(func))] = vcat(results[String(Symbol(func))], [cumprod_return[end] - 1])
        end
        num_iter += 1
    end
    results = DataFrames.DataFrame(results)
    std_results, var_results = std.(eachcol(results)), var.(eachcol(results))
    return hcat(std_results, var_results, var_results ./ var_results[2] .- 1)
end
