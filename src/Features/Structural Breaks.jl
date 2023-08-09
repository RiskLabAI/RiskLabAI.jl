"""
Compute β using Ordinary Least Squares (OLS) estimator.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: snippet 17.4
"""
function compute_beta(X,  # matrix of independent variable
                     y)   # dependent variable
    β = inv(transpose(X) * X) * transpose(X) * y  # Compute β with OLS estimator
    ϵ = y .- X * β  # compute error
    beta_variance = transpose(ϵ) * ϵ / (size(X)[1] - size(X)[2]) * inv(transpose(X) * X)  # compute variance of β
    return β, beta_variance
end

"""
Prepare data for the test.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: snippet 17.2 & 17.3
"""
function prepare_data(data,  # data of price or log price
                     constant,  # string that must be "nc" or "ct" or "ctt"
                     lags)  # array of lag or integer that show the number of lags
    if isinteger(lags)
        lags = 1:lags
    else
        lags = [Int(lag) for lag in lags]
    end
    diff_data = diff(data)  # compute data difference
    x_index = [Int.((zeros(length(lags)) .+ i) .- lags) for i in maximum(lags) + 1:length(diff_data)]  # index of each row, each row shows features
    X = [diff_data[x_index] for x_index in x_index]  # create feature matrix
    X = hcat(X...)'
    y = diff_data[maximum(lags) + 1:length(diff_data)]  # create dependent variable array respecting X
    X = hcat(data[(length(data) - length(y)):(end - 1)], X)  # add actual data columns to the feature matrix
    if constant != "nc"
        X = hcat(X, ones(size(X)[1]))  # append columns with 1
        trend = 1:size(X)[1]
        if constant[1:2] == "ct"
            X = hcat(X, trend)  # append columns t
        end
        if constant == "ctt"
            X = hcat(X, trend .^ 2)  # append columns t^2
        end
    end
    return X, y
end

"""
Compute the Augmented Dickey-Fuller (ADF) statistics.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: snippet 17.1
"""
function adf(X,  # feature matrix that contains data with lags
             y,  # dependent variable
             min_sample_length)  # minimum sample length for computing OLS
    start_points = 1:(length(y) - min_sample_length + 1)  # create start_points array
    adf_statistics = zeros(length(start_points))  # create ADF array
    for (i, start) in enumerate(start_points)
        y_, X_ = y[start:end], X[start:end, :]
        β_mean, β_std = compute_beta(X_, y_)  # compute β and its variance
        if !isnan(β_mean[1])
            β_mean_, β_std_ = β_mean[1], β_std[1, 1] ^ 0.5  # get coefficient of the first independent variable
            adf_statistics[i] = β_mean_ / β_std_  # compute Augmented Dickey Fuller statistics
        end
    end
    return adf_statistics
end

"""
Compute Augmented Dickey-Fuller (ADF) test statistics.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: -
"""
function adf_test_type(data,  # dataframe of price that has date index
                     min_sample_length,  # minimum sample length for OLS estimator
                     constant,  # string that must be "nc" or "ct" or "ctt"
                     lags,  # array of lag or integer that show the number of lags
                     type;  # type of ADF test: SADF or QADF or CADF
                     quantile = nothing,  # quantile for QADF
                     probability = ones(size(data)[1]))  # probability or weight for CADF
    X, y = prepare_data(data.price, constant, lags)  # Prepare data for the test
    max_lag = isinteger(lags) ? lags : maximum(lags)
    index_range = (min_sample_length - max_lag) : length(y)  # initial index range
    result = zeros(length(index_range))
    for (i, index) in enumerate(index_range)
        X_, y_ = X[1:index, :], y[1:index]
        adf_stats = adf(X_, y_, min_sample_length)  # compute ADF statistics
        if isempty(adf_stats)
            continue
        end
        if type == "SADF"
            result[i] = maximum(adf_stats)
        elseif type == "QADF"
            result[i] = sort(adf_stats)[floor(quantile * length(adf_stats))]
        elseif type == "CADF"
            perm = sortperm(adf_stats)
            perm = perm[floor(quantile * length(adf_stats)):end]
            result[i] = adf_stats[perm] .* probability[perm] / sum(probability[perm])
        else
            println("type must be SADF or QADF or CADF")
        end
    end
    adf_statistics = DataFrame(index = data.index[min_sample_length:length(y) + max_lag],
                                statistics = result)
    return adf_statistics
end

"""
Implement the Brown-Durbin-Evans cumsum test.

Reference: Techniques for Testing the Constancy of Regression Relationships over Time (1975)
Methodology: -
"""
function brown_durbin_evans_test(X,  # feature matrix (price with lags)
                                y,  # dependent variable
                                lags,  # integer lags
                                k,  # first index where statistics are computed from
                                index)  # index of feature matrix, typically date
    β, _ = compute_beta(X, y)
    ϵ = y - X * β
    σ = ϵ' * ϵ / (length(y) - 1 + lags - k)

    start_index = k - lags + 1
    cumsum = 0
    bde_cumsum_stats = zeros(length(y) - start_index)
    for i in start_index:length(y) - 1
        X_, y_ = X[1:i, :], y[1:i]
        β, _ = compute_beta(X_, y_)

        ω = (y[i + 1] - X[i + 1, :]' * β) / sqrt(1 + X[i + 1, :]' * inv(X_' * X_) * X[i + 1, :])
        cumsum += ω
        bde_cumsum_stats[i - start_index + 1] = cumsum / sqrt(σ)
    end
    bde_stats_df = DataFrame(index = index[k:length(y) + lags - 2],
                            bde_statistics = bde_cumsum_stats)
    return bde_stats_df
end

"""
Implement the Chu-Stinchcombe-White test.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: p.251
"""
function chu_stinchcombe_white_test(data,  # dataframe of log price with date index
                                    test_type,  # type of test: one_sided or two_sided
                                    interval)  # interval where test statistics are computed
    csw_stats, csw_threshold = zeros(length(interval)), zeros(length(interval))
    for (idx, i) in enumerate(interval)
        price = data[data.index .<= i, :].price
        diff_log_price = diff(price)
        σ = sum(diff_log_price .^ 2) / length(diff_log_price)
        max_s_n, max_s_n_rejecting_threshold = -1 * Inf, 0
        for j in 1:length(price) - 1
            s_nt = 0
            if test_type == "one_sided"
                s_nt = price[end] - data.price[j]
            elseif test_type == "two_sided"
                s_nt = abs(price[end] - data.price[j])
            end
            s_nt /= sqrt(σ * (length(price) - j))
            if s_nt > max_s_n
                max_s_n = s_nt
                max_s_n_rejecting_threshold = sqrt(4.6 + log(length(price) - j))
            end
        end
        csw_stats[idx] = max_s_n
        csw_threshold[idx] = max_s_n_rejecting_threshold
    end
    csw_test = DataFrame(index = interval, S_n = csw_stats, threshold = csw_threshold)
    return csw_test
end

"""
Implement the inner loop of the Chow-Type Dickey-Fuller test.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: p.252
"""
function chow_type_dickey_fuller_test_inner_loop(data,  # dataframe of log price with date index
                                                 interval)  # interval where test statistics are computed
    dfc_stats = zeros(length(interval))
    for (i, dates) in enumerate(interval)
        δy = diff(data.price)
        y = [(data.index[j] <= dates) ? 0.0 : data.price[j] for j in 1:length(δy)]
        y = reshape(y, (length(y), 1))
        δy = reshape(δy, (length(y), 1))
        β, β_std = compute_beta(y, δy)
        dfc_stats[i] = β[1] / sqrt(β_std[1, 1])
    end
    dfc_df = DataFrame(index = interval, dfc_statistics = dfc_stats)
    return dfc_df
end

"""
Implement the Chow-Type Dickey-Fuller test.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: p.252
"""
function sdfc(data,  # dataframe of price or log price with date index
              interval,  # interval where test statistics are computed
              τ,  # number between 0 and 1 to compute the interval for the inner loop
              T)  # maximum lags for estimating the model
    sdfc_stats = zeros(length(interval))
    for (idx, i) in enumerate(interval)
        select_data = data[(data.index .>= (i - Dates.Day(T))) .& (data.index .<= i), :]
        integertau = floor(T * τ)
        select_interval = data.index[(data.index .>= (i - Dates.Day(T - integertau))) .& (data.index .<= (i - Dates.Day(integertau)))]
        dfc = chow_type_dickey_fuller_test_inner_loop(select_data, select_interval)
        sdfc_stats[idx] = maximum(dfc.dfc_statistics)
    end
    sdfc_df = DataFrame(index = interval, sdfc_statistics = sdfc_stats)
    return sdfc_df
end
