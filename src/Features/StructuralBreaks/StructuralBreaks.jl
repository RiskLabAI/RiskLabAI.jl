using PyCall

@pyimport scipy.stats as Stats

"""
Compute β using Ordinary Least Squares (OLS) estimator.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: snippet 17.4
"""
function computeBeta(X,  # matrix of independent variable
                     y)   # dependent variable
    β = inv(transpose(X) * X) * transpose(X) * y  # Compute β with OLS estimator
    ϵ = y .- X * β  # compute error
    betaVariance = transpose(ϵ) * ϵ / (size(X)[1] - size(X)[2]) * inv(transpose(X) * X)  # compute variance of β
    return β, betaVariance
end

"""
Prepare data for the test.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: snippet 17.2 & 17.3
"""
function prepareData(data,  # data of price or log price
                     constant,  # string that must be "nc" or "ct" or "ctt"
                     lags)  # array of lag or integer that show the number of lags
    if isinteger(lags)
        lags = 1:lags
    else
        lags = [Int(lag) for lag in lags]
    end
    diffData = diff(data)  # compute data difference
    xIndex = [Int.((zeros(length(lags)) .+ i) .- lags) for i in maximum(lags) + 1:length(diffData)]  # index of each row, each row shows features
    X = [diffData[xIndex] for xIndex in xIndex]  # create feature matrix
    X = hcat(X...)'
    y = diffData[maximum(lags) + 1:length(diffData)]  # create dependent variable array respecting X
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
             minSampleLength)  # minimum sample length for computing OLS
    startPoints = 1:(length(y) - minSampleLength + 1)  # create startPoints array
    adfStatistics = zeros(length(startPoints))  # create ADF array
    for (i, start) in enumerate(startPoints)
        y_, X_ = y[start:end], X[start:end, :]
        βMean, βStd = computeBeta(X_, y_)  # compute β and its variance
        if !isnan(βMean[1])
            βMean_, βStd_ = βMean[1], βStd[1, 1] ^ 0.5  # get coefficient of the first independent variable
            adfStatistics[i] = βMean_ / βStd_  # compute Augmented Dickey Fuller statistics
        end
    end
    return adfStatistics
end

"""
Compute Augmented Dickey-Fuller (ADF) test statistics.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: -
"""
function adfTestType(data,  # dataframe of price that has date index
                     minSampleLength,  # minimum sample length for OLS estimator
                     constant,  # string that must be "nc" or "ct" or "ctt"
                     lags,  # array of lag or integer that show the number of lags
                     type;  # type of ADF test: SADF or QADF or CADF
                     quantile = nothing,  # quantile for QADF
                     probability = ones(size(data)[1]))  # probability or weight for CADF
    X, y = prepareData(data.price, constant, lags)  # Prepare data for the test
    maxLag = isinteger(lags) ? lags : maximum(lags)
    indexRange = (minSampleLength - maxLag) : length(y)  # initial index range
    result = zeros(length(indexRange))
    for (i, index) in enumerate(indexRange)
        X_, y_ = X[1:index, :], y[1:index]
        adfStats = adf(X_, y_, minSampleLength)  # compute ADF statistics
        if isempty(adfStats)
            continue
        end
        if type == "SADF"
            result[i] = maximum(adfStats)
        elseif type == "QADF"
            result[i] = sort(adfStats)[floor(quantile * length(adfStats))]
        elseif type == "CADF"
            perm = sortperm(adfStats)
            perm = perm[floor(quantile * length(adfStats)):end]
            result[i] = adfStats[perm] .* probability[perm] / sum(probability[perm])
        else
            println("type must be SADF or QADF or CADF")
        end
    end
    adfStatistics = DataFrame(index = data.index[minSampleLength:length(y) + maxLag],
                                statistics = result)
    return adfStatistics
end

"""
Implement the Brown-Durbin-Evans cumsum test.

Reference: Techniques for Testing the Constancy of Regression Relationships over Time (1975)
Methodology: -
"""
function brownDurbinEvansTest(X,  # feature matrix (price with lags)
                                y,  # dependent variable
                                lags,  # integer lags
                                k,  # first index where statistics are computed from
                                index)  # index of feature matrix, typically date
    β, _ = computeBeta(X, y)
    ϵ = y - X * β
    σ = ϵ' * ϵ / (length(y) - 1 + lags - k)

    startIndex = k - lags + 1
    cumsum = 0
    bdeCumsumStats = zeros(length(y) - startIndex)
    for i in startIndex:length(y) - 1
        X_, y_ = X[1:i, :], y[1:i]
        β, _ = computeBeta(X_, y_)

        ω = (y[i + 1] - X[i + 1, :]' * β) / sqrt(1 + X[i + 1, :]' * inv(X_' * X_) * X[i + 1, :])
        cumsum += ω
        bdeCumsumStats[i - startIndex + 1] = cumsum / sqrt(σ)
    end
    bdeStatsDf = DataFrame(index = index[k:length(y) + lags - 2],
                            bdeStatistics = bdeCumsumStats)
    return bdeStatsDf
end

"""
Implement the Chu-Stinchcombe-White test.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: p.251
"""
function chuStinchcombeWhiteTest(data,  # dataframe of log price with date index
                                    testType,  # type of test: one_sided or two_sided
                                    interval)  # interval where test statistics are computed
    cswStats, cswThreshold = zeros(length(interval)), zeros(length(interval))
    for (idx, i) in enumerate(interval)
        price = data[data.index .<= i, :].price
        diffLogPrice = diff(price)
        σ = sum(diffLogPrice .^ 2) / length(diffLogPrice)
        maxSN, maxSNRejectingThreshold = -1 * Inf, 0
        for j in 1:length(price) - 1
            sNT = 0
            if testType == "one_sided"
                sNT = price[end] - data.price[j]
            elseif testType == "two_sided"
                sNT = abs(price[end] - data.price[j])
            end
            sNT /= sqrt(σ * (length(price) - j))
            if sNT > maxSN
                maxSN = sNT
                maxSNRejectingThreshold = sqrt(4.6 + log(length(price) - j))
            end
        end
        cswStats[idx] = maxSN
        cswThreshold[idx] = maxSNRejectingThreshold
    end
    cswTest = DataFrame(index = interval, S_n = cswStats, threshold = cswThreshold)
    return cswTest
end

"""
Implement the inner loop of the Chow-Type Dickey-Fuller test.

Reference: De Prado, M. (2018) Advances in financial machine learning.
Methodology: p.252
"""
function chowTypeDickeyFullerTestInnerLoop(data,  # dataframe of log price with date index
                                                 interval)  # interval where test statistics are computed
    dfcStats = zeros(length(interval))
    for (i, dates) in enumerate(interval)
        δy = diff(data.price)
        y = [(data.index[j] <= dates) ? 0.0 : data.price[j] for j in 1:length(δy)]
        y = reshape(y, (length(y), 1))
        δy = reshape(δy, (length(y), 1))
        β, βStd = computeBeta(y, δy)
        dfcStats[i] = β[1] / sqrt(βStd[1, 1])
    end
    dfcDf = DataFrame(index = interval, dfcStatistics = dfcStats)
    return dfcDf
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
    sdfcStats = zeros(length(interval))
    for (idx, i) in enumerate(interval)
        selectData = data[(data.index .>= (i - Dates.Day(T))) .& (data.index .<= i), :]
        integertau = floor(T * τ)
        selectInterval = data.index[(data.index .>= (i - Dates.Day(T - integertau))) .& (data.index .<= (i - Dates.Day(integertau)))]
        dfc = chowTypeDickeyFullerTestInnerLoop(selectData, selectInterval)
        sdfcStats[idx] = maximum(dfc.dfcStatistics)
    end
    sdfcDf = DataFrame(index = interval, sdfcStatistics = sdfcStats)
    return sdfcDf
end
