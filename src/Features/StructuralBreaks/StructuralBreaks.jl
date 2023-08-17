using LinearAlgebra
using Statistics
using DataFrames

"""
    computeBeta(X::Matrix, y::Vector)::Tuple{Vector, Matrix}

Compute β using Ordinary Least Squares (OLS) estimator.

# Parameters
- `X`: Matrix, matrix of independent variables
- `y`: Vector, dependent variable

# Returns
- `β`: Vector, OLS estimator of β
- `betaVariance`: Matrix, variance of β

# References
- De Prado, M. (2018) Advances in financial machine learning, Methodology: snippet 17.4
"""
function computeBeta(
        X::Matrix,
        y::Vector
    )::Tuple{Vector, Matrix}
    
    β = inv(transpose(X) * X) * transpose(X) * y
    ϵ = y .- X * β
    betaVariance = transpose(ϵ) * ϵ / (size(X, 1) - size(X, 2)) * inv(transpose(X) * X)
    return β, betaVariance
end

"""
    prepareData(data::Vector, constant::String, lags)::Tuple{Matrix, Vector}

Prepare data for the test.

# Parameters
- `data`: Vector, data of price or log price
- `constant`: String, must be "nc" or "ct" or "ctt"
- `lags`: Int or Vector, array of lag or integer that show the number of lags

# Returns
- `X`: Matrix, feature matrix
- `y`: Vector, dependent variable

# References
- De Prado, M. (2018) Advances in financial machine learning, Methodology: snippets 17.2 & 17.3
"""
function prepareData(
        data::Vector,
        constant::String,
        lags
    )::Tuple{Matrix, Vector}
    
    if isinteger(lags)
        lags = 1:lags
    else
        lags = [Int(lag) for lag in lags]
    end
    diffData = diff(data)
    xIndex = [Int.((zeros(length(lags)) .+ i) .- lags) for i in maximum(lags) + 1:length(diffData)]
    X = [diffData[xIndex] for xIndex in xIndex]
    X = hcat(X...)'
    y = diffData[maximum(lags) + 1:length(diffData)]
    X = hcat(data[(length(data) - length(y)):(end - 1)], X)
    if constant != "nc"
        X = hcat(X, ones(size(X, 1)))
        trend = 1:size(X, 1)
        if constant[1:2] == "ct"
            X = hcat(X, trend)
        end
        if constant == "ctt"
            X = hcat(X, trend .^ 2)
        end
    end
    return X, y
end

"""
    adf(X::Matrix, y::Vector, minSampleLength::Int)::Vector

Compute the Augmented Dickey-Fuller (ADF) statistics.

# Parameters
- `X`: Matrix, feature matrix that contains data with lags
- `y`: Vector, dependent variable
- `minSampleLength`: Int, minimum sample length for computing OLS

# Returns
- `adfStatistics`: Vector, ADF statistics

# References
- De Prado, M. (2018) Advances in financial machine learning, Methodology: snippet 17.1
"""
function adf(
        X::Matrix,
        y::Vector,
        minSampleLength::Int
    )::Vector
    
    startPoints = 1:(length(y) - minSampleLength + 1)
    adfStatistics = zeros(length(startPoints))
    for (i, start) in enumerate(startPoints)
        y_, X_ = y[start:end], X[start:end, :]
        βMean, βStd = computeBeta(X_, y_)
        if !isnan(βMean[1])
            βMean_, βStd_ = βMean[1], βStd[1, 1] ^ 0.5
            adfStatistics[i] = βMean_ / βStd_
        end
    end
    return adfStatistics
end

"""
    adfTestType(data::DataFrame, minSampleLength::Int, constant::String, lags; type::String, quantile::Union{Nothing, Float64} = nothing, probability::Vector = ones(size(data, 1)))::DataFrame

Compute Augmented Dickey-Fuller (ADF) test statistics.

# Parameters
- `data`: DataFrame, dataframe of price that has date index
- `minSampleLength`: Int, minimum sample length for OLS estimator
- `constant`: String, must be "nc" or "ct" or "ctt"
- `lags`: Int or Vector, array of lag or integer that show the number of lags
- `type`: String, type of ADF test: SADF or QADF or CADF
- `quantile`: Union{Nothing, Float64}, quantile for QADF
- `probability`: Vector, probability or weight for CADF

# Returns
- `adfStatistics`: DataFrame, ADF test statistics

# References
- De Prado, M. (2018) Advances in financial machine learning
"""
function adfTestType(
        data::DataFrame,
        minSampleLength::Int,
        constant::String,
        lags,
        type::String;
        quantile::Union{Nothing, Float64} = nothing,
        probability::Vector = ones(size(data, 1))
    )::DataFrame
    
    X, y = prepareData(data.price, constant, lags)
    maxLag = isinteger(lags) ? lags : maximum(lags)
    indexRange = (minSampleLength - maxLag) : length(y)
    result = zeros(length(indexRange))
    for (i, index) in enumerate(indexRange)
        X_, y_ = X[1:index, :], y[1:index]
        adfStats = adf(X_, y_, minSampleLength)
        if isempty(adfStats)
            continue
        end
        if type == "SADF"
            result[i] = maximum(adfStats)
        elseif type == "QADF"
            result[i] = sort(adfStats)[floor(Int, quantile * length(adfStats))]
        elseif type == "CADF"
            perm = sortperm(adfStats)
            perm = perm[floor(Int, quantile * length(adfStats)):end]
            result[i] = adfStats[perm] .* probability[perm] / sum(probability[perm])
        else
            println("type must be SADF or QADF or CADF")
        end
    end
    adfStatistics = DataFrame(index = data.index[minSampleLength:length(y) + maxLag], statistics = result)
    return adfStatistics
end

using DataFrames
using LinearAlgebra
using Statistics

"""
    brownDurbinEvansTest(
        X::Matrix,
        y::Vector,
        lags::Int,
        k::Int,
        index::Vector
    )::DataFrame

Compute the Brown-Durbin-Evans cumsum test, which tests for the constancy of regression relationships over time.

# Parameters
- `X::Matrix`: Feature matrix (price with lags)
- `y::Vector`: Dependent variable
- `lags::Int`: Number of lags
- `k::Int`: First index where statistics are computed from
- `index::Vector`: Index of the feature matrix, typically date

# Returns
- `bdeStatsDf::DataFrame`: DataFrame with index and BDE statistics

# References
- Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for Testing the Constancy of Regression Relationships over Time.
"""
function brownDurbinEvansTest(
        X::Matrix,
        y::Vector,
        lags::Int,
        k::Int,
        index::Vector
    )::DataFrame
    
    β, _ = computeBeta(X, y)
    residuals = y - X * β
    σ = residuals' * residuals / (length(y) - 1 + lags - k)

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
    bdeStatsDf = DataFrame(index = index[k:length(y) + lags - 2], bdeStatistics = bdeCumsumStats)
    return bdeStatsDf
end

"""
    chuStinchcombeWhiteTest(
        data::DataFrame,
        testType::String,
        interval::Vector
    )::DataFrame

Compute the Chu-Stinchcombe-White test, a statistical test for detecting explosive bubbles in financial data.

# Parameters
- `data::DataFrame`: DataFrame of log price with date index
- `testType::String`: Type of test: "one_sided" or "two_sided"
- `interval::Vector`: Interval where test statistics are computed

# Returns
- `cswTest::DataFrame`: DataFrame with S_n statistics and threshold

# References
- De Prado, M. (2018) Advances in financial machine learning, p.251
"""
function chuStinchcombeWhiteTest(
        data::DataFrame,
        testType::String,
        interval::Vector
    )::DataFrame
    
    cswStats, cswThreshold = zeros(length(interval)), zeros(length(interval))
    for (idx, i) in enumerate(interval)
        price = data[data.index .<= i, :].price
        diffLogPrice = diff(price)
        σ = sum(diffLogPrice .^ 2) / length(diffLogPrice)
        maxSN, maxSNRejectingThreshold = -Inf, 0
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


using DataFrames
using Statistics
using Dates

"""
    chowTypeDickeyFullerTestInnerLoop(
        data::DataFrame,
        interval::Vector{Date}
    )::DataFrame

Implement the inner loop of the Chow-Type Dickey-Fuller test to calculate the Dickey-Fuller test statistics.

# Parameters
- `data::DataFrame`: DataFrame of log price with date index
- `interval::Vector{Date}`: Interval where test statistics are computed

# Returns
- `dfcDf::DataFrame`: DataFrame with index and DFC statistics

# References
- De Prado, M. (2018) Advances in financial machine learning, p.252
"""
function chowTypeDickeyFullerTestInnerLoop(
        data::DataFrame,
        interval::Vector{Date}
    )::DataFrame
    
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
    sdfc(
        data::DataFrame,
        interval::Vector{Date},
        τ::Float64,
        T::Int
    )::DataFrame

Implement the Chow-Type Dickey-Fuller test to detect structural breaks in a time series.

# Parameters
- `data::DataFrame`: DataFrame of price or log price with date index
- `interval::Vector{Date}`: Interval where test statistics are computed
- `τ::Float64`: Number between 0 and 1 to compute the interval for the inner loop
- `T::Int`: Maximum lags for estimating the model

# Returns
- `sdfcDf::DataFrame`: DataFrame with index and SDFC statistics

# References
- De Prado, M. (2018) Advances in financial machine learning, p.252
"""
function sdfc(
        data::DataFrame,
        interval::Vector{Date},
        τ::Float64,
        T::Int
    )::DataFrame
    
    sdfcStats = zeros(length(interval))
    for (idx, i) in enumerate(interval)
        selectData = data[(data.index .>= (i - Day(T))) .& (data.index .<= i), :]
        integertau = floor(Int, T * τ)
        selectInterval = data.index[(data.index .>= (i - Day(T - integertau))) .& (data.index .<= (i - Day(integertau)))]
        dfc = chowTypeDickeyFullerTestInnerLoop(selectData, selectInterval)
        sdfcStats[idx] = maximum(dfc.dfcStatistics)
    end
    sdfcDf = DataFrame(index = interval, sdfcStatistics = sdfcStats)
    return sdfcDf
end
