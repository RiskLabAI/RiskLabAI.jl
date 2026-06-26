using Distributions
using Statistics
using Random
using DataFrames

"""
    expectedMaxSharpeRatio(
        nTrials::Int,
        meanSharpeRatio::Float64,
        stdSharpeRatio::Float64
    )::Float64

Validate the False Strategy Theorem experimentally.

# Arguments
- `nTrials::Int`: Number of trials.
- `meanSharpeRatio::Float64`: Mean Sharpe Ratio.
- `stdSharpeRatio::Float64`: Standard deviation of Sharpe ratios.

# Returns
- `Float64`: Expected maximum Sharpe ratio.
"""
function expectedMaxSharpeRatio(
    nTrials::Int,
    meanSharpeRatio::Float64,
    stdSharpeRatio::Float64
)::Float64
    emc = MathConstants.eulergamma

    sharpeRatio = (1 - emc) * quantile(Normal(0, 1), 1 - 1 / nTrials) +
        emc * quantile(Normal(0, 1), 1 - 1 / (nTrials * MathConstants.e))
    sharpeRatio = meanSharpeRatio + stdSharpeRatio * sharpeRatio

    return sharpeRatio
end

"""
    generateMaxSharpeRatio(
        nSims::Int,
        nTrials::Vector{Int},
        stdSharpeRatio::Float64,
        meanSharpeRatio::Float64
    )::DataFrame

Generate maximum Sharpe ratio.

# Arguments
- `nSims::Int`: Number of simulations.
- `nTrials::Vector{Int}`: Number of trials.
- `stdSharpeRatio::Float64`: Standard deviation of Sharpe Ratios.
- `meanSharpeRatio::Float64`: Mean Sharpe Ratio.

# Returns
- `DataFrame`: Contains generated maximum Sharpe ratios.
"""
function generateMaxSharpeRatio(
    nSims::Int,
    nTrials::Vector{Int},
    stdSharpeRatio::Float64,
    meanSharpeRatio::Float64
)::DataFrame
    out = DataFrame()

    for trials in nTrials
        sharpeRatio = randn(nSims, trials)
        sharpeRatio = (sharpeRatio .- mean(sharpeRatio, dims = 2)) ./ std(sharpeRatio, dims = 2)
        sharpeRatio = meanSharpeRatio .+ sharpeRatio .* stdSharpeRatio

        output = DataFrame(maxSharpeRatio = vec(maximum(sharpeRatio, dims = 2)), nTrials = trials)
        append!(out, output)
    end
    return out
end

"""
    meanStdError(
        numSimulations0::Int,
        numSimulations1::Int,
        numTrials::Vector{Int},
        stdSharpeRatio::Float64,
        meanSharpeRatio::Float64
    )::DataFrame

Calculate the mean and standard deviation of the predicted errors.

Parameters
----------
- `numSimulations0`: Number of max{SR} used to estimate E[max{SR}].
- `numSimulations1`: Number of errors on which std is computed.
- `numTrials`: Array of numbers of SR used to derive max{SR}.
- `stdSharpeRatio`: Standard deviation of Sharpe Ratios.
- `meanSharpeRatio`: Mean Sharpe Ratio.

Returns
-------
- DataFrame containing mean and standard deviation of errors.
"""
function meanStdError(
    numSimulations0::Int,
    numSimulations1::Int,
    numTrials::Vector{Int},
    stdSharpeRatio::Float64,
    meanSharpeRatio::Float64
)::DataFrame
    sharpeRatio0 = DataFrame(
        nT = numTrials,
        ExpectedMaxSR = [expectedMaxSharpeRatio(i, meanSharpeRatio, stdSharpeRatio) for i in numTrials]
    )
    errors = DataFrame()
    output = DataFrame()

    for i in 1:numSimulations1
        sharpeRatio1 = generatedMaxSharpeRatio(numSimulations0, numTrials, stdSharpeRatio, meanSharpeRatio)
        sharpeRatio1 = combine(groupby(sharpeRatio1, :nTrials), :maxSharpeRatio => mean; renamecols=false)
        error = DataFrame(sharpeRatio1)

        error[!, :ExpectedMaxSR] = sharpeRatio0.ExpectedMaxSR
        error[!, :err] = error.maxSharpeRatio ./ error.ExpectedMaxSR .- 1
        append!(errors, error)
    end
    output[!, :meanErr] = combine(groupby(errors, :nTrials), :err => mean; renamecols=false).err
    output[!, :nTrials] = combine(groupby(errors, :nTrials), :err => mean; renamecols=false).nTrials
    output[!, :stdErr] = combine(groupby(errors, :nTrials), :err => std; renamecols=false).err

    return output
end

"""
    estimatedSharpeRatioZStatistics(
        sharpeRatio::Float64,
        t::Int,
        sharpeRatio_::Float64 = 0,
        skew::Float64 = 0,
        kurt::Float64 = 3
    )::Float64

Calculate the estimated Z-statistics for the Sharpe Ratios.

Parameters
----------
- `sharpeRatio`: Estimated Sharpe Ratio.
- `t`: Number of observations.
- `sharpeRatio_`: True Sharpe Ratio.
- `skew`: Skewness of returns.
- `kurt`: Kurtosis of returns.

Returns
-------
- Estimated Z-statistics for the Sharpe Ratios.
"""
function estimatedSharpeRatioZStatistics(
    sharpeRatio::Float64,
    t::Int,
    sharpeRatio_::Float64 = 0,
    skew::Float64 = 0,
    kurt::Float64 = 3
)::Float64
    z = (sharpeRatio - sharpeRatio_) * sqrt(t - 1)
    z /= sqrt(1 - skew * sharpeRatio + (kurt - 1) / 4 * sharpeRatio^2)

    return z
end

"""
    strategyType1ErrorProbability(z::Float64, k::Int = 1)

Calculate the type I error probability of strategies.

Parameters
----------
- `z`: Z statistic for the estimated Sharpe Ratios.
- `k`: Number of tests.

Returns
-------
- Type I error probability.
"""
function strategyType1ErrorProbability(z::Float64, k::Int = 1)
    α = cdf(Normal(0, 1), -z)
    α_k = 1 - (1 - α)^k

    return α_k
end

"""
    thetaForType2Error(
        sharpeRatio::Float64,
        t::Int,
        sharpeRatio_::Float64 = 0,
        skew::Float64 = 0,
        kurt::Float64 = 3
    )

Calculate the theta parameter for type II error probability.

Parameters
----------
- `sharpeRatio`: Estimated Sharpe Ratio.
- `t`: Number of observations.
- `sharpeRatio_`: True Sharpe Ratio.
- `skew`: Skewness of returns.
- `kurt`: Kurtosis of returns.

Returns
-------
- Calculated theta parameter.
"""
function thetaForType2Error(
    sharpeRatio::Float64,
    t::Int,
    sharpeRatio_::Float64 = 0,
    skew::Float64 = 0,
    kurt::Float64 = 3
)
    θ = sharpeRatio_ * sqrt(t - 1)
    θ /= sqrt(1 - skew * sharpeRatio + (kurt - 1) / 4 * sharpeRatio^2)

    return θ
end

"""
    strategyType2ErrorProbability(α::Float64, k::Int, θ::Float64)

Calculate the strategy type II error probability.

Parameters
----------
- `α`: Type I error.
- `k`: Number of tests.
- `θ`: Calculated theta parameter.

Returns
-------
- Type II error probability.
"""
function strategyType2ErrorProbability(α::Float64, k::Int, θ::Float64)
    z = quantile(Normal(0, 1), (1 - α)^(1 / k))
    β = cdf(Normal(0, 1), z - θ)

    return β
end
