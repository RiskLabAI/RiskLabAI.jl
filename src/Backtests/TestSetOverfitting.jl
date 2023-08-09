using Distributions
using Statistics
using Random
using DataFrames


"""
Function to validate the False Strategy Theorem experimentally.

:param nTrials: Number of trials.
:param meanSharpeRatio: Mean Sharpe Ratio.
:param stdSharpeRatio: Standard deviation of Sharpe ratios.
:return: Expected maximum Sharpe ratio.
"""
function expectedMaxSharpeRatio(
    nTrials, meanSharpeRatio, stdSharpeRatio
)::Float64
    emc = MathConstants.eulergamma

    sharpeRatio = (1 - emc) * quantile(Normal(0, 1), 1 - 1 / nTrials) +
        emc * quantile(Normal(0, 1), 1 - 1 / (nTrials * MathConstants.e))
    sharpeRatio = meanSharpeRatio + stdSharpeRatio * sharpeRatio

    return sharpeRatio
end

"""
Function to generate maximum Sharpe ratio.

:param nSims: Number of simulations.
:param nTrials: Number of trials.
:param stdSharpeRatio: Standard deviation of Sharpe Ratios.
:param meanSharpeRatio: Mean Sharpe Ratio.
:return: DataFrame containing generated maximum Sharpe ratios.
"""
function generatedMaxSharpeRatio(
    nSims, nTrials, stdSharpeRatio, meanSharpeRatio
)::DataFrame
    out = DataFrame()

    for nTrials_ in nTrials
        sharpeRatio = randn((Int64(nSims), Int64(nTrials_)))
        sharpeRatio = (sharpeRatio .- mean(sharpeRatio, dims = 2)) ./ std(sharpeRatio, dims = 2)
        sharpeRatio = meanSharpeRatio .+ sharpeRatio .* stdSharpeRatio

        output = DataFrame(maxSharpeRatio = vec(maximum(sharpeRatio, dims = 2)), nTrials = nTrials_)
        append!(out, output)
    end
    return out
end

"""
Function to calculate mean and standard deviation of the predicted errors.

:param nSims0: Number of max{SR} used to estimate E[max{SR}].
:param nSims1: Number of errors on which std is computed.
:param nTrials: Array of numbers of SR used to derive max{SR}.
:param stdSharpeRatio: Standard deviation of Sharpe Ratios.
:param meanSharpeRatio: Mean Sharpe Ratio.
:return: DataFrame containing mean and standard deviation of errors.
"""
function meanStdError(
    nSims0, nSims1, nTrials, stdSharpeRatio, meanSharpeRatio
)::DataFrame
    sharpeRatio0 = DataFrame(nT = nTrials,
        ExpectedMaxSR = [expectedMaxSharpeRatio(i, meanSharpeRatio, stdSharpeRatio) for i in nTrials])
    error = DataFrame()
    out = DataFrame()

    for i in 1:nSims1
        sharpeRatio1 = generatedMaxSharpeRatio(nSims0, nTrials, stdSharpeRatio, meanSharpeRatio)
        sharpeRatio1 = combine(groupby(sharpeRatio1, :nTrials), :maxSharpeRatio => mean; renamecols=false)
        error_ = DataFrame(sharpeRatio1)

        error_[!, :ExpectedMaxSR] = sharpeRatio0.ExpectedMaxSR
        error_[!, :err] = error_.maxSharpeRatio ./ error_.ExpectedMaxSR .- 1
        append!(error, error_)
    end    
    out[!, :meanErr] = combine(groupby(error, :nTrials), :err => mean; renamecols=false).err
    out[!, :nTrials] = combine(groupby(error, :nTrials), :err => mean; renamecols=false).nTrials
    out[!, :stdErr] = combine(groupby(error, :nTrials), :err => std; renamecols=false).err

    return out
end

"""
Function to calculate type I error probability of strategies.

:param sharpeRatio: Estimated Sharpe Ratio.
:param t: Number of observations.
:param sharpeRatio_: True Sharpe Ratio.
:param skew: Skewness of returns.
:param kurt: Kurtosis of returns.
:return: Estimated Z-statistics for the Sharpe Ratios.
"""
function estimatedSharpeRatioZStatistics(
    sharpeRatio, t, sharpeRatio_ = 0, skew = 0, kurt = 3
)::Float64
    z = (sharpeRatio - sharpeRatio_) * (t - 1)^0.5
    z /= (1 - skew * sharpeRatio + (kurt - 1) / 4 * sharpeRatio^2)^0.5

    return z
end

"""
Function to calculate strategy type I error probability.

:param z: Z statistic for the estimated Sharpe Ratios.
:param k: Number of tests.
:return: Type I error probability.
"""
function strategyType1ErrorProbability(z, k = 1)
    α = cdf(Normal(0, 1), -z)
    α_k = 1 - (1 - α)^k

    return α_k
end

"""
Function to calculate theta for type II error probability.

:param sharpeRatio: Estimated Sharpe Ratio.
:param t: Number of observations.
:param sharpeRatio_: True Sharpe Ratio.
:param skew: Skewness of returns.
:param kurt: Kurtosis of returns.
:return: Calculated theta parameter.
"""
function thetaForType2Error(sharpeRatio, t, sharpeRatio_ = 0, skew = 0, kurt = 3)
    θ = sharpeRatio_ * (t - 1)^0.5
    θ /= (1 - skew * sharpeRatio + (kurt - 1) / 4 * sharpeRatio^2)^0.5

    return θ
end


"""
Function to calculate strategy type II error probability.

:param α: Type I error.
:param k: Number of tests.
:param θ: Calculated theta parameter.
:return: Type II error probability.
"""
function strategyType2ErrorProbability(α, k, θ)
    z = quantile(Normal(0, 1), (1 - α)^(1 / k))
    β = cdf(Normal(0, 1), z - θ)

    return β
end
