# todo: 
"""
List of Collaborators, Developers, and Research Assistants (in alphabetical order)
Daniel Norouzi 
"""

using Distributions
using Statistics
using Random
using DataFrames

#---------------------------------------------------
"""
function: validates the False Strategy Theorem experimentally
refernce: De Prado, M (2020) Machine Learning for Asset Managers
methodology: page 110, snippet 8.1
"""
function expectedMaxSharpeRatio(
    nTrials, # number of trials
    meanSharpeRatio, # mean Sharpe Ratio
    stdSharpeRatio# standard deviation of Sharpe ratios
)::Float64

    emc = MathConstants.eulergamma # euler gamma constant

    sharpeRatio = (1 - emc) * quantile(Normal(0, 1), 1 - 1 / nTrials) + emc * quantile(Normal(0, 1) , 1 - 1 / (nTrials * MathConstants.e)) # get expected value of sharpe ratio by using false strategy theorem
    sharpeRatio = meanSharpeRatio + stdSharpeRatio * sharpeRatio # get max Sharpe Ratio, controlling for SBuMT
    
    return sharpeRatio
end

#---------------------------------------------------
function generatedMaxSharpeRatio(
    nSims,#::Int, # number of simulations 
    nTrials,#::Int, # number of trials
    stdSharpeRatio,#::Float64, # mean Sharpe Ratio
    meanSharpeRatio#::Float64 # standard deviation of Sharpe Ratios
)::DataFrame
    out = DataFrame() # initialize output

    # Monte Carlo of max{SR} on nTrials, from nSims simulations
    for nTrials_ in nTrials
        #1) Simulated Sharpe ratios
        sharpeRatio = randn((Int64(nSims), Int64(nTrials_))) # generate random numbers for Sharpe Ratios
        sharpeRatio = (sharpeRatio .- mean(sharpeRatio, dims = 2)) ./std(sharpeRatio, dims = 2) # standardize Sharpe Ratios 
        sharpeRatio = meanSharpeRatio .+ sharpeRatio .* stdSharpeRatio # set the mean and standard deviation
        
        #2) Store output
        output = DataFrame(maxSharpeRatio = vec(maximum(sharpeRatio, dims = 2)), nTrials = nTrials_) # generate output
        append!(out, output) # append output
    end
    return out
end
#---------------------------------------------------

"""
function: calculates mean and standard deviation of the predicted errors
refernce: De Prado, M (2020) Machine Learning for Asset Managers
methodology: page 112, snippet 8.2
"""
function meanStdError(
    nSims0::Int, # number of max{SR} used to estimate E[max{SR}]
    nSims1::Int, # number of errors on which std is computed
    nTrials::Vector, # array of numbers of SR used to derive max{SR}
    stdSharpeRatio::Float64, # mean Sharpe Ratio
    meanSharpeRatio::Float64 # standard deviation of Sharpe Ratios
)::DataFrame
    # Compute standard deviation of errors per nTrial
    sharpeRatio0 = DataFrame(nT = nTrials, ExpectedMaxSR = [expectedMaxSharpeRatio(i, meanSharpeRatio, stdSharpeRatio) for i in nTrials]) # compute expected max Sharpe Ratios
    error = DataFrame() # initialize errors
    out = DataFrame() # initialize output

    for i in 1:nSims1
        sharpeRatio1 = generatedMaxSharpeRatio(nSims0, nTrials, stdSharpeRatio, meanSharpeRatio) # generate max Sharpe Ratios 
        sharpeRatio1 = combine(groupby(sharpeRatio1, :nTrials), :maxSharpeRatio => mean; renamecols=false) # calculate mean max Sharpe Ratios 
        error_ = DataFrame(sharpeRatio1) # create DataFrame of generated Max Sharpe Ratios with errors
        
        error_[!, :ExpectedMaxSR] = sharpeRatio0.ExpectedMaxSR # add expected max Sharpe Ratios
        error_[!, :err] = error_.maxSharpeRatio ./ error_.ExpectedMaxSR .- 1 # calculate errors
        append!(error, error_) # append errors
      
    end    
    out[!, :meanErr] = combine(groupby(error, :nTrials), :err => mean; renamecols=false).err # calculate mean errors
    out[!, :nTrials] = combine(groupby(error, :nTrials), :err => mean; renamecols=false).nTrials # add number of trials
    out[!, :stdErr] = combine(groupby(error, :nTrials), :err => std; renamecols=false).err # calculate standard deviation oferrors

    return out
end
#---------------------------------------------------

"""
function: calculates type I error probability of stratgies
refernce: De Prado, M (2020) Machine Learning for Asset Managers
methodology: page 119, snippet 8.3
"""
function estimatedSharpeRatioZStatistics(
    sharpeRatio, # estimated Sharpe Ratio
    t, # number of observations
    sharpeRatio_ = 0, # true Sharpe Ratio
    skew = 0, # skewness of returns
    kurt = 3 # kurtosis of returns
)::Float64

    z = (sharpeRatio - sharpeRatio_)*(t - 1)^0.5 # calculate first part of z statistic
    z /= (1 - skew*sr + (kurt - 1) / 4*sr^2)^0.5 # calculate z statistic

    return z
end

#---------------------------------------------------
function strategyType1ErrorProbability(z, # z statistic for the estimated Sharpe Ratios
                    k = 1) # number of tests

    α = cdf(Normal(0, 1), -z) # find false positive rate
    α_k = 1 - (1 - α) ^ k # correct for multi-testing 

    return α_k
end


#---------------------------------------------------
"""
function: calculates type II error probability of stratgies
refernce: De Prado, M (2020) Machine Learning for Asset Managers
methodology: page 121, snippet 8.4
"""
function thetaForType2Error(sharpeRatio, # estimated Sharpe Ratio
               t, # number of observations
               sharpeRatio_ = 0, # true Sharpe Ratio
               skew = 0, # skewness of returns
               kurt = 3) # kurtosis of returns

    θ = sharpeRatio_*(t - 1)^0.5 # calculate first part of theta
    θ /= (1 - skew*sharpeRatio + (kurt - 1) / 4*sharpeRatio^2)^0.5 # calculate theta

    return θ
end

#---------------------------------------------------
function strategyType2ErrorProbability(α, # type I error
                    k, # number of tests
                    θ) # calculated theta parameter

    z = quantile(Normal(0, 1),(1 - α) ^ (1 / k)) # perform Sidak’s correction
    β = cdf(Normal(0, 1), z - θ) # calculate false negative rate

    return β
end

