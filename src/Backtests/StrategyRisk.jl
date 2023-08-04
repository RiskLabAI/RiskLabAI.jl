# todo: fix function signatures

using Distributions
using Random
using Statistics
using SymPy

"""
	function: targets a Sharpe ratio as a function of the number of bets
	reference: De Prado, M. (2018) Advances in financial machine learning.
	methodology: page 213, snippet 15.1
"""
function sharpeRatioTrials(
		p::Float64, # probability of success
		nRun::Int64 # number of runs
)::Tuple{Float64, Float64, Float64} 

    result = [] # initial results list

    for i in 1:nRun
        b = Binomial(1, p) # binomial distribution with probability p
        random = rand(b, 1) # generate random number with binomial distribution b

        if random[1] == 1 # find if the generated number is 1
            x = 1 # define x as 1
        else 
            x = -1 # define x as -1
        end

        append!(result, [x]) # append result
    end

    return (mean(result), std(result), mean(result)/std(result))
end 

"""
	function: uses the SymPy library for symbolic operations 
	reference: De Prado, M. (2018) Advances in financial machine learning.
	methodology: page 214, snippet 15.2
"""
function targetSharpeRatioSymbolic()

    p,u,d = symbols("p u d") # Create symbols

    m2 = p*u^2 + (1 - p)*d^2 # do symbolic operations
    m1 = p*u + (1 - p)*d # do symbolic operations
    v = m2 - m1^2 # do symbolic operations

    factor(v) 
end

"""
function: computes implied precision 
reference: De Prado, M. (2018) Advances in financial machine learning.
methodology: page 214, snippet 15.3
"""
function impliedPrecision(
		stopLoss::Float64, # stop loss threshold
		profitTaking::Float64, # profit taking threshold
		frequency::Int64, # number of bets per year
		targetSharpeRatio::Flaot64 # target annual Sharpe ratio
)::Float64

	a = (frequency + targetSharpeRatio^2)*(profitTaking - stopLoss)^2 # calculate the "a" parameter for the quadratic euation
	b = (2*frequency*stopLoss - targetSharpeRatio^2*(profitTaking - stopLoss))*(profitTaking - stopLoss) # calculate the "b" parameter for the quadratic euation
	c = frequency*stopLoss^2 # calculate the "c" parameter for the quadratic euation
	precision = (-b + (b^2 - 4*a*c)^0.5)/(2*a) # solve the quadratic equation

	return precision
end

"""
function: computes the number of bets/year needed to achieve a Sharpe ratio with a certain precision rate
reference: De Prado, M. (2018) Advances in financial machine learning.
methodology: page 215, snippet 15.4
"""
function binFrequency(stopLoss, # stop loss threshold
					  profitTaking, # profit taking threshold
					  precision, # precision rate p
					  targetSharpeRatio) # target annual Sharpe ratio

	frequency = (targetSharpeRatio*(profitTaking - stopLoss))^2*precision*(1 - precision) / ((profitTaking - stopLoss)*precision + stopLoss)^2  # calculate possible extraneous
	
	binSR(sl0, pt0, freq0, p0) = (((pt0 - sl0)*p0 + sl0)*freq0^0.5) / ((pt0 - sl0)*(p0*(1 - p0))^0.5) # Define Sharpe Ratio function
	
	if !isapprox(binSR(stopLoss, profitTaking, frequency, precision), targetSharpeRatio, atol=0.5) # check if it's near the target Sharpe Ratio
		return nothing
    end

	return frequency
end

"""
function: calculates the strategy risk in practice
reference: De Prado, M. (2018) Advances in financial machine learning.
methodology: page 215, snippet 15.4
"""
function mixGaussians(μ1, # mean of the first gaussian distribution to generate bet outcomes
					  μ2, # mean of the second gaussian distribution to generate bet outcomes
					  σ1, # standard deviation of the first gaussian distribution to generate bet outcomes
					  σ2, # standard deviation of the second gaussian distribution to generate bet outcomes
					  probability, # probability of success
					  nObs) # number of observations

	return1=rand(Normal(μ1, σ1), trunc(Int,(nObs*probability))) # draw random bet outcomes from a gaussian distribution
    return2=rand(Normal(μ2, σ2), trunc(Int,(nObs))-trunc(Int,(nObs*probability))) # draw random bet outcomes from a gaussian distribution
	
	returns = append!(return1, return2) # append returns
	shuffle!(returns) # shuffle returns

	return returns
end 

function failureProbability(returns, # returns list
							frequency, # number of bets per year
							targetSharpeRatio) # target annual Sharpe ratio

    # derive the probability that the strategy may fail
    rPositive, rNegative = mean(returns[returns .> 0]), mean(returns[returns .<= 0]) # divide returns
    p = size(returns[returns .> 0], 1)/size(returns, 1) # find success rate
	thresholdP = impliedPrecision(rNegative, rPositive, frequency, targetSharpeRatio) # calculate success rate threshold
	risk = cdf(Normal(p, p*(1 - p)), thresholdP) # approximate to bootstrap

	return risk
end

function calculateStrategyRisk(μ1, # mean of the first gaussian distribution to generate bet outcomes
							   μ2, # mean of the second gaussian distribution to generate bet outcomes
							   σ1, # standard deviation of the first gaussian distribution to generate bet outcomes
							   σ2, # standard deviation of the second gaussian distribution to generate bet outcomes
							   probability, # probability of success
							   nObs, # number of observations
							   frequency, # number of bets per year
							   targetSharpeRatio) # target annual Sharpe ratio
							   #1) Parameters

    returns = mixGaussians(μ1, μ2, σ1, σ2, probability, nObs) # 2) Generate sample from mixture

    probabilityFail = failureProbability(returns, frequency, targetSharpeRatio) # 3) Compute failure probability
    print("Prob strategy will fail ", probabilityFail) # print result

    return probabilityFail
end