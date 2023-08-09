"""
Function to calculate the Sharpe ratio trials.

:param p: Probability of success.
:param nRun: Number of runs.
:return: Tuple containing mean, standard deviation, and Sharpe ratio.
"""
function sharpeRatioTrials(p::Float64, nRun::Int64)::Tuple{Float64, Float64, Float64}
    result = []
    
    for i in 1:nRun
        b = Binomial(1, p)
        random = rand(b, 1)
        
        if random[1] == 1
            x = 1
        else
            x = -1
        end
        
        push!(result, [x])
    end
    
    return (mean(result), std(result), mean(result) / std(result))
end

"""
Function to compute the target Sharpe ratio using symbolic operations.

:return: Symbolic expression for the target Sharpe ratio.
"""
function targetSharpeRatioSymbolic()
    p, u, d = symbols("p u d")
    
    m2 = p * u^2 + (1 - p) * d^2
    m1 = p * u + (1 - p) * d
    v = m2 - m1^2
    
    return factor(v)
end

"""
Function to compute implied precision.

:param stopLoss: Stop loss threshold.
:param profitTaking: Profit taking threshold.
:param frequency: Number of bets per year.
:param targetSharpeRatio: Target annual Sharpe ratio.
:return: Implied precision.
"""
function impliedPrecision(stopLoss::Float64, profitTaking::Float64, frequency::Int64, targetSharpeRatio::Float64)::Float64
    a = (frequency + targetSharpeRatio^2) * (profitTaking - stopLoss)^2
    b = (2 * frequency * stopLoss - targetSharpeRatio^2 * (profitTaking - stopLoss)) * (profitTaking - stopLoss)
    c = frequency * stopLoss^2
    precision = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
    
    return precision
end

"""
Function to compute the number of bets/year needed to achieve a Sharpe ratio with a certain precision rate.

:param stopLoss: Stop loss threshold.
:param profitTaking: Profit taking threshold.
:param precision: Precision rate p.
:param targetSharpeRatio: Target annual Sharpe ratio.
:return: Number of bets per year.
"""
function binFrequency(stopLoss, profitTaking, precision, targetSharpeRatio)
    frequency = (targetSharpeRatio * (profitTaking - stopLoss))^2 * precision * (1 - precision) /
        ((profitTaking - stopLoss) * precision + stopLoss)^2
    
    binSR(sl0, pt0, freq0, p0) =
        (((pt0 - sl0) * p0 + sl0) * freq0^0.5) / ((pt0 - sl0) * (p0 * (1 - p0))^0.5)
    
    if !isapprox(binSR(stopLoss, profitTaking, frequency, precision), targetSharpeRatio, atol = 0.5)
        return nothing
    end
    
    return frequency
end

"""
Function to calculate the strategy risk in practice.

:param μ1: Mean of the first Gaussian distribution to generate bet outcomes.
:param μ2: Mean of the second Gaussian distribution to generate bet outcomes.
:param σ1: Standard deviation of the first Gaussian distribution to generate bet outcomes.
:param σ2: Standard deviation of the second Gaussian distribution to generate bet outcomes.
:param probability: Probability of success.
:param nObs: Number of observations.
:param frequency: Number of bets per year.
:param targetSharpeRatio: Target annual Sharpe ratio.
:return: Probability that the strategy may fail.
"""
function failureProbability(returns, frequency, targetSharpeRatio)
    rPositive, rNegative = mean(returns[returns .> 0]), mean(returns[returns .<= 0])
    p = count(returns .> 0) / length(returns)
    thresholdP = impliedPrecision(rNegative, rPositive, frequency, targetSharpeRatio)
    risk = cdf(Normal(p, p * (1 - p)), thresholdP)
    
    return risk
end

"""
Function to calculate the strategy risk in practice.

:param μ1: Mean of the first Gaussian distribution to generate bet outcomes.
:param μ2: Mean of the second Gaussian distribution to generate bet outcomes.
:param σ1: Standard deviation of the first Gaussian distribution to generate bet outcomes.
:param σ2: Standard deviation of the second Gaussian distribution to generate bet outcomes.
:param probability: Probability of success.
:param nObs: Number of observations.
:param frequency: Number of bets per year.
:param targetSharpeRatio: Target annual Sharpe ratio.
:return: Probability that the strategy may fail.
"""
function calculateStrategyRisk(μ1, μ2, σ1, σ2, probability, nObs, frequency, targetSharpeRatio)
    returns = mixGaussians(μ1, μ2, σ1, σ2, probability, nObs)
    probabilityFail = failureProbability(returns, frequency, targetSharpeRatio)
    
    println("Probability strategy will fail: ", probabilityFail)
    return probabilityFail
end
