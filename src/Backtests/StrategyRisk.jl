using Distributions
using SymPy
using Statistics

"""
    sharpeRatioTrials(p::Float64, nRun::Int)::Tuple{Float64, Float64, Float64}

Calculates the Sharpe ratio trials.

# Arguments
- `p::Float64`: Probability of success.
- `nRun::Int`: Number of runs.

# Returns
- `Tuple{Float64, Float64, Float64}`: Tuple containing mean, standard deviation, and Sharpe ratio.
"""
function sharpeRatioTrials(p::Float64, nRun::Int)::Tuple{Float64, Float64, Float64}
    b = Binomial(1, p)
    results = [rand(b) == 1 ? 1.0 : -1.0 for _ in 1:nRun]
    
    return (mean(results), std(results), mean(results) / std(results))
end

"""
    targetSharpeRatioSymbolic()::Symbolic expression

Computes the target Sharpe ratio using symbolic operations.

# Returns
- `Symbolic expression`: Symbolic expression for the target Sharpe ratio.
"""
function targetSharpeRatioSymbolic()::SymPy.SymbolicObject
    @vars p u d
    
    m2 = p * u^2 + (1 - p) * d^2
    m1 = p * u + (1 - p) * d
    v = m2 - m1^2
    
    return factor(v)
end

"""
    impliedPrecision(
        stopLoss::Float64,
        profitTaking::Float64,
        frequency::Int,
        targetSharpeRatio::Float64
    )::Float64

Computes implied precision.

# Arguments
- `stopLoss::Float64`: Stop loss threshold.
- `profitTaking::Float64`: Profit taking threshold.
- `frequency::Int`: Number of bets per year.
- `targetSharpeRatio::Float64`: Target annual Sharpe ratio.

# Returns
- `Float64`: Implied precision.
"""
function impliedPrecision(
    stopLoss::Float64,
    profitTaking::Float64,
    frequency::Int,
    targetSharpeRatio::Float64
)::Float64
    a = (frequency + targetSharpeRatio^2) * (profitTaking - stopLoss)^2
    b = (2 * frequency * stopLoss - targetSharpeRatio^2 * (profitTaking - stopLoss)) * (profitTaking - stopLoss)
    c = frequency * stopLoss^2
    precision = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
    
    return precision
end

"""
    binFrequency(
        stopLoss::Float64,
        profitTaking::Float64,
        precision::Float64,
        targetSharpeRatio::Float64
    )::Union{Float64, Nothing}

Computes the number of bets/year needed to achieve a Sharpe ratio with a certain precision rate.

# Arguments
- `stopLoss::Float64`: Stop loss threshold.
- `profitTaking::Float64`: Profit taking threshold.
- `precision::Float64`: Precision rate p.
- `targetSharpeRatio::Float64`: Target annual Sharpe ratio.

# Returns
- `Union{Float64, Nothing}`: Number of bets per year or nothing if the approximation is not within the tolerance.
"""
function binFrequency(
    stopLoss::Float64,
    profitTaking::Float64,
    precision::Float64,
    targetSharpeRatio::Float64
)::Union{Float64, Nothing}
    frequency = (targetSharpeRatio * (profitTaking - stopLoss))^2 * precision * (1 - precision) /
        ((profitTaking - stopLoss) * precision + stopLoss)^2
    
    binSR(sl0, pt0, freq0, p0) =
        (((pt0 - sl0) * p0 + sl0) * sqrt(freq0)) / ((pt0 - sl0) * sqrt(p0 * (1 - p0)))
    
    if !isapprox(binSR(stopLoss, profitTaking, frequency, precision), targetSharpeRatio, atol = 0.5)
        return nothing
    end
    
    return frequency
end

"""
    failureProbability(
        returns::Vector{Float64},
        frequency::Int,
        targetSharpeRatio::Float64
    )::Float64

Calculates the strategy risk in practice.

# Arguments
- `returns::Vector{Float64}`: Returns of the strategy.
- `frequency::Int`: Number of bets per year.
- `targetSharpeRatio::Float64`: Target annual Sharpe ratio.

# Returns
- `Float64`: Probability that the strategy may fail.
"""
function failureProbability(
    returns::Vector{Float64},
    frequency::Int,
    targetSharpeRatio::Float64
)::Float64
    rPositive, rNegative = mean(returns[returns .> 0]), mean(returns[returns .<= 0])
    p = count(returns .> 0) / length(returns)
    thresholdP = impliedPrecision(rNegative, rPositive, frequency, targetSharpeRatio)
    risk = cdf(Normal(p, sqrt(p * (1 - p))), thresholdP)
    
    return risk
end
using Distributions
using Statistics

"""
    calculateStrategyRisk(
        mean1::Float64,
        mean2::Float64,
        stdDev1::Float64,
        stdDev2::Float64,
        probability::Float64,
        numberOfObservations::Int,
        frequency::Int,
        targetSharpeRatio::Float64
    )::Float64

Calculates the strategy risk in practice.

This function calculates the probability that the strategy will fail given the parameters of the mixed Gaussian distribution that generates bet outcomes, the number of observations, the number of bets per year, and the target annual Sharpe ratio.

# Arguments
- `mean1::Float64`: Mean of the first Gaussian distribution to generate bet outcomes.
- `mean2::Float64`: Mean of the second Gaussian distribution to generate bet outcomes.
- `stdDev1::Float64`: Standard deviation of the first Gaussian distribution to generate bet outcomes.
- `stdDev2::Float64`: Standard deviation of the second Gaussian distribution to generate bet outcomes.
- `probability::Float64`: Probability of success.
- `numberOfObservations::Int`: Number of observations.
- `frequency::Int`: Number of bets per year.
- `targetSharpeRatio::Float64`: Target annual Sharpe ratio.

# Returns
- `Float64`: Probability that the strategy may fail.

# Related mathematical formulae
The function computes the probability of failure based on the probability distribution function of a mixed Gaussian distribution. It calculates the implied precision required to achieve the target Sharpe ratio and uses the cumulative distribution function to find the probability that the strategy will fail.

.. math::
    p_{fail} = CDF(Normal(p, \sqrt{p(1-p)}), thresholdP)
"""
function calculateStrategyRisk(
    mean1::Float64,
    mean2::Float64,
    stdDev1::Float64,
    stdDev2::Float64,
    probability::Float64,
    numberOfObservations::Int,
    frequency::Int,
    targetSharpeRatio::Float64
)::Float64
    # Note: The mixGaussians function was not provided. If you can provide it, I can include it in this code.
    returns = mixGaussians(mean1, mean2, stdDev1, stdDev2, probability, numberOfObservations)
    probabilityFail = failureProbability(returns, frequency, targetSharpeRatio)
    
    println("Probability strategy will fail: ", probabilityFail)
    return probabilityFail
end
