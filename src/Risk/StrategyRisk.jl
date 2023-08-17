using Distributions
using Random
using Statistics
using SymPy

"""
    sharpeRatioTrials(p::Float64, nRun::Int)::Tuple{Float64, Float64, Float64}

Calculate the Sharpe ratio as a function of the number of bets.

# Arguments
- `p::Float64`: Probability of success for each bet.
- `nRun::Int`: Number of bets.

# Returns
- `Tuple{Float64, Float64, Float64}`: A tuple containing the mean, standard deviation, and Sharpe ratio of the bets.

# Mathematical Formulae
The Sharpe ratio is calculated as the ratio of the mean to the standard deviation of the returns:
\[
\text{Sharpe Ratio} = \frac{\mu}{\sigma}
\]
where \(\mu\) is the mean and \(\sigma\) is the standard deviation.

# Reference
De Prado, M. (2018) Advances in financial machine learning. Page 213, Snippet 15.1.
"""
function sharpeRatioTrials(
    p::Float64,
    nRun::Int
)::Tuple{Float64, Float64, Float64}
    betResults = [rand(Binomial(1, p)) == 1 ? 1 : -1 for _ in 1:nRun]
    meanValue = mean(betResults)
    stdValue = std(betResults)
    return (meanValue, stdValue, meanValue / stdValue)
end

"""
    targetSharpeRatioSymbolic()::Sym

Use the SymPy library for symbolic operations.

# Returns
- `Sym`: A symbolic expression for the variance of the Sharpe ratio.

# Mathematical Formulae
The variance \(v\) is calculated as:
\[
v = p \cdot u^2 + (1 - p) \cdot d^2 - (p \cdot u + (1 - p) \cdot d)^2
\]
where \(p\) is the probability of success, \(u\) is the return on success, and \(d\) is the return on failure.

# Reference
De Prado, M. (2018) Advances in financial machine learning. Page 214, Snippet 15.2.
"""
function targetSharpeRatioSymbolic()::Sym
    @vars p u d
    m1 = p * u + (1 - p) * d
    m2 = p * u^2 + (1 - p) * d^2
    v = m2 - m1^2
    return factor(v)
end

"""
    impliedPrecision(
        stopLoss::Float64,
        profitTaking::Float64,
        freq::Float64,
        targetSharpeRatio::Float64
    )::Float64

Compute the implied precision required to achieve a certain target Sharpe ratio.

# Arguments
- `stopLoss::Float64`: The stop-loss level for the strategy.
- `profitTaking::Float64`: The profit-taking level for the strategy.
- `freq::Float64`: The frequency of bets or trades.
- `targetSharpeRatio::Float64`: The desired Sharpe ratio.

# Returns
- `Float64`: The implied precision required to achieve the target Sharpe ratio.

# Mathematical Formulae
The implied precision \( p \) is calculated by solving the quadratic equation:
\[
a \cdot p^2 + b \cdot p + c = 0
\]
where
\[
a = (f + t^2) \cdot (P - S)^2
\]
\[
b = (2 \cdot f \cdot S - t^2 \cdot (P - S)) \cdot (P - S)
\]
\[
c = f \cdot S^2
\]
where \( f \) is the frequency of bets, \( t \) is the target Sharpe ratio, \( P \) is the profit-taking level, and \( S \) is the stop-loss level.

# Reference
De Prado, M. (2018) Advances in financial machine learning. Page 214, Snippet 15.3.
"""
function impliedPrecision(
    stopLoss::Float64,
    profitTaking::Float64,
    freq::Float64,
    targetSharpeRatio::Float64
)::Float64
    a = (freq + targetSharpeRatio^2) * (profitTaking - stopLoss)^2
    b = (2 * freq * stopLoss - targetSharpeRatio^2 * (profitTaking - stopLoss)) * (profitTaking - stopLoss)
    c = freq * stopLoss^2
    return (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
end

"""
    calculateBetsPerYear(
        stopLoss::Float64,
        profitTaking::Float64,
        precision::Float64,
        targetSharpeRatio::Float64
    )::Union{Float64, Nothing}

Compute the number of bets per year needed to achieve a target Sharpe ratio with a certain precision rate.

# Arguments
- `stopLoss::Float64`: The stop-loss level for the strategy.
- `profitTaking::Float64`: The profit-taking level for the strategy.
- `precision::Float64`: The precision rate of the strategy.
- `targetSharpeRatio::Float64`: The desired Sharpe ratio.

# Returns
- `Union{Float64, Nothing}`: The number of bets per year required to achieve the target Sharpe ratio, or `nothing` if it's not achievable.

# Mathematical Formulae
The number of bets per year \( f \) is calculated as follows:
\[
f = \frac{t^2 \cdot (P - S)^2 \cdot p \cdot (1 - p)}{(P - S) \cdot p + S)^2}
\]
where \( t \) is the target Sharpe ratio, \( P \) is the profit-taking level, \( S \) is the stop-loss level, and \( p \) is the precision rate.

# Reference
De Prado, M. (2018) Advances in financial machine learning. Page 215, Snippet 15.4.
"""
function calculateBetsPerYear(
    stopLoss::Float64,
    profitTaking::Float64,
    precision::Float64,
    targetSharpeRatio::Float64
)::Union{Float64, Nothing}
    frequency = (targetSharpeRatio * (profitTaking - stopLoss))^2 * precision * (1 - precision) / ((profitTaking - stopLoss) * precision + stopLoss)^2
    binSr(sl::Float64, pt::Float64, freq::Float64, p::Float64) = (((pt - sl) * p + sl) * freq^0.5) / ((pt - sl) * (p * (1 - p))^0.5)

    if !isapprox(binSr(stopLoss, profitTaking, frequency, precision), targetSharpeRatio, atol = 0.5)
        return nothing
    end

    return frequency
end

using Random
using Distributions

"""
    calculateStrategyRisk(
        μ1::Float64,
        μ2::Float64,
        σ1::Float64,
        σ2::Float64,
        probability1::Float64,
        nObs::Int
    )::Float64

Calculate the strategy risk in practice using a mixture of Gaussian distributions.

# Arguments
- `μ1::Float64`: Mean of the first Gaussian distribution.
- `μ2::Float64`: Mean of the second Gaussian distribution.
- `σ1::Float64`: Standard deviation of the first Gaussian distribution.
- `σ2::Float64`: Standard deviation of the second Gaussian distribution.
- `probability1::Float64`: Probability of selecting the first Gaussian distribution.
- `nObs::Int`: Number of observations.

# Returns
- `Float64`: The calculated returns from the mixture of Gaussian distributions.

# Reference
De Prado, M. (2018) Advances in financial machine learning. Page 215, Snippet 15.4.
"""
function calculateStrategyRisk(
    μ1::Float64,
    μ2::Float64,
    σ1::Float64,
    σ2::Float64,
    probability1::Float64,
    nObs::Int
)::Float64
    return1 = rand(Normal(μ1, σ1), trunc(Int, nObs * probability1))
    return2 = rand(Normal(μ2, σ2), trunc(Int, nObs) - trunc(Int, nObs * probability1))
    returns = append!(return1, return2)
    shuffle!(returns)
    return returns
end
using Distributions
using Statistics

"""
    failureProbability(
        returns::Vector{Float64},
        freq::Float64,
        targetSharpeRatio::Float64
    )::Float64

Calculate the probability of failure of a strategy based on returns.

# Arguments
- `returns::Vector{Float64}`: Returns of the strategy.
- `freq::Float64`: Frequency parameter.
- `targetSharpeRatio::Float64`: Target Sharpe ratio.

# Returns
- `Float64`: The calculated probability of failure.

# Reference
De Prado, M. (2018) Advances in financial machine learning. Page 215, Snippet 15.4.
"""
function failureProbability(
    returns::Vector{Float64},
    freq::Float64,
    targetSharpeRatio::Float64
)::Float64
    rPositive = mean(returns[returns .> 0])
    rNegative = mean(returns[returns .<= 0])
    p = count(returns .> 0) / length(returns)
    thresholdP = impliedPrecision(rNegative, rPositive, freq, targetSharpeRatio)
    risk = cdf(Normal(p, p * (1 - p)), thresholdP)
    return risk
end

using Distributions
using Random
using Statistics

"""
    calculateStrategyRisk(
        μ1::Float64,
        μ2::Float64,
        σ1::Float64,
        σ2::Float64,
        probability1::Float64,
        nObs::Int,
        freq::Float64,
        targetSharpeRatio::Float64
    )::Float64

Calculate the strategy risk in practice.

# Arguments
- `μ1::Float64`: Mean of the first Gaussian distribution.
- `μ2::Float64`: Mean of the second Gaussian distribution.
- `σ1::Float64`: Standard deviation of the first Gaussian distribution.
- `σ2::Float64`: Standard deviation of the second Gaussian distribution.
- `probability1::Float64`: Probability parameter for the first Gaussian distribution.
- `nObs::Int`: Number of observations.
- `freq::Float64`: Frequency parameter.
- `targetSharpeRatio::Float64`: Target Sharpe ratio.

# Returns
- `Float64`: The calculated probability of strategy failure.

# Reference
De Prado, M. (2018) Advances in financial machine learning. Page 215, Snippet 15.4.
"""
function calculateStrategyRisk(
    μ1::Float64,
    μ2::Float64,
    σ1::Float64,
    σ2::Float64,
    probability1::Float64,
    nObs::Int,
    freq::Float64,
    targetSharpeRatio::Float64
)::Float64
    returns = mixGaussians(μ1, μ2, σ1, σ2, probability1, nObs)
    probabilityFail = failureProbability(returns, freq, targetSharpeRatio)
    println("Probability strategy will fail: ", probabilityFail)
    return probabilityFail
end
