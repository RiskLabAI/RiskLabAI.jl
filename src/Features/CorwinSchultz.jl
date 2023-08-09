using ShiftedArrays: lag, lead
using Statistics: cov, mean
using TimeSeries

"""
    Corwin and Schultz β Estimation

    Estimates β values based on high and low prices.

    Parameters:
    - highPrices::Vector: Vector of high prices.
    - lowPrices::Vector: Vector of low prices.
    - windowSpan::Int: Rolling window span.

    Returns:
    - β::Vector: Estimated β values.
"""
function betaEstimates(
    highPrices::Vector, 
    lowPrices::Vector, 
    windowSpan::Int
)::Vector
    logRatios = log.(highPrices ./ lowPrices) .^ 2
    β = rolling(sum, logRatios, 2)
    β = rolling(mean, β, windowSpan)
    return β
end

"""
    Corwin and Schultz γ Estimation

    Estimates γ values based on high and low prices.

    Parameters:
    - highPrices::Vector: Vector of high prices.
    - lowPrices::Vector: Vector of low prices.

    Returns:
    - γ::Vector: Estimated γ values.
"""
function gammaEstimates(
    highPrices::Vector, 
    lowPrices::Vector
)::Vector
    highPricesMax = rolling(maximum, highPrices, 2)
    lowPricesMin = rolling(minimum, lowPrices, 2)
    γ = log.(highPricesMax ./ lowPricesMin) .^ 2
    γ
end

"""
    Corwin and Schultz α Estimation

    Estimates α values based on β and γ values.

    Parameters:
    - β::Vector: β Estimates vector.
    - γ::Vector: γ Estimates vector.

    Returns:
    - α::Vector: Estimated α values.
"""
function alphaEstimates(
    β::Vector, 
    γ::Vector
)::Vector
    denominator = 3 - 2 * 2^0.5
    α = (2^0.5 - 1) .* (β .^ 0.5) ./ denominator
    α .-= (γ ./ denominator) .^ 0.5
    α[isless.(α, 0)] .= 0 # Set negative alphas to 0 (see p.727 of paper)
    return α
end

"""
    Corwin and Schultz spread estimator

    Estimates spread values based on α values.

    Parameters:
    - highPrices::Vector: Vector of high prices.
    - lowPrices::Vector: Vector of low prices.
    - windowSpan::Int: Rolling window span.

    Returns:
    - spread::Vector: Estimated spread values.
"""
function corwinSchultzEstimator(
    highPrices::Vector, 
    lowPrices::Vector; 
    windowSpan::Int=20
)::Vector
    β = betaEstimates(highPrices, lowPrices, windowSpan)
    γ = gammaEstimates(highPrices, lowPrices)
    α = alphaEstimates(β, γ)
    spread = 2 .* (α .- 1) ./ (1 .+ exp.(α))
    return spread
end
