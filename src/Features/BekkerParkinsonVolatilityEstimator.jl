include("CorwinSchultz.jl")

"""
function: Bekker-Parkinson volatility σ Estimation 
reference: De Prado, M. (18) Advances in Financial Machine Learning
methodology: page 286 snippet 19.2
"""
function σEstimates(
    β::Vector, # β Estimates vector
    γ::Vector # γ Estimates vector
)::Vector
    k2 = (8 / π)^0.5
    denominator = 3 - 2 * (2^0.5)
    σ = (2^0.5 - 1) .* (β .^ 0.5) ./ denominator
    σ .+= (γ / (k2^2 * denominator)) .^ 0.5
    # .< does'nt work with missing values
    σ[isless.(σ, 0)] .= 0

    return σ
end

"""
function: Bekker-Parkinson volatility Estimation
reference: De Prado, M. (18) Advances in Financial Machine Learning
methodology: page 286 Corwin and Schultz section 
"""
function bekkerParkinsonVolatilityEstimates(
    highPrices::Vector, # high prices vector
    lowPrices::Vector; # low prices vector
    windowSpan::Int=20 # rolling window span
)::Vector

    β = βEstimates(highPrices, lowPrices, windowSpan)
    γ = γEstimates(highPrices, lowPrices)

    σEstimates(β, γ)
end