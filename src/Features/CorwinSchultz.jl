include("Utils/RollingFunctions.jl")
include("Utils/VectorFunctions.jl")

import .RollingFunctions: rolling
import .VectorFunctions: differences
import ShiftedArrays: lag, lead
import Statistics: cov, mean
using TimeSeries


"""
    function: Corwin and Schultz β Estimation 
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 285 snippet 19.1
"""
function βEstimates(
    highPrices::Vector, # high prices vector
    lowPrices::Vector, # low prices vector
    windowSpan::Int # rolling window span
)::Vector

    logRatios = log.(highPrices ./ lowPrices) .^ 2
    β = rolling(sum, logRatios, 2)
    β = rolling(mean, β, windowSpan)
    return β
end



"""
    function: Corwin and Schultz γ Estimation 
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 285 snippet 19.1
"""
function γEstimates(
    highPrices::Vector, # high prices vector
    lowPrices::Vector, # low prices vector
)::Vector

    highPricesMax = rolling(maximum, highPrices, 2)
    lowPricesMin = rolling(minimum, lowPrices, 2)
    γ = log.(highPricesMax ./ lowPricesMin) .^ 2
    γ
end

"""
    function: Corwin and Schultz α Estimation 
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 285 snippet 19.1
"""
function αEstimates(
    β::Vector, # β Estimates vector
    γ::Vector, # γ Estimates vector
)::Vector

    denominator = 3 - 2 * 2^0.5
    α = (2^0.5 - 1) .* (β .^ 0.5) ./ denominator
    α .-= (γ ./ denominator) .^ 0.5
    # .< does'nt work with missing values
    α[isless.(α, 0)] .= 0 # set negative alphas to 0 (see p.727 of paper)

    return α
end


"""
    function: Corwin and Schultz spread estimator 
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 285 snippet 19.1
"""
function corwinSchultzEstimator(
    highPrices::Vector, # high prcies vector
    lowPrices::Vector; # low prices vector
    windowSpan::Int=20 # rolling window span
)::Vector

    # Note: S<0 iif α < 0
    β = βEstimates(highPrices, lowPrices, windowSpan)
    γ = γEstimates(highPrices, lowPrices)
    α = αEstimates(β, γ)

    spread = 2 .* (α .- 1) ./ (1 .+ exp.(α))

    return spread
end