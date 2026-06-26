using RollingFunctions
using Statistics

"""
    betaEstimates(
        highPrices::Vector{Float64}, 
        lowPrices::Vector{Float64}, 
        windowSpan::Int
    ) -> Vector{Float64}

Estimate β values based on high and low prices using the Corwin and Schultz method.

- `highPrices` (Vector{Float64}): Vector of high prices.
- `lowPrices` (Vector{Float64}): Vector of low prices.
- `windowSpan` (Int): Rolling window span.

Returns:
- Vector{Float64}: Estimated β values.
"""
function betaEstimates(
    highPrices::Vector{Float64}, 
    lowPrices::Vector{Float64}, 
    windowSpan::Int
)::Vector{Float64}
    logRatios = log.(highPrices ./ lowPrices) .^ 2
    β = rolling(sum, logRatios, 2)
    β = rolling(mean, β, windowSpan)
    return β
end

"""
    gammaEstimates(
        highPrices::Vector{Float64}, 
        lowPrices::Vector{Float64}
    ) -> Vector{Float64}

Estimate γ values based on high and low prices using the Corwin and Schultz method.

- `highPrices` (Vector{Float64}): Vector of high prices.
- `lowPrices` (Vector{Float64}): Vector of low prices.

Returns:
- Vector{Float64}: Estimated γ values.
"""
function gammaEstimates(
    highPrices::Vector{Float64}, 
    lowPrices::Vector{Float64}
)::Vector{Float64}
    highPricesMax = rolling(maximum, highPrices, 2)
    lowPricesMin = rolling(minimum, lowPrices, 2)
    γ = log.(highPricesMax ./ lowPricesMin) .^ 2
    return γ
end

"""
    alphaEstimates(
        β::Vector{Float64}, 
        γ::Vector{Float64}
    ) -> Vector{Float64}

Estimate α values based on β and γ values using the Corwin and Schultz method.

- `β` (Vector{Float64}): β estimates vector.
- `γ` (Vector{Float64}): γ estimates vector.

Returns:
- Vector{Float64}: Estimated α values.
"""
function alphaEstimates(
    β::Vector{Float64}, 
    γ::Vector{Float64}
)::Vector{Float64}
    denominator = 3 - 2 * 2^0.5
    α = (2^0.5 - 1) .* (β .^ 0.5) ./ denominator
    α .-= (γ ./ denominator) .^ 0.5
    α[α .< 0] .= 0 # Set negative alphas to 0 (see p.727 of paper)
    return α
end

"""
    corwinSchultzEstimator(
        highPrices::Vector{Float64}, 
        lowPrices::Vector{Float64}; 
        windowSpan::Int=20
    ) -> Vector{Float64}

Estimate spread values based on α values using the Corwin and Schultz method.

- `highPrices` (Vector{Float64}): Vector of high prices.
- `lowPrices` (Vector{Float64}): Vector of low prices.
- `windowSpan` (Int, optional): Rolling window span. Default is 20.

Returns:
- Vector{Float64}: Estimated spread values.
"""
function corwinSchultzEstimator(
    highPrices::Vector{Float64}, 
    lowPrices::Vector{Float64}; 
    windowSpan::Int=20
)::Vector{Float64}
    β = betaEstimates(highPrices, lowPrices, windowSpan)
    γ = gammaEstimates(highPrices, lowPrices)
    α = alphaEstimates(β, γ)
    spread = 2 .* (α .- 1) ./ (1 .+ exp.(α))
    return spread
end
