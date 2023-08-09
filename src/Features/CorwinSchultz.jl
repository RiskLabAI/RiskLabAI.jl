using ShiftedArrays: lag, lead
using Statistics: cov, mean
using TimeSeries

"""
    Corwin and Schultz β Estimation

    Estimates β values based on high and low prices.

    Parameters:
    - high_prices::Vector: Vector of high prices.
    - low_prices::Vector: Vector of low prices.
    - window_span::Int: Rolling window span.

    Returns:
    - β::Vector: Estimated β values.
"""
function beta_estimates(
    high_prices::Vector, 
    low_prices::Vector, 
    window_span::Int
)::Vector
    log_ratios = log.(high_prices ./ low_prices) .^ 2
    β = rolling(sum, log_ratios, 2)
    β = rolling(mean, β, window_span)
    return β
end

"""
    Corwin and Schultz γ Estimation

    Estimates γ values based on high and low prices.

    Parameters:
    - high_prices::Vector: Vector of high prices.
    - low_prices::Vector: Vector of low prices.

    Returns:
    - γ::Vector: Estimated γ values.
"""
function gamma_estimates(
    high_prices::Vector, 
    low_prices::Vector
)::Vector
    high_prices_max = rolling(maximum, high_prices, 2)
    low_prices_min = rolling(minimum, low_prices, 2)
    γ = log.(high_prices_max ./ low_prices_min) .^ 2
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
function alpha_estimates(
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
    - high_prices::Vector: Vector of high prices.
    - low_prices::Vector: Vector of low prices.
    - window_span::Int: Rolling window span.

    Returns:
    - spread::Vector: Estimated spread values.
"""
function corwin_schultz_estimator(
    high_prices::Vector, 
    low_prices::Vector; 
    window_span::Int=20
)::Vector
    β = beta_estimates(high_prices, low_prices, window_span)
    γ = gamma_estimates(high_prices, low_prices)
    α = alpha_estimates(β, γ)
    spread = 2 .* (α .- 1) ./ (1 .+ exp.(α))
    return spread
end
