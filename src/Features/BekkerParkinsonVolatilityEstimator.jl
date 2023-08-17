using Statistics

"""
    sigmaEstimates(
        beta::Vector{Float64},
        gamma::Vector{Float64}
    )::Vector{Float64}

Estimate the volatility σ using the Bekker-Parkinson method.

This function calculates the volatility estimates using the Bekker-Parkinson method based on the
formulas presented in De Prado (2018), Advances in Financial Machine Learning, page 286, snippet 19.2.

# Parameters
- `beta::Vector{Float64}`: β estimates vector.
- `gamma::Vector{Float64}`: γ estimates vector.

# Returns
- `Vector{Float64}`: Estimated volatility σ.

# Formulas
σ = \frac{\sqrt{2} - 1}{3 - 2\sqrt{2}} \beta^{0.5} + \left(\frac{\gamma}{{(k2^2 * (3 - 2\sqrt{2}))}}\right)^{0.5}

where k2 = \sqrt{\frac{8}{\pi}}

# References
- De Prado, M. (2018), Advances in Financial Machine Learning, page 286, snippet 19.2.
"""
function sigmaEstimates(
        beta::Vector{Float64},
        gamma::Vector{Float64}
    )::Vector{Float64}
    
    k2 = sqrt(8 / π)
    denominator = 3 - 2 * sqrt(2)
    sigma = (sqrt(2) - 1) .* (beta .^ 0.5) ./ denominator
    sigma .+= (gamma / (k2^2 * denominator)) .^ 0.5
    sigma[isless.(sigma, 0)] .= 0

    return sigma
end

"""
    bekkerParkinsonVolatilityEstimates(
        highPrices::Vector{Float64},
        lowPrices::Vector{Float64},
        windowSpan::Int = 20
    )::Vector{Float64}

Estimate the volatility using the Bekker-Parkinson method.

This function calculates the volatility estimates using the Bekker-Parkinson method based on the
formulas presented in De Prado (2018), Advances in Financial Machine Learning, page 286, Corwin and Schultz section.

# Parameters
- `highPrices::Vector{Float64}`: Vector of high prices.
- `lowPrices::Vector{Float64}`: Vector of low prices.
- `windowSpan::Int`: Rolling window span. Default is 20.

# Returns
- `Vector{Float64}`: Estimated volatility.

# References
- De Prado, M. (2018), Advances in Financial Machine Learning, page 286, Corwin and Schultz section.
"""
function bekkerParkinsonVolatilityEstimates(
        highPrices::Vector{Float64},
        lowPrices::Vector{Float64},
        windowSpan::Int = 20
    )::Vector{Float64}

    beta = betaEstimates(highPrices, lowPrices, windowSpan)
    gamma = gammaEstimates(highPrices, lowPrices)

    return sigmaEstimates(beta, gamma)
end
