include("CorwinSchultz.jl")

"""
Estimates the volatility σ using the Bekker-Parkinson method.

This function calculates the volatility estimates using the Bekker-Parkinson method based on the
formulas presented in De Prado (2018), Advances in Financial Machine Learning, page 286, snippet 19.2.

Parameters:
- β (Vector): β estimates vector.
- γ (Vector): γ estimates vector.

Returns:
- Vector: Estimated volatility σ.
"""
function σ_estimates(
    β::Vector,
    γ::Vector
)::Vector
    k2 = (8 / π)^0.5
    denominator = 3 - 2 * (2^0.5)
    σ = (2^0.5 - 1) .* (β .^ 0.5) ./ denominator
    σ .+= (γ / (k2^2 * denominator)) .^ 0.5
    # .< doesn't work with missing values
    σ[isless.(σ, 0)] .= 0

    return σ
end

"""
Estimates the volatility using the Bekker-Parkinson method.

This function calculates the volatility estimates using the Bekker-Parkinson method based on the
formulas presented in De Prado (2018), Advances in Financial Machine Learning, page 286, Corwin and Schultz section.

Parameters:
- highPrices (Vector): Vector of high prices.
- lowPrices (Vector): Vector of low prices.
- windowSpan (Int, optional): Rolling window span. Default is 20.

Returns:
- Vector: Estimated volatility.
"""
function bekker_parkinson_volatility_estimates(
    highPrices::Vector,
    lowPrices::Vector;
    windowSpan::Int=20
)::Vector

    β = β_estimates(highPrices, lowPrices, windowSpan)
    γ = γ_estimates(highPrices, lowPrices)

    σ_estimates(β, γ)
end
