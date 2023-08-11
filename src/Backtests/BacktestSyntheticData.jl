using Distributions
using Statistics
using DataFrames
using GLM
using LinearAlgebra
using PlotlyJS

"""
Function to perform backtesting with synthetic data.

This function implements the methodology from De Prado's book "Advances in Financial Machine Learning" (p.175 snippet 13.1).

Args:
    forecast::Float64: Long run price.
    halfLife::Float64: Half life of the model.
    σ::Float64: Standard deviation used in the model.
    maximumIteration::Int: Maximum number of iterations.
    maximumHoldingPeriod::Int: Maximum holding period.
    profitTakingRange::LinRange: Profit taking range.
    stopLossRange::LinRange: Stop loss range.
    seed::Float64: Starting price.

Returns:
    Array{Float64, 2}: Resulting Sharpe ratio values.
"""
function syntheticBacktesting(
    forecast::Float64,
    halfLife::Float64,
    σ::Float64;
    maximumIteration::Int = 1e3,
    maximumHoldingPeriod::Int = 100,
    profitTakingRange::LinRange = LinRange(0.5, 10, 20),
    stopLossRange::LinRange = LinRange(0.5, 10, 20),
    seed::Float64 = 0.0
)
    ϕ = 2^(-1 / halfLife)
    output = zeros(Float64, length(profitTakingRange), length(stopLossRange))
    standardNormalDistribution = Normal()

    for (iIdx, i) in enumerate(profitTakingRange)
        for (jIdx, j) in enumerate(stopLossRange)
            output2 = Float64[]
            for _ in 1:maximumIteration
                price, holdingPeriod = seed, 0
                while true
                    r = rand(standardNormalDistribution)
                    price = (1 - ϕ) * forecast + ϕ * price + σ * r
                    priceDifference = price - seed
                    holdingPeriod += 1
                    if priceDifference > i * σ || priceDifference < -j * σ || holdingPeriod > maximumHoldingPeriod
                        push!(output2, priceDifference)
                        break
                    end
                end
            end
            meanVal, stdVal = mean(output2), std(output2)
            output[iIdx, jIdx] = meanVal / stdVal
        end
    end
    return output
end

"""
Function to fit an Ornstein-Uhlenbeck (O-U) process on data.

This function implements the methodology from De Prado's book "Advances in Financial Machine Learning" (p.173).

Args:
    price::Vector{Float64}: Vector of stock prices.

Returns:
    Tuple{Float64, Float64, Float64}: Coefficients ρ, future, and σ of the fitted O-U process.
"""
function fitOuProcess(price::Vector{Float64})
    data = DataFrame(
        Y = price[2:end] .- price[1:end-1],
        X = price[1:end-1]
    )
    ols = lm(@formula(Y ~ X), data)
    ρ = GLM.coef(ols)[2] + 1
    future = GLM.coef(ols)[1] / (1 - ρ)
    σ = std(data[!,:Y] .- GLM.coef(ols)[1] .- GLM.coef(ols)[2] * data[!,:X])
    return ρ, future, σ
end

"""
Function to simulate an Ornstein-Uhlenbeck (O-U) process on data.

Args:
    ρ::Float64: Coefficient related to half-life.
    future::Float64: Long run price.
    σ::Float64: Standard deviation of the model.
    p0::Float64: Starting price.
    periodLength::Int: Number of days to simulate.

Returns:
    Vector{Float64}: Simulated prices.
"""
function simulateOuProcess(
    ρ::Float64,
    future::Float64,
    σ::Float64,
    p0::Float64,
    periodLength::Int
)
    price = Vector{Float64}(undef, periodLength)
    price[1] = p0
    standardNormalDistribution = Normal()

    for i in 2:periodLength
        r = rand(standardNormalDistribution)
        price[i] = (1 - ρ) * future + ρ * price[i - 1] + σ * r
    end
    return price
end

"""
Function to perform backtesting with synthetic data for specific prices.

Args:
    price::Vector{Float64}: Vector of stock prices.
    maximumIteration::Int: Maximum number of iterations.
    maximumHoldingPeriod::Int: Maximum holding period.
    profitTakingRange::LinRange: Profit taking range.
    stopLossRange::LinRange: Stop loss range.
    seed::Float64: Starting price.

Returns:
    Array{Float64, 2}: Resulting Sharpe ratio values.
"""
function syntheticBacktesting(
    price::Vector{Float64};
    maximumIteration::Int = 1e5,
    maximumHoldingPeriod::Int = 100,
    profitTakingRange::LinRange = LinRange(0.5, 10, 20),
    stopLossRange::LinRange = LinRange(0.5, 10, 20),
    seed::Float64 = 0.0
)
    ρ, future, σ = fitOuProcess(price)
    out = syntheticBacktesting(
        future, -1.0 / log2(ρ), σ;
        maximumHoldingPeriod = maximumHoldingPeriod,
        maximumIteration = maximumIteration,
        seed = seed
    )
    
    p = heatmap(
        profitTakingRange,
        stopLossRange,
        transpose(out),
        c = cgrad([:black, :white]),
        xlabel = "Profit-Taking",
        ylabel = "Stop-Loss",
        title = "Forecast = $(round(future, digits = 3)) | H-L=$(round((-1.0 / log2(ρ)), digits = 3)) | Sigma=$(round(σ, digits = 3))"
    )
    Plots.display(p)
    return out
end
