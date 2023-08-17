using StatsBase, LinearAlgebra, Statistics

"""
    mutualInformationScore(histogramXY::Array{Int, 2})::Float64

Calculates mutual information score between two datasets.

# Arguments
- `histogramXY::Array{Int, 2}`: 2D histogram matrix of the two datasets.

# Returns
- `Float64`: Mutual information score.
"""
function mutualInformationScore(histogramXY::Array{Int, 2})
    score = 0.0
    histogramX = vec(sum(histogramXY, dims=2))
    histogramY = vec(sum(histogramXY, dims=1))
    total = sum(histogramXY)

    for i in 1:size(histogramXY)[1]
        for j in 1:size(histogramXY)[2]
            if histogramXY[i, j] != 0
                score += (histogramXY[i, j] / total) *
                         log(total * histogramXY[i, j] / (histogramX[i] * histogramY[j]))
            end
        end
    end
    
    return score
end

"""
    variationOfInformation(x::Array{Float64}, y::Array{Float64}, numberOfBins::Int; norm::Bool=false)::Float64

Calculates Variation of Information between two datasets.

# Arguments
- `x::Array{Float64}`: First dataset.
- `y::Array{Float64}`: Second dataset.
- `numberOfBins::Int`: Number of bins for discretization.
- `norm::Bool=false`: Normalize the result.

# Returns
- `Float64`: Variation of Information.
"""
function variationOfInformation(x::Array{Float64}, y::Array{Float64}, numberOfBins::Int; norm::Bool=false)
    rangeX = range(minimum(x), maximum(x), length=numberOfBins)
    rangeY = range(minimum(y), maximum(y), length=numberOfBins)
    histogramXY = fit(Histogram, (x, y), (rangeX, rangeY)).weights
    mutualInformation = mutualInformationScore(histogramXY)
    marginalX = entropy(normalize(fit(Histogram, x, rangeX).weights, 1))
    marginalY = entropy(normalize(fit(Histogram, y, rangeY).weights, 1))
    variationXY = marginalX + marginalY - 2 * mutualInformation
    
    if norm
        jointXY = marginalX + marginalY - mutualInformation
        variationXY /= jointXY
    end
    
    return variationXY
end

"""
    numberOfBins(numberObservations::Int, correlation::Union{Nothing, Float64}=nothing)::Int

Calculates the number of bins for histogram based on the number of observations and correlation.

# Arguments
- `numberObservations::Int`: Number of observations.
- `correlation::Union{Nothing, Float64}=nothing`: Correlation between two datasets.

# Returns
- `Int`: Number of bins.
"""
function numberOfBins(numberObservations::Int, correlation::Union{Nothing, Float64}=nothing)
    if isnothing(correlation)
        z = (8 + 324 * numberObservations + 12 * sqrt(36 * numberObservations + 729 * numberObservations^2))^(1/3)
        bins = round(z / 6 + 2 / (3 * z) + 1 / 3)
    else
        bins = round(2^-0.5 * sqrt(1 + sqrt(1 + 24 * numberObservations / (1 - correlation^2))))
    end

    return Int(bins)
end

"""
    variationOfInformationExtended(x::Array{Float64}, y::Array{Float64}; norm::Bool=false)::Float64

Calculates Variation of Information between two datasets while calculating the number of bins.

# Arguments
- `x::Array{Float64}`: First dataset.
- `y::Array{Float64}`: Second dataset.
- `norm::Bool=false`: Normalize the result.

# Returns
- `Float64`: Variation of Information.
"""
function variationOfInformationExtended(x::Array{Float64}, y::Array{Float64}; norm::Bool=false)
    numberOfBins = numberOfBins(size(x)[1], cor(x, y))
    return variationOfInformation(x, y, numberOfBins; norm=norm)
end

"""
    mutualInformation(x::Array{Float64}, y::Array{Float64}; norm::Bool=false)::Float64

Calculates Mutual Information between two datasets while calculating the number of bins.

# Arguments
- `x::Array{Float64}`: First dataset.
- `y::Array{Float64}`: Second dataset.
- `norm::Bool=false`: Normalize the result.

# Returns
- `Float64`: Mutual Information.
"""
function mutualInformation(x::Array{Float64}, y::Array{Float64}; norm::Bool=false)
    numberOfBins = numberOfBins(size(x)[1], cor(x, y))
    mutualInformation = mutualInformationScore(fit(Histogram, (x, y), (range(minimum(x), maximum(x), length=numberOfBins), range(minimum(y), maximum(y), length=numberOfBins))).weights)
    
    if norm
        marginalX = entropy(normalize(fit(Histogram, x, range(minimum(x), maximum(x), length=numberOfBins)).weights, 1))
        marginalY = entropy(normalize(fit(Histogram, y, range(minimum(y), maximum(y), length=numberOfBins)).weights, 1))
        mutualInformation /= min(marginalX, marginalY)
    end
    
    return mutualInformation
end

"""
    distance(dependence::Matrix, metric::String="angular")::Float64

Calculates the distance from a dependence matrix using the specified metric.

# Arguments
- `dependence::Matrix`: Dependence matrix.
- `metric::String="angular"`: Metric to use ("angular" or "absolute_angular").

# Returns
- `Float64`: Distance value.
"""
function distance(dependence::Matrix, metric::String="angular")
    if metric == "angular"
        distanceFunction = q -> sqrt((1 - q).round(5) / 2.)
    elseif metric == "absolute_angular"
        distanceFunction = q -> sqrt((1 - abs(q)).round(5) / 2.)
    end
    
    return distanceFunction(dependence)
end

"""
    kullbackLeibler(p::Array{Float64}, q::Array{Float64})::Float64

Calculates Kullback-Leibler divergence between two discrete probability distributions defined on the same probability space.

# Arguments
- `p::Array{Float64}`: First distribution.
- `q::Array{Float64}`: Second distribution.

# Returns
- `Float64`: Kullback-Leibler divergence.
"""
function kullbackLeibler(p::Array{Float64}, q::Array{Float64})
    return -sum(p .* log.(q ./ p))
end

"""
    crossEntropy(p::Array{Float64}, q::Array{Float64})::Float64

Calculates cross-entropy between two discrete probability distributions defined on the same probability space.

# Arguments
- `p::Array{Float64}`: First distribution.
- `q::Array{Float64}`: Second distribution.

# Returns
- `Float64`: Cross-entropy.
"""
function crossEntropy(p::Array{Float64}, q::Array{Float64})
    return -sum(p .* log.(q))
end
