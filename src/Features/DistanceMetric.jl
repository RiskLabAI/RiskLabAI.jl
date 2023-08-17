# What is it? the same folder and file are exist in the directory.

# using Statistics
# using StatsBase

# """
#     Calculate mutual information score between two datasets

#     Parameters:
#     - histogramXY::Matrix: 2D histogram matrix of two datasets.

#     Returns:
#     - score::Float64: Mutual information score.
# """
# function mutualInfoScore(histogramXY::Matrix)::Float64
#     score = 0.0
#     histogramX = vec(sum(histogramXY, dims=2))
#     histogramY = vec(sum(histogramXY, dims=1))
#     for i in 1:size(histogramXY)[1]
#         for j in 1:size(histogramXY)[2]
#             if histogramXY[i, j] != 0
#                 score += (histogramXY[i, j] / sum(histogramXY)) *
#                          log(sum(histogramXY) * histogramXY[i, j] / (histogramX[i] * histogramY[j]))
#             end
#         end
#     end
#     return score
# end

# """
#     Calculate Variation of Information between two datasets

#     Parameters:
#     - x::Vector: First dataset.
#     - y::Vector: Second dataset.
#     - numberOfBins::Int: Number of bins for histograms.
#     - norm::Bool: Whether to normalize the variation of information.

#     Returns:
#     - variationXY::Float64: Variation of Information score.
# """
# function variationsInformation(
#     x::Vector, 
#     y::Vector, 
#     numberOfBins::Int, 
#     norm::Bool=false
# )::Float64
#     rangeX = range(minimum(x), maximum(x), length = numberOfBins)
#     rangeY = range(minimum(y), maximum(y), length = numberOfBins)
#     histogramXY = fit(Histogram, (x, y), (rangeX, rangeY)).weights
#     mutualInformation = mutualInfoScore(histogramXY)
#     marginalX = entropy(normalize(fit(Histogram, x, rangeX).weights, 1))
#     marginalY = entropy(normalize(fit(Histogram, y, rangeY).weights, 1))
#     variationXY = marginalX + marginalY - 2 * mutualInformation
#     if norm
#         jointXY = marginalX + marginalY - mutualInformation
#         variationXY /= jointXY
#     end
#     return variationXY
# end

# """
#     Calculate the number of bins for histograms

#     Parameters:
#     - numberObservations::Int: Number of observations.
#     - correlation::Union{Nothing, Float64}: Correlation value.

#     Returns:
#     - bins::Int: Number of bins.
# """
# function numberBins(numberObservations::Int, correlation::Union{Nothing, Float64})::Int
#     if isnothing(correlation)
#         z = (8 + 324 * numberObservations + 12 * sqrt(36 * numberObservations + 729 * numberObservations^2))^(1/3)
#         bins = round(z / 6 + 2 / (3 * z) + 1 / 3)
#     else
#         bins = round(2^-0.5 * sqrt(1 + sqrt(1 + 24 * numberObservations / (1 - correlation^2))))
#     end
#     return Int(bins)
# end

# """
#     Calculate Variation of Information with adaptive number of bins

#     Parameters:
#     - x::Vector: First dataset.
#     - y::Vector: Second dataset.
#     - norm::Bool: Whether to normalize the variation of information.

#     Returns:
#     - variationXY::Float64: Variation of Information score.
# """
# function variationsInformationExtended(
#     x::Vector, 
#     y::Vector, 
#     norm::Bool=false
# )::Float64
#     numberOfBins = numberBins(size(x)[1], cor(x, y))
#     rangeX = range(minimum(x), maximum(x), length = numberOfBins)
#     rangeY = range(minimum(y), maximum(y), length = numberOfBins)
#     histogramXY = fit(Histogram, (x, y),(rangeX, rangeY)).weights
#     mutualInformation = mutualInfoScore(histogramXY)
#     marginalX = entropy(normalize(fit(Histogram, x, rangeX).weights, 1))
#     marginalY = entropy(normalize(fit(Histogram, y, rangeY).weights, 1))
#     variationXY = marginalX + marginalY - 2 * mutualInformation
#     if norm
#         jointXY = marginalX + marginalY - mutualInformation
#         variationXY /= jointXY
#     end
#     return variationXY
# end

# """
#     Calculate Mutual Information with adaptive number of bins

#     Parameters:
#     - x::Vector: First dataset.
#     - y::Vector: Second dataset.
#     - norm::Bool: Whether to normalize mutual information.

#     Returns:
#     - mutualInformation::Float64: Mutual Information score.
# """
# function mutualInformation(
#     x::Vector, 
#     y::Vector, 
#     norm::Bool=false
# )::Float64
#     numberOfBins = numberBins(size(x)[1], cor(x, y))
#     rangeX = range(minimum(x), maximum(x), length = numberOfBins)
#     rangeY = range(minimum(y), maximum(y), length = numberOfBins)
#     histogramXY = fit(Histogram, (x, y),(rangeX, rangeY)).weights
#     mutualInformation = mutualInfoScore(histogramXY)
#     if norm
#         marginalX = entropy(normalize(fit(Histogram, x, rangeX).weights, 1))
#         marginalY = entropy(normalize(fit(Histogram, y, rangeY).weights, 1))
#         mutualInformation /= min(marginalX, marginalY)
#     end
#     return mutualInformation
# end

# """
#     Calculate distance from a dependence matrix

#     Parameters:
#     - dependence::Matrix: Dependence matrix.
#     - metric::String: Distance metric.

#     Returns:
#     - distance::Float64: Calculated distance.
# """
# function distance(dependence::Matrix, metric::String="angular")::Float64
#     if metric == "angular"
#         distanceFunction = q -> sqrt((1 - q).round(5) / 2)
#     elseif metric == "absolute_angular"
#         distanceFunction = q -> sqrt((1 - abs(q)).round(5) / 2)
#     end
#     return distanceFunction(dependence)
# end

# """
#     Calculate Kullback-Leibler divergence from two discrete probability distributions

#     Parameters:
#     - p::Vector: First distribution.
#     - q::Vector: Second distribution.

#     Returns:
#     - divergence::Float64: Kullback-Leibler divergence.
# """
# function KullbackLeibler(p::Vector, q::Vector)::Float64
#     divergence = -sum(p .* log.(q ./ p))
#     return divergence 
# end

# """
#     Calculate crossentropy from two discrete probability distributions

#     Parameters:
#     - p::Vector: First distribution.
#     - q::Vector: Second distribution.

#     Returns:
#     - entropy::Float64: Cross-entropy.
# """
# function crossEntropy(p::Vector, q::Vector)::Float64
#     entropy = -sum(p .* log.(q))
#     return entropy
# end
