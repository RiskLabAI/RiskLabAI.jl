"""
Function: Calculates mutual information score between 2 datasets.

Calculates mutual information score between two datasets.

:param histogramXY: 2D histogram matrix of the two datasets.
:return: score::Float64: Mutual information score.
"""
function mutualInfoScore(histogramXY)
    score = 0.0
    histogramX = vec(sum(histogramXY, dims=2))
    histogramY = vec(sum(histogramXY, dims=1))
    
    for i in 1:size(histogramXY)[1]
        for j in 1:size(histogramXY)[2]
            if histogramXY[i, j] != 0
                score += (histogramXY[i, j] / sum(histogramXY)) *
                         log(sum(histogramXY) * histogramXY[i, j] / (histogramX[i] * histogramY[j]))
            else
                score += 0.0
            end
        end
    end
    
    return score
end

"""
Function: Calculates Variation of Information between two datasets.

Calculates Variation of Information between two datasets.

:param x: First dataset.
:param y: Second dataset.
:param numberOfBins: Number of bins for discretization.
:param norm: Normalize the result (default = false).
:return: variationXY::Float64: Variation of Information.
"""
function variationsInformation(x, y, numberOfBins; norm = false)
    rangeX = range(minimum(x), maximum(x), length = numberOfBins)
    rangeY = range(minimum(y), maximum(y), length = numberOfBins)
    histogramXY = fit(Histogram, (x, y), (rangeX, rangeY)).weights
    mutualInformation = mutualInfoScore(histogramXY)
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
Function: Calculates the number of bins for histogram.

Calculates the number of bins for histogram based on the number of observations and correlation.

:param numberObservations: Number of observations.
:param correlation: Correlation between two datasets (default = nothing).
:return: bins::Int: Number of bins.
"""
function numberBins(numberObservations, correlation = nothing)
    if isnothing(correlation)
        z = (8 + 324 * numberObservations + 12 * sqrt(36 * numberObservations + 729 * numberObservations^2))^(1/3)
        bins = round(z / 6 + 2 / (3 * z) + 1 / 3)
    else
        bins = round(2^-0.5 * sqrt(1 + sqrt(1 + 24 * numberObservations / (1 - correlation^2))))
    end
    return Int(bins)
end

"""
Function: Calculates Variation of Information with calculating the number of bins.

Calculates Variation of Information between two datasets while calculating the number of bins.

:param x: First dataset.
:param y: Second dataset.
:param norm: Normalize the result (default = false).
:return: variationXY::Float64: Variation of Information.
"""
function variationsInformationExtended(x, y; norm = false)
    numberOfBins = numberBins(size(x)[1], cor(x, y))
    rangeX = range(minimum(x), maximum(x), length = numberOfBins)
    rangeY = range(minimum(y), maximum(y), length = numberOfBins)
    histogramXY = fit(Histogram, (x, y), (rangeX, rangeY)).weights
    mutualInformation = mutualInfoScore(histogramXY)
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
Function: Calculates Mutual Information with calculating the number of bins.

Calculates Mutual Information between two datasets while calculating the number of bins.

:param x: First dataset.
:param y: Second dataset.
:param norm: Normalize the result (default = false).
:return: mutualInformation::Float64: Mutual Information.
"""
function mutualInformation(x, y; norm = false)
    numberOfBins = numberBins(size(x)[1], cor(x, y))
    rangeX = range(minimum(x), maximum(x), length = numberOfBins)
    rangeY = range(minimum(y), maximum(y), length = numberOfBins)
    histogramXY = fit(Histogram, (x, y), (rangeX, rangeY)).weights
    mutualInformation = mutualInfoScore(histogramXY)
    
    if norm
        marginalX = entropy(normalize(fit(Histogram, x, rangeX).weights, 1))
        marginalY = entropy(normalize(fit(Histogram, y, rangeY).weights, 1))
        mutualInformation /= min(marginalX, marginalY)
    end
    
    return mutualInformation
end

"""
Function: Calculates the distance from a dependence matrix.

Calculates the distance from a dependence matrix using the specified metric.

:param dependence: Dependence matrix.
:param metric: Metric to use ("angular" or "absolute_angular").
:return: distance::Float64: Distance value.
"""
function distance(dependence::Matrix, metric::String = "angular")
    if metric == "angular"
        distanceFunction = q -> sqrt((1 - q).round(5) / 2.)
    elseif metric == "absolute_angular"
        distanceFunction = q -> sqrt((1 - abs(q)).round(5) / 2.)
    end
    
    return distanceFunction(dependence)
end

"""
Function: Calculates Kullback-Leibler divergence between two discrete probability distributions.

Calculates Kullback-Leibler divergence between two discrete probability distributions defined on the same probability space.

:param p: First distribution.
:param q: Second distribution.
:return: divergence::Float64: Kullback-Leibler divergence.
"""
function KullbackLeibler(p, q)
    divergence = -sum(p .* log.(q ./ p))
    return divergence 
end

"""
Function: Calculates cross-entropy between two discrete probability distributions.

Calculates cross-entropy between two discrete probability distributions defined on the same probability space.

:param p: First distribution.
:param q: Second distribution.
:return: entropy::Float64: Cross-entropy.
"""
function crossEntropy(p, q)
    entropy = -sum(p .* log.(q))
    return entropy
end
