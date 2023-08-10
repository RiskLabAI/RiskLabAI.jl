
"""----------------------------------------------------------------------
function: Calculates mutual information score between 2 datasets
reference: n/a
methodology: n/a
----------------------------------------------------------------------"""
function mutualInfoScore(histogramXY) # hist2d matrix of 2 data
    score = 0. # initial score
    histogramX = vec(sum(histogramXY, dims=2)) # hist of first data
    histogramY = vec(sum(histogramXY, dims=1)) # hist of second data
    for i in 1:size(histogramXY)[1]
        for j in 1:size(histogramXY)[2]
            if histogramXY[i, j] != 0
                # update score 
                score += (histogramXY[i, j]/sum(histogramXY))*
                         log(sum(histogramXY)*histogramXY[i, j]/(histogramX[i]*histogramY[j]))
            else
                score += 0.
            end
        end
    end
    return score
end

"""----------------------------------------------------------------------
function: Calculates Variation of Information
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 3.2, Page 44
----------------------------------------------------------------------"""
function variationsInformation(x, # first data
                               y, # second data
                               numberOfBins; # number of bins
                               norm = false) # for normalized
    rangeX = range(minimum(x), maximum(x), length = numberOfBins) # range for hist
    rangeY = range(minimum(y), maximum(y), length = numberOfBins)# range for hist
    # variation of information
    histogramXY = fit(Histogram, (x, y), (rangeX, rangeY)).weights # hist2d of x, y
    mutualInformation = mutualInfoScore(histogramXY) # mutual score from hist2d
    marginalX = entropy(normalize(fit(Histogram, x, rangeX).weights, 1)) # marginal
    marginalY = entropy(normalize(fit(Histogram, y, rangeY).weights, 1)) # marginal
    variationXY = marginalX + marginalY -2*mutualInformation # variation of information
    if norm
        jointXY = marginalX + marginalY - mutualInformation # joint
        variationXY /= jointXY # normalized variation of information
    end
    return variationXY
end

"""----------------------------------------------------------------------
function: Calculates number of bins for histogram
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 3.3, Page 46
----------------------------------------------------------------------"""
function numberBins(numberObservations, # number of obs 
                    correlation = nothing) # corr
    # Optimal number of bins for discretization
    if isnothing(correlation) # univariate case
        z = (8 + 324*numberObservations + 12*sqrt(36*numberObservations + 729*numberObservations^2))^(1/3)
        bins = round(z/6 + 2/(3*z) + 1/3) # bins
    else # bivariate case
        bins = round(2^-.5*sqrt(1 + sqrt(1 + 24*numberObservations/(1 - correlation^2)))) # bins
    end
    return Int(bins)
end

"""----------------------------------------------------------------------
function: Calculates Variation of Information with calculating number of bins
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 3.3, Page 46
----------------------------------------------------------------------"""
function variationsInformationExtended(x, # data1
                                       y; # data2
                                       norm = false) # for normalized variations of info
    #variation of information
    numberOfBins = numberBins(size(x)[1], cor(x, y)) # calculate number of bins
    rangeX = range(minimum(x), maximum(x), length = numberOfBins) # range for hist
    rangeY = range(minimum(y), maximum(y), length = numberOfBins) # range for hist
    # variation of information
    histogramXY = fit(Histogram, (x, y),(rangeX, rangeY)).weights # hist2d of x,y
    mutualInformation = mutualInfoScore(histogramXY) # mutual score
    marginalX = entropy(normalize(fit(Histogram, x, rangeX).weights, 1)) # marginal
    marginalY = entropy(normalize(fit(Histogram, y, rangeY).weights, 1)) # marginal
    variationXY = marginalX + marginalY - 2*mutualInformation # variation of information
    if norm
        jointXY = marginalX + marginalY - mutualInformation # joint
        variationXY /= jointXY # normalized variation of information 
    end
    return variationXY
end

"""----------------------------------------------------------------------
function: Calculates Mutual  Information with calculating number of bins
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: Snipet 3.4, Page 48
----------------------------------------------------------------------"""
function mutualInformation(x, # data1
                           y; # data2
                           norm = false) # for normalized variations of info
    # mutual information
    numberOfBins = numberBins(size(x)[1], cor(x, y)) # calculate number of bins
    rangeX = range(minimum(x), maximum(x), length = numberOfBins) # range of x for histogram
    rangeY = range(minimum(y), maximum(y), length = numberOfBins) # range of y for histogram
    histogramXY = fit(Histogram, (x, y),(rangeX, rangeY)).weights  # hist2d of x,y
    mutualInformation = mutualInfoScore(histogramXY) # mutual score
    if norm
        marginalX = entropy(normalize(fit(Histogram, x, rangeX).weights, 1)) # marginal
        marginalY = entropy(normalize(fit(Histogram, y, rangeY).weights, 1)) # marginal
        mutualInformation /= min(marginalX, marginalY) # normalized mutual information
    end
    return mutualInformation
end

"""----------------------------------------------------------------------
function: Calculates distance from a dependence matrix
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
----------------------------------------------------------------------"""
function distance(dependence::Matrix, # dependence matrix
                  metric::String = "angular") # method
    if metric == "angular"
        distanceFunction = q -> sqrt((1 - q).round(5)/2.)
    elseif metric == "absolute_angular"
        distanceFunction = q -> sqrt((1 - abs(q)).round(5)/2.)
    end
    return distanceFunction(dependence)
end

"""----------------------------------------------------------------------
function: Calculates KullbackLeibler divergence from two discrete probability distributions defined on the same probability space
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
----------------------------------------------------------------------"""
function KullbackLeibler(p, # first distribution
                         q) # second distribution
    divergence = -sum(p.*log.(q./p))  # calculate divenrgence
    return divergence 
end

"""----------------------------------------------------------------------
function: Calculates crossentropy from two discrete probability distributions defined on the same probability space
reference: De Prado, M. (2020) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
----------------------------------------------------------------------"""
function crossEntropy(p, # first distribution
                      q) # second distribution
    entropy = -sum(p.*log.(q)) # calculate entropy
    return entropy
end