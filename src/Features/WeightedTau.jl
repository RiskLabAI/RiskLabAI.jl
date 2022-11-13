using PyCall

@pyimport scipy.stats as Stats

"""
function: Weighted τ calculation
reference: De Prado, M. (2018) Advances In Financial Machine Learning
methodology: page 121 Orthogonal Features section snippet 8.6
"""
function weightedτ(
    featureImportances::Vector, # vector of feature importances 
    principalComponentRanks::Vector, # vector of principal component ranks
)::Float64
    return Stats.weightedtau(featureImportances, principalComponentRanks .^ -1)[1]
end
