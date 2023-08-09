using PyCall

@pyimport scipy.stats as Stats

"""
Compute the weighted τ (Kendall's Tau) calculation.

Reference: De Prado, M. (2018) Advances In Financial Machine Learning
Methodology: page 121 Orthogonal Features section snippet 8.6
"""
function weightedτ(
    featureImportances::Vector,  # vector of feature importances
    principalComponentRanks::Vector  # vector of principal component ranks
)::Float64
    return Stats.weightedtau(featureImportances, principalComponentRanks .^ -1)[1]
end

