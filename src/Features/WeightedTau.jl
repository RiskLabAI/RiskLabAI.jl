using PyCall

@pyimport scipy.stats as Stats

"""
    weightedTau

Compute the weighted τ (Kendall's Tau) calculation.

This function computes a weighted version of Kendall's τ (tau), which is a measure of the rank correlation between two variables. In this case, the function computes the weighted τ between feature importances and the ranks of principal components.

Parameters:
- `featureImportances::Vector{Float64}`: Vector of feature importances.
- `principalComponentRanks::Vector{Int64}`: Vector of principal component ranks.

Returns:
- `Float64`: Weighted τ value.

Reference:
De Prado, M. (2018) Advances In Financial Machine Learning
Methodology: page 121 Orthogonal Features section snippet 8.6
"""
function weightedTau(
    featureImportances::Vector{Float64},
    principalComponentRanks::Vector{Int64}
)::Float64
    return Stats.weightedtau(featureImportances, principalComponentRanks .^ -1)[1]
end
