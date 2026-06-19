"""
Feature importance — native Julia port mirroring the Python
`RiskLabAI.features.feature_importance` sub-package (López de Prado, AFML Ch. 8).

This file wires the **backend-independent** pieces: PCA-based feature
orthogonalisation and the weighted Kendall-τ. The classifier-driven importances
(MDI, MDA, SFI and their clustered variants) build on a tree ensemble and are
wired with the `DecisionTree.jl` backend in a follow-up.

Parity notes:
  * `orthogonal_features` is **exact** up to the unavoidable sign ambiguity of
    eigenvectors (eigenvalues, cumulative variance and the retained component
    count match Python exactly; the transformed columns are sign-free
    orthogonal). Verified in `test/runtests.jl`.
  * `calculate_weighted_tau` implements Vigna's additive-hyperbolic weighted
    Kendall-τ (the estimator `scipy.stats.weightedtau` computes); validated
    against its mathematical definition (perfect concordance → 1, perfect
    discordance → −1, a hand-computed case).

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 8.
"""

using Statistics: mean, std
using LinearAlgebra: eigen, Symmetric

"""
    orthogonal_features(features; variance_threshold=0.95)
        -> (orthogonal, eigenvalues, eigenvectors, cumulative_variance)

PCA orthogonalisation of a `observations × features` matrix: z-score normalise,
eigendecompose `Xᵀ X` (eigenvalues descending), retain the leading components
whose cumulative explained variance first reaches `variance_threshold`, and
project. Returns the transformed (orthogonal) features and the retained
eigenvalues/eigenvectors/cumulative variance. Mirrors Python's
`orthogonal_features`.
"""
function orthogonal_features(
    features::AbstractMatrix{<:Real};
    variance_threshold::Real = 0.95,
)
    means = mean(features; dims = 1)
    stds = std(features; dims = 1)                       # sample std (ddof = 1)
    normalized = (features .- means) ./ stds

    dot_product = normalized' * normalized
    factorization = eigen(Symmetric(Matrix(dot_product)))
    order = sortperm(factorization.values; rev = true)
    eigenvalues = factorization.values[order]
    eigenvectors = factorization.vectors[:, order]

    cumulative_variance = cumsum(eigenvalues) ./ sum(eigenvalues)
    keep = searchsortedfirst(cumulative_variance, variance_threshold)
    keep = min(keep, length(eigenvalues))

    retained_values = eigenvalues[1:keep]
    retained_vectors = eigenvectors[:, 1:keep]
    orthogonal = normalized * retained_vectors

    return orthogonal, retained_values, retained_vectors, cumulative_variance[1:keep]
end

"""
    calculate_weighted_tau(feature_importances, principal_component_ranks) -> Float64

Weighted Kendall-τ between `feature_importances` and the inverse of
`principal_component_ranks` (more weight on disagreements among top-ranked
items). Implements Vigna's additive-hyperbolic weighted-τ — the estimator used
by Python's `calculate_weighted_tau` (`scipy.stats.weightedtau`). Mirrors
Python's `calculate_weighted_tau`.
"""
function calculate_weighted_tau(
    feature_importances::AbstractVector{<:Real},
    principal_component_ranks::AbstractVector{<:Real},
)
    weights = 1.0 ./ principal_component_ranks
    return _weighted_tau(collect(float(feature_importances)), collect(weights))
end

# One-ordering weighted-τ: rank by `key` descending, additive hyperbolic weights.
function _weighted_tau_one(x, y, key)
    n = length(x)
    order = sortperm(key; rev = true)
    rank = Vector{Int}(undef, n)
    for (position, idx) in enumerate(order)
        rank[idx] = position - 1                         # 0-based rank
    end
    w = 1.0 ./ (rank .+ 1.0)

    numerator = 0.0
    denominator_x = 0.0
    denominator_y = 0.0
    for i = 1:n, j = (i+1):n
        weight = w[i] + w[j]
        sign_x = sign(x[i] - x[j])
        sign_y = sign(y[i] - y[j])
        numerator += weight * sign_x * sign_y
        denominator_x += weight * abs(sign_x)
        denominator_y += weight * abs(sign_y)
    end
    return numerator / sqrt(denominator_x * denominator_y)
end

# Symmetric weighted-τ (scipy's rank=True): average over both orderings.
_weighted_tau(x, y) = 0.5 * (_weighted_tau_one(x, y, x) + _weighted_tau_one(x, y, y))
