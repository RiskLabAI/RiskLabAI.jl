"""
    RiskLabAI.Optimization

Portfolio-optimization submodule, mirroring the Python `RiskLabAI.optimization`
sub-package.

This PR wires the pure-numeric pieces: **Hierarchical Risk Parity** building
blocks (inverse-variance weights, cluster variance, quasi-diagonalisation,
recursive bisection) and **PCA hedging** weights. The Nested Clustered
Optimisation (`nco`) wrapper — which needs a k-means/clustering backend — and the
sklearn-based hyper-parameter tuning are wired in subsequent PRs.
"""
module Optimization

# Hierarchical Risk Parity building blocks (AFML Ch. 16).
include("HierarchicalRiskParity.jl")

# PCA-based hedging weights (AFML Ch. 36).
include("Hedging.jl")

export
    # hierarchical risk parity
    inverse_variance_weights,
    cluster_variance,
    quasi_diagonal,
    recursive_bisection,
    distance_corr,
    # hedging
    pca_weights

end # module Optimization
