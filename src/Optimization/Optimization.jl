"""
    RiskLabAI.Optimization

Portfolio-optimization submodule, mirroring the Python `RiskLabAI.optimization`
sub-package.

Wires **Hierarchical Risk Parity** (building blocks + the `hrp` wrapper), **PCA
hedging** weights, and **Nested Clustered Optimisation** (`nco`, built on the
`Cluster` k-means backend). The sklearn-based hyper-parameter tuning awaits the
ML-backend decision.
"""
module Optimization

# Hierarchical Risk Parity building blocks + hrp() wrapper (AFML Ch. 16).
include("HierarchicalRiskParity.jl")

# PCA-based hedging weights (AFML Ch. 36).
include("Hedging.jl")

# Nested Clustered Optimisation, built on the Cluster k-means backend (AFML Ch. 16).
include("NestedClusteredOptimization.jl")

export
    # hierarchical risk parity
    inverse_variance_weights,
    cluster_variance,
    quasi_diagonal,
    recursive_bisection,
    distance_corr,
    hrp,
    # hedging
    pca_weights,
    # nested clustered optimisation
    get_optimal_portfolio_weights,
    get_optimal_portfolio_weights_nco

end # module Optimization
