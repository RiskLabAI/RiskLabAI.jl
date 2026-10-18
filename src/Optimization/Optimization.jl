"""
    RiskLabAI.Optimization

Portfolio-optimization submodule, mirroring the Python `RiskLabAI.optimization`
sub-package.

Wires **Hierarchical Risk Parity** (building blocks + the `hrp` wrapper), **PCA
hedging** weights, and **Nested Clustered Optimisation** (`nco`, built on the
`Cluster` k-means backend). The sklearn-based hyper-parameter tuning awaits the
ML-backend decision. Constrained quadratic and ellipsoid-robust portfolios use
JuMP and Clarabel with caller-supplied covariance, means and constraints.
"""
module Optimization

# Hierarchical Risk Parity building blocks + hrp() wrapper (AFML Ch. 16).
include("HierarchicalRiskParity.jl")

# PCA-based hedging weights (AFML Ch. 36).
include("Hedging.jl")

# Nested Clustered Optimisation, built on the Cluster k-means backend (AFML Ch. 16).
include("NestedClusteredOptimization.jl")

include("Constrained.jl")
include("Clearing.jl")
include("Execution.jl")
include("Majorization.jl")
include("DecisionFocused.jl")

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
    get_optimal_portfolio_weights_nco,
    constrained_minimum_variance,
    robust_mean_variance,
    eisenberg_noe_clearing,
    almgren_chriss_execution,
    is_doubly_stochastic,
    is_majorized,
    majorization_matrix,
    birkhoff_von_neumann_decomposition,
    spo_plus_loss,
    spo_plus

end # module Optimization
