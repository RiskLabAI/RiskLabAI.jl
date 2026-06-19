"""
    RiskLabAI.Cluster

Clustering submodule, mirroring the Python `RiskLabAI.cluster` sub-package:
the Optimized Nested Clustering (ONC) algorithm, the k-means base step, the
exact silhouette-score computation, and random block-correlation generators
(López de Prado, *Machine Learning for Asset Managers*, Ch. 4).

The stochastic k-means pieces are **behavioural** ports (k-means is not
bit-identical across backends); `silhouette_samples` and
`covariance_to_correlation` are exact.
"""
module Cluster

include("Clustering.jl")

export
    covariance_to_correlation,
    silhouette_samples,
    cluster_k_means_base,
    cluster_k_means_top,
    make_new_outputs,
    random_covariance_sub,
    random_block_covariance,
    random_block_correlation

end # module Cluster
