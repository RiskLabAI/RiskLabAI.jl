"""
Nested Clustered Optimisation — native Julia port mirroring the Python
`RiskLabAI.optimization.nco` API (López de Prado, AFML Ch. 16 / *Machine
Learning for Asset Managers* Ch. 7): Markowitz optimal weights and the NCO
algorithm that clusters assets, optimises within and between clusters, then
recombines.

Behavioural note: NCO is built on the stochastic k-means base step
(`Cluster.cluster_k_means_base`), so it is a **behavioural** port — tests assert
structural properties (shape, weights summing to one). `get_optimal_portfolio_weights`
(Markowitz GMV/MVO) is deterministic and exact.

Representation note (deliberate divergence): pandas frames become `Matrix`es /
`Vector`s and asset labels become 1-based integer indices.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 16.
"""

using LinearAlgebra: inv
using ..Cluster: cluster_k_means_base, covariance_to_correlation

"""
    get_optimal_portfolio_weights(covariance; mu=nothing) -> Vector{Float64}

Markowitz optimal weights `w = C⁻¹ μ / (1ᵀ C⁻¹ μ)`. With `mu === nothing` this is
the Global Minimum Variance portfolio (`μ = 1`); otherwise it is the
Mean-Variance Optimal portfolio. Mirrors Python's `get_optimal_portfolio_weights`.
"""
function get_optimal_portfolio_weights(
    covariance::AbstractMatrix{<:Real};
    mu::Union{Nothing,AbstractVecOrMat{<:Real}} = nothing,
)
    inverse_covariance = inv(Matrix(covariance))
    ones_vector = ones(size(inverse_covariance, 1))
    expected = mu === nothing ? ones_vector : vec(mu)
    weights = inverse_covariance * expected
    return weights ./ (ones_vector' * weights)
end

"""
    get_optimal_portfolio_weights_nco(covariance; mu=nothing, number_clusters=nothing)
        -> Vector{Float64}

Nested Clustered Optimisation weights: cluster assets by correlation, compute
optimal weights within each cluster, compute optimal weights between clusters
(each cluster treated as one asset), then recombine. `number_clusters` defaults
to `N ÷ 2`. Behavioural (built on stochastic k-means). Mirrors Python's
`get_optimal_portfolio_weights_nco`.
"""
function get_optimal_portfolio_weights_nco(
    covariance::AbstractMatrix{<:Real};
    mu::Union{Nothing,AbstractVector{<:Real}} = nothing,
    number_clusters::Union{Nothing,Integer} = nothing,
)
    n = size(covariance, 2)
    correlation = covariance_to_correlation(covariance)
    nc = number_clusters === nothing ? max(n ÷ 2, 2) : number_clusters

    _, clusters, _ = cluster_k_means_base(correlation; max_clusters = nc, iterations = 10)
    labels = collect(keys(clusters))

    weights_intra = zeros(Float64, n, length(labels))
    for (column, label) in enumerate(labels)
        assets = clusters[label]
        cov_intra = covariance[assets, assets]
        mu_intra = mu === nothing ? nothing : mu[assets]
        weights_intra[assets, column] =
            get_optimal_portfolio_weights(cov_intra; mu = mu_intra)
    end

    covariance_inter = weights_intra' * covariance * weights_intra
    mu_inter = mu === nothing ? nothing : weights_intra' * mu
    weights_inter = get_optimal_portfolio_weights(covariance_inter; mu = mu_inter)

    return weights_intra * weights_inter
end
