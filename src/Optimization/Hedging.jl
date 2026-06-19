"""
PCA hedging — native Julia port mirroring the Python
`RiskLabAI.optimization.hedging` API (López de Prado, AFML Ch. 36): PCA-based
portfolio weights matching a target risk distribution across principal
components.

Deliberate divergence / note: eigenvectors are sign-ambiguous, so the raw
weight signs are not bit-identical across implementations; the meaningful,
sign-free invariant `wᵀ C w = risk_target² · Σ(risk_distribution)` holds exactly
and is what the tests assert.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 36.
"""

using LinearAlgebra: eigen, Symmetric

"""
    pca_weights(cov; risk_distribution=nothing, risk_target=1.0) -> Vector{Float64}

PCA hedging weights `w = V · √(risk_target² · ρ / λ)` over the principal
components of `cov` (eigenvalues descending). With `risk_distribution === nothing`
all risk is placed on the smallest-eigenvalue component (minimum-variance).
Mirrors Python's `pca_weights`.
"""
function pca_weights(
    cov::AbstractMatrix{<:Real};
    risk_distribution::Union{Nothing,AbstractVector{<:Real}} = nothing,
    risk_target::Real = 1.0,
)
    factorization = eigen(Symmetric(Matrix(cov)))
    order = sortperm(factorization.values; rev = true)
    eigen_values = factorization.values[order]
    eigen_vectors = factorization.vectors[:, order]

    rd = risk_distribution
    if rd === nothing
        rd = zeros(size(cov, 1))
        rd[end] = 1.0
    end

    loads = risk_target .* sqrt.(rd ./ eigen_values)
    return eigen_vectors * loads
end
