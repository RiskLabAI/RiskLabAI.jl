"""
Hierarchical Risk Parity — native Julia port mirroring the Python
`RiskLabAI.optimization.hrp` API (López de Prado, AFML Ch. 16): inverse-variance
weights, cluster variance, quasi-diagonalisation, and recursive bisection.

Representation note (deliberate divergence): pandas DataFrames become `Matrix`es
and asset labels become 1-based integer indices. These deterministic building
blocks match the Python implementation exactly (verified in `test/runtests.jl`).
The top-level `hrp(cov, corr)` wrapper uses native single-linkage minimum-spanning
tree construction with stable edge ordering and ascending cluster identifiers.
These preserve the reference leaf order, including equal-distance ties that
can affect subsequent recursive allocation.
`quasi_diagonal` accepts a SciPy-format linkage matrix.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 16.
"""

using LinearAlgebra: diag

"""
    inverse_variance_weights(covariance_matrix) -> Vector{Float64}

Inverse-variance portfolio weights `(1/diag) / Σ(1/diag)`. Mirrors Python's
`inverse_variance_weights`.
"""
function inverse_variance_weights(covariance_matrix::AbstractMatrix{<:Real})
    weights = 1.0 ./ diag(covariance_matrix)
    return weights ./ sum(weights)
end

"""
    cluster_variance(covariance_matrix, clustered_items) -> Float64

Inverse-variance-weighted variance `wᵀ C w` of the sub-portfolio on
`clustered_items` (1-based indices). Mirrors Python's `cluster_variance`.
"""
function cluster_variance(
    covariance_matrix::AbstractMatrix{<:Real},
    clustered_items::AbstractVector{<:Integer},
)
    cov_slice = covariance_matrix[clustered_items, clustered_items]
    weights = inverse_variance_weights(cov_slice)
    return (weights' * cov_slice * weights)
end

"""
    quasi_diagonal(linkage_matrix) -> Vector{Int}

Quasi-diagonal ordering of original items from a SciPy-format linkage matrix
(columns `[child0, child1, distance, count]`, 0-based item ids). Returns 1-based
item indices (Python returns 0-based). Mirrors Python's `quasi_diagonal`.
"""
function quasi_diagonal(linkage_matrix::AbstractMatrix{<:Real})
    link = Int.(round.(linkage_matrix))
    num_items = link[end, 4]
    to_process = [link[end, 1], link[end, 2]]
    sorted_items = Int[]
    while !isempty(to_process)
        item = popfirst!(to_process)
        if item >= num_items
            cluster_id = item - num_items            # 0-based row index
            pushfirst!(to_process, link[cluster_id+1, 2])
            pushfirst!(to_process, link[cluster_id+1, 1])
        else
            push!(sorted_items, item)
        end
    end
    return sorted_items .+ 1
end

"""
    recursive_bisection(covariance_matrix, sorted_items) -> Vector{Float64}

Hierarchical Risk Parity weights via top-down recursive bisection over the
`sorted_items` order (1-based indices), allocating between sibling clusters by
inverse cluster variance. Mirrors Python's `recursive_bisection`.
"""
function recursive_bisection(
    covariance_matrix::AbstractMatrix{<:Real},
    sorted_items::AbstractVector{<:Integer},
)
    weights = Dict(i => 1.0 for i in sorted_items)
    clusters = [collect(sorted_items)]
    while !isempty(clusters)
        next_clusters = Vector{Int}[]
        for cluster in clusters
            length(cluster) <= 1 && continue
            half = length(cluster) ÷ 2
            push!(next_clusters, cluster[1:half])
            push!(next_clusters, cluster[(half+1):end])
        end
        clusters = next_clusters
        for i = 1:2:length(clusters)
            cluster_0 = clusters[i]
            cluster_1 = clusters[i+1]
            variance_0 = cluster_variance(covariance_matrix, cluster_0)
            variance_1 = cluster_variance(covariance_matrix, cluster_1)
            alpha = (variance_0 + variance_1) == 0 ? 0.5 : 1 - variance_0 / (variance_0 + variance_1)
            for item in cluster_0
                weights[item] *= alpha
            end
            for item in cluster_1
                weights[item] *= (1 - alpha)
            end
        end
    end
    return [weights[i] for i in sorted_items]
end

"""
    distance_corr(correlation_matrix) -> Matrix{Float64}

Correlation-based distance `√((1 - ρ)/2)`. Mirrors Python's `distance_corr`.
"""
distance_corr(correlation_matrix::AbstractMatrix{<:Real}) =
    ((1 .- correlation_matrix) ./ 2.0) .^ 0.5

"""
    hrp(covariance, correlation) -> Vector{Float64}

Hierarchical Risk Parity portfolio weights (in original asset order): correlation
distance → single-linkage clustering → quasi-diagonal leaf order → recursive
bisection. Mirrors Python's `hrp`. The single-linkage leaf order
uses the reference minimum-spanning-tree traversal, stable edge sorting and
ascending child identifiers, including ties.
"""
function hrp(
    covariance::AbstractMatrix{<:Real},
    correlation::AbstractMatrix{<:Real},
)
    distance = distance_corr(correlation)
    distance = (distance .+ distance') ./ 2          # enforce exact symmetry for hclust
    distance[diagind(distance)] .= 0.0
    order = _single_linkage_order(distance)
    bisection = recursive_bisection(covariance, order)
    weights = zeros(Float64, length(order))
    weights[order] = bisection
    return weights
end

function _single_linkage_order(distance::AbstractMatrix{<:Real})
    n = size(distance, 1)
    size(distance, 2) == n && all(isfinite, distance) && all(>=(0), distance) || throw(ArgumentError("distance must be a finite nonnegative square matrix"))
    n <= 1 && return collect(1:n)
    visited = falses(n)
    nearest = fill(Inf, n)
    edge_start, edge_end, heights = Int[], Int[], Float64[]
    current = 1
    for step in 1:(n-1)
        visited[current] = true
        candidate, height = 0, Inf
        for item in 1:n
            visited[item] && continue
            nearest[item] = min(nearest[item], distance[current, item])
            if nearest[item] < height
                candidate, height = item, nearest[item]
            end
        end
        push!(edge_start, current)
        push!(edge_end, candidate)
        push!(heights, height)
        current = candidate
    end
    parents = collect(1:(2n-1))
    counts = ones(Int, 2n-1)
    linkage = zeros(n-1, 4)
    for (step, edge) in enumerate(sortperm(heights; alg = Base.Sort.MergeSort))
        left, right = edge_start[edge], edge_end[edge]
        while parents[left] != left
            left = parents[left]
        end
        while parents[right] != right
            right = parents[right]
        end
        combined = n + step
        counts[combined] = counts[left] + counts[right]
        linkage[step, :] = [min(left, right)-1, max(left, right)-1, heights[edge], counts[combined]]
        parents[left] = parents[right] = combined
    end
    return quasi_diagonal(linkage)
end
