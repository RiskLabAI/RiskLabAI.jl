"""
Clustering — native Julia port mirroring the Python `RiskLabAI.cluster.clustering`
API (López de Prado, AFML Ch. 4): the Optimized Nested Clustering (ONC)
algorithm and its k-means base step, plus random block-correlation generators.

Parity note: k-means is **stochastic** and not bit-identical across
implementations (`Clustering.jl` here vs scikit-learn in Python). Supplied label
providers validate selection and refinement independently of those generators;
supplied block cuts and Gaussian primitives replay random correlations.
The default fitted partitions remain behavioural. `silhouette_samples` matches the
canonical (scikit-learn) definition exactly. `covariance_to_correlation`
delegates to `Data.cov_to_corr` (the single source of truth).

Representation note (deliberate divergence): correlation/covariance are `Matrix`es
and items are 1-based integer indices (Python uses pandas labels).

Reference: De Prado, M. (2020), Machine Learning for Asset Managers, Ch. 4.
"""

using Statistics: mean, std, cov
using Random: AbstractRNG, MersenneTwister, default_rng, shuffle, randn
using Clustering: kmeans, assignments
using ..Data: cov_to_corr

_rng(random_state) =
    random_state === nothing ? default_rng() :
    random_state isa AbstractRNG ? random_state : MersenneTwister(random_state)

"""
    covariance_to_correlation(covariance) -> Matrix

Correlation matrix from a covariance matrix; delegates to `Data.cov_to_corr`
(Snippet 2.3). Mirrors Python's `covariance_to_correlation`.
"""
covariance_to_correlation(covariance::AbstractMatrix{<:Real}) = cov_to_corr(covariance)

"""
    silhouette_samples(distance, labels) -> Vector{Float64}

Per-sample silhouette scores from a **precomputed** distance matrix and cluster
`labels`: `s(i) = (b(i) - a(i)) / max(a(i), b(i))`, with `a` the mean
intra-cluster distance and `b` the mean nearest-other-cluster distance
(`s(i) = 0` for singleton clusters). Matches scikit-learn's `silhouette_samples`
(metric `"precomputed"`) exactly.
"""
function silhouette_samples(
    distance::AbstractMatrix{<:Real},
    labels::AbstractVector{<:Integer},
)
    n = length(labels)
    size(distance) == (n, n) || throw(DimensionMismatch("Distance and label dimensions differ."))
    unique_labels = unique(labels)
    2 <= length(unique_labels) < n || throw(ArgumentError("Silhouettes require between two and n-1 distinct labels."))
    scores = zeros(Float64, n)
    for i = 1:n
        same = findall(==(labels[i]), labels)
        n_same = length(same)
        if n_same <= 1
            scores[i] = 0.0
            continue
        end
        a = sum(distance[i, j] for j in same) / (n_same - 1)   # self distance is 0
        b = Inf
        for c in unique_labels
            c == labels[i] && continue
            members = findall(==(c), labels)
            isempty(members) && continue
            b = min(b, sum(distance[i, j] for j in members) / length(members))
        end
        denominator = max(a, b)
        scores[i] = denominator > 0 ? (b - a) / denominator : 0.0
    end
    return scores
end

_correlation_distance(correlation) =
    sqrt.((1 .- map(x -> isnan(x) ? 0.0 : x, correlation)) ./ 2.0)

function _embedding_distances(features)
    n = size(features, 1)
    distance = zeros(Float64, n, n)
    for right in 2:n, left in 1:(right-1)
        value = sqrt(sum(abs2(features[left, column] - features[right, column]) for column in axes(features, 2)))
        distance[left, right] = distance[right, left] = value
    end
    return distance
end

"""
    cluster_k_means_base(correlation; max_clusters=10, iterations=10, random_state=nothing)
        -> (correlation_sorted, clusters, silhouette)

K-means base step: over `2:max_clusters` clusters and `iterations` initialisations,
keep the clustering with the highest silhouette t-statistic (mean/std, population
std) using Euclidean distances between rows of the correlation-distance
embedding. Returns the cluster-sorted correlation matrix, a `Dict(label => item
indices)`, and the per-item silhouette vector (original order). Optional
`label_provider(features, cluster_count, rng)` supplies integer candidate labels.
Labels retain provider values; item positions are one-based. Ordering within
equal labels is stable. Default native k-means remains stochastic.
"""
function cluster_k_means_base(
    correlation::AbstractMatrix{<:Real};
    max_clusters::Integer = 10,
    iterations::Integer = 10,
    random_state = nothing,
    label_provider = nothing,
)
    n = size(correlation, 1)
    size(correlation, 2) == n || throw(DimensionMismatch("Correlation must be square."))
    2 <= max_clusters < n || throw(ArgumentError("max_clusters must be between two and n-1."))
    iterations > 0 || throw(ArgumentError("iterations must be positive."))
    distance = _correlation_distance(correlation)
    silhouette_distance = _embedding_distances(distance)
    rng = _rng(random_state)

    best_labels = nothing
    best_silhouette = nothing
    best_score = -Inf
    for _ = 1:iterations
        for n_clusters = 2:max_clusters
            labels = label_provider === nothing ? assignments(kmeans(permutedims(distance), n_clusters; rng = rng)) : Int.(label_provider(copy(distance), n_clusters, rng))
            length(labels) == n || throw(DimensionMismatch("Candidate labels must cover every item."))
            silhouette = silhouette_samples(silhouette_distance, labels)
            stat_mean = mean(silhouette)
            stat_std = std(silhouette; corrected = false)
            score = stat_std == 0 ? sign(stat_mean) * Inf : stat_mean / stat_std
            if score > best_score
                best_score = score
                best_labels = labels
                best_silhouette = silhouette
            end
        end
    end

    best_labels === nothing && throw(ArgumentError("Clustering failed to find a valid solution."))
    index_sorted = sortperm(best_labels)
    correlation_sorted = correlation[index_sorted, index_sorted]
    clusters = Dict(c => findall(==(c), best_labels) for c in unique(best_labels))
    return (correlation_sorted, clusters, best_silhouette)
end

"""
    make_new_outputs(correlation, clusters_1, clusters_2)
        -> (correlation_new, clusters_new, silhouette_new)

Merge two disjoint cluster dictionaries (values are item indices into
`correlation`), reorder the correlation matrix accordingly, and recompute
silhouette scores. Mirrors Python's `make_new_outputs`.
"""
function make_new_outputs(
    correlation::AbstractMatrix{<:Real},
    clusters_1::AbstractDict,
    clusters_2::AbstractDict,
)
    n = size(correlation, 2)
    clusters_new = Dict{Int,Vector{Int}}()
    next_key = 0
    for d in (clusters_1, clusters_2)
        for key in sort(collect(keys(d)))
            clusters_new[next_key] = collect(d[key])
            next_key += 1
        end
    end

    index_new = reduce(vcat, (clusters_new[k] for k in sort(collect(keys(clusters_new)))))
    correlation_new = correlation[index_new, index_new]

    distance = _correlation_distance(correlation)
    labels = zeros(Int, n)
    for (key, items) in clusters_new
        labels[items] .= key
    end
    silhouette_new = silhouette_samples(_embedding_distances(distance), labels)
    return (correlation_new, clusters_new, silhouette_new)
end

# cluster silhouette t-statistic (pandas .std() -> sample std, ddof = 1)
function _cluster_t_stats(silhouette, clusters)
    t_stats = Dict{Int,Float64}()
    for i in sort(collect(keys(clusters)))
        items = clusters[i]
        s = std(silhouette[items]; corrected = true)
        s > 0 && (t_stats[i] = mean(silhouette[items]) / s)
    end
    return t_stats
end

"""
    cluster_k_means_top(correlation; max_clusters=nothing, iterations=10, random_state=nothing)
        -> (correlation_sorted, clusters, silhouette)

Optimized Nested Clustering (ONC): run the base k-means, then recursively
re-cluster the clusters whose silhouette t-statistic is below average, keeping
the re-clustering only if it improves the mean t-statistic. Behavioural (built on
stochastic k-means). A supplied `label_provider` is forwarded through every
recursive candidate fit, preserving the strict public refinement decision.
"""
function cluster_k_means_top(
    correlation::AbstractMatrix{<:Real};
    max_clusters = nothing,
    iterations::Integer = 10,
    random_state = nothing,
    label_provider = nothing,
)
    n = size(correlation, 2)
    mc = max_clusters === nothing ? n - 1 : min(max_clusters, n - 1)
    if mc < 2
        return (correlation, Dict(0 => collect(1:n)), Float64[])
    end

    correlation_sorted, clusters, silhouette = cluster_k_means_base(
        correlation;
        max_clusters = mc,
        iterations = iterations,
        random_state = random_state,
        label_provider = label_provider,
    )

    cluster_t_stats = _cluster_t_stats(silhouette, clusters)
    isempty(cluster_t_stats) && return (correlation_sorted, clusters, silhouette)
    ordered_keys = sort(collect(keys(cluster_t_stats)))
    t_stat_mean = mean(cluster_t_stats[key] for key in ordered_keys)
    redo = [key for key in ordered_keys if cluster_t_stats[key] < t_stat_mean]
    length(redo) <= 1 && return (correlation_sorted, clusters, silhouette)

    keys_redo = reduce(vcat, (clusters[i] for i in redo))
    correlation_temp = correlation[keys_redo, keys_redo]
    t_stat_mean_redo = mean(cluster_t_stats[i] for i in redo)
    n_good = length(clusters) - length(redo)
    remained = mc - n_good

    _, clusters_2_local, _ = cluster_k_means_top(
        correlation_temp;
        max_clusters = min(remained, size(correlation_temp, 2) - 1),
        iterations = iterations,
        random_state = random_state,
        label_provider = label_provider,
    )
    clusters_2 = Dict(k => keys_redo[v] for (k, v) in clusters_2_local)
    clusters_1 = Dict(i => clusters[i] for i in keys(clusters) if !(i in redo))

    correlation_new, clusters_new, silhouette_new =
        make_new_outputs(correlation, clusters_1, clusters_2)
    new_stats = _cluster_t_stats(silhouette_new, clusters_new)
    new_t_stats = [new_stats[key] for key in sort(collect(keys(new_stats)))]
    isempty(new_t_stats) && return (correlation_sorted, clusters, silhouette)

    if mean(new_t_stats) <= t_stat_mean_redo
        return (correlation_sorted, clusters, silhouette)
    end
    return (correlation_new, clusters_new, silhouette_new)
end

"""
    random_covariance_sub(n_observations, n_columns, sigma; random_state=nothing) -> Matrix

Random covariance of one block: a shared factor plus idiosyncratic noise.
Optional `common_draws` and `idiosyncratic_draws` are standard Gaussian primitives
with shapes `(n_observations,)` and `(n_observations, n_columns)`. A singleton
block retains the public unit covariance without consuming Gaussian draws.
"""
function random_covariance_sub(
    n_observations::Integer,
    n_columns::Integer,
    sigma::Real;
    random_state = nothing,
    common_draws = nothing,
    idiosyncratic_draws = nothing,
)
    n_columns == 1 && return ones(1, 1)
    rng = _rng(random_state)
    common = common_draws === nothing ? randn(rng, n_observations) : Vector{Float64}(common_draws)
    noise = idiosyncratic_draws === nothing ? randn(rng, n_observations, n_columns) : Matrix{Float64}(idiosyncratic_draws)
    length(common) == n_observations || throw(DimensionMismatch("Common draws must match n_observations."))
    size(noise) == (n_observations, n_columns) || throw(DimensionMismatch("Idiosyncratic draws have the wrong shape."))
    data = reshape(common, :, 1) .+ sigma .* noise
    return cov(data)
end

"""
    random_block_covariance(n_columns, n_blocks; block_size_min=1, sigma=1.0, random_state=nothing) -> Matrix

Random block-diagonal covariance matrix. Optional distinct integer `cuts` replay
the block partition and `block_draws` supplies one `(common, noise)` record per
block. The sample covariance and observation-count rules match the reference.
"""
function random_block_covariance(
    n_columns::Integer,
    n_blocks::Integer;
    block_size_min::Integer = 1,
    sigma::Real = 1.0,
    random_state = nothing,
    cuts = nothing,
    block_draws = nothing,
)
    n_blocks > 0 && block_size_min > 0 && n_columns >= n_blocks * block_size_min || throw(ArgumentError("Requested blocks cannot fit the available columns."))
    rng = _rng(random_state)
    limit = n_columns - (block_size_min - 1) * n_blocks
    chosen = cuts === nothing ? sort(shuffle(rng, collect(1:(limit-1)))[1:(n_blocks-1)]) : sort(Vector{Int}(cuts))
    length(chosen) == n_blocks - 1 && length(unique(chosen)) == length(chosen) && all(cut -> 1 <= cut < limit, chosen) || throw(ArgumentError("Block cuts must be distinct interior positions."))
    block_draws === nothing || length(block_draws) == n_blocks || throw(DimensionMismatch("Supply one primitive record per block."))
    parts = vcat(chosen, limit)
    parts = vcat(parts[1], diff(parts)) .- 1 .+ block_size_min

    blocks = Matrix{Float64}[]
    for (index, col_size) in enumerate(parts)
        n_obs = Int(max(col_size * (col_size + 1) / 2.0, 100))
        common = block_draws === nothing ? nothing : block_draws[index].common
        noise = block_draws === nothing ? nothing : block_draws[index].noise
        push!(blocks, random_covariance_sub(n_obs, col_size, sigma; random_state = rng, common_draws = common, idiosyncratic_draws = noise))
    end

    total = sum(size(b, 1) for b in blocks)
    out = zeros(Float64, total, total)
    offset = 0
    for b in blocks
        s = size(b, 1)
        out[(offset+1):(offset+s), (offset+1):(offset+s)] = b
        offset += s
    end
    return out
end

"""
    random_block_correlation(n_columns, n_blocks; random_state=nothing, block_size_min=1) -> Matrix

Random block-diagonal correlation matrix with an added market component.
Optional `draws=(cuts, blocks, market)` supplies block cuts, per-block
`(common, noise)` primitives and the corresponding market primitives. Equal
seeds across Julia and NumPy are not equivalent random streams.
"""
function random_block_correlation(
    n_columns::Integer,
    n_blocks::Integer;
    random_state = nothing,
    block_size_min::Integer = 1,
    draws = nothing,
)
    rng = _rng(random_state)
    covariance_1 = random_block_covariance(
        n_columns,
        n_blocks;
        block_size_min = block_size_min,
        sigma = 0.5,
        random_state = rng,
        cuts = draws === nothing ? nothing : draws.cuts,
        block_draws = draws === nothing ? nothing : draws.blocks,
    )
    covariance_2 = random_block_covariance(
        n_columns,
        1;
        block_size_min = n_columns,
        sigma = 1.0,
        random_state = rng,
        block_draws = draws === nothing ? nothing : [draws.market],
    )
    return covariance_to_correlation(covariance_1 .+ covariance_2)
end
