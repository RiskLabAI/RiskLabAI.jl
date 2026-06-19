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

# --------------------------------------------------------------------------- #
# Classifier-driven importances (DecisionTree.jl backend).
#
# Parity note: these are **behavioural** — the Julia random forest
# (`DecisionTree.jl`) is not bit-identical to scikit-learn's, so importances are
# validated structurally (informative features rank above noise on a separable
# dataset). Sample weights are not supported by the DecisionTree.jl backend
# (deliberate divergence). Clustered MDI/MDA follow in a small follow-up.
# --------------------------------------------------------------------------- #

using Random: AbstractRNG, MersenneTwister, default_rng, shuffle!
using DecisionTree: build_forest, build_tree, apply_forest, apply_forest_proba,
    impurity_importance

# Cross-entropy / negative-log-likelihood score (sklearn `log_loss` equivalent).
function _log_loss(y_true::AbstractVector, proba::AbstractMatrix, classes::AbstractVector)
    epsilon = 1e-15
    column_of = Dict(c => i for (i, c) in enumerate(classes))
    total = 0.0
    for n in eachindex(y_true)
        p = clamp(proba[n, column_of[y_true[n]]], epsilon, 1 - epsilon)
        total += log(p)
    end
    return -total / length(y_true)
end

_default_subfeatures(p) = max(floor(Int, sqrt(p)), 1)

# Shuffled / contiguous K-Fold (numpy array_split semantics), 1-based indices.
function _kfold(n::Integer, k::Integer; shuffle::Bool = false, rng = default_rng())
    indices = collect(1:n)
    shuffle && shuffle!(rng, indices)
    base, rem = divrem(n, k)
    folds = Tuple{Vector{Int},Vector{Int}}[]
    start = 1
    for i = 1:k
        size = base + (i <= rem ? 1 : 0)
        test = indices[start:(start+size-1)]
        start += size
        push!(folds, (setdiff(indices, test), test))
    end
    return folds
end

_nan_mean(v) = (vals = filter(!isnan, v); isempty(vals) ? NaN : sum(vals) / length(vals))
function _nan_std(v)
    vals = filter(!isnan, v)
    length(vals) < 2 && return NaN
    m = sum(vals) / length(vals)
    return sqrt(sum((vals .- m) .^ 2) / (length(vals) - 1))
end

"""
    feature_importance_mdi(x, y; n_trees=100, n_subfeatures=-1, max_depth=-1, random_state=0)
        -> (; mean, std)

Mean-Decrease-Impurity importance: per-tree normalised impurity importances over
a bootstrap forest, averaged feature-wise (zeros → `NaN` so they are skipped, as
in Python), then normalised to sum to one. Behavioural. Mirrors Python's
`FeatureImportanceMDI`.
"""
function feature_importance_mdi(
    x::AbstractMatrix{<:Real},
    y::AbstractVector;
    n_trees::Integer = 100,
    n_subfeatures::Integer = -1,
    max_depth::Integer = -1,
    random_state::Integer = 0,
)
    n, p = size(x)
    subfeatures = n_subfeatures < 0 ? _default_subfeatures(p) : n_subfeatures
    rng = MersenneTwister(random_state)

    per_tree = Matrix{Float64}(undef, n_trees, p)
    for t = 1:n_trees
        sample = rand(rng, 1:n, n)                     # bootstrap with replacement
        tree = build_tree(y[sample], x[sample, :], subfeatures, max_depth, 1, 2, 0.0; rng = rng)
        importance = impurity_importance(tree; normalize = true)
        for j = 1:p
            per_tree[t, j] = (j <= length(importance)) ? importance[j] : 0.0
        end
    end
    per_tree[per_tree.==0.0] .= NaN

    means = [_nan_mean(view(per_tree, :, j)) for j = 1:p]
    stds = [_nan_std(view(per_tree, :, j)) * (n_trees^-0.5) for j = 1:p]
    total = sum(filter(!isnan, means))
    return (mean = means ./ total, std = stds ./ total)
end

"""
    feature_importance_mda(x, y; n_splits=10, n_trees=100, n_subfeatures=-1, max_depth=-1, random_state=42)
        -> (; mean, std)

Mean-Decrease-Accuracy importance: over a shuffled K-Fold, the drop in negative
log-loss when each feature's test column is permuted, averaged over folds.
Behavioural. Mirrors Python's `FeatureImportanceMDA`.
"""
function feature_importance_mda(
    x::AbstractMatrix{<:Real},
    y::AbstractVector;
    n_splits::Integer = 10,
    n_trees::Integer = 100,
    n_subfeatures::Integer = -1,
    max_depth::Integer = -1,
    random_state::Integer = 42,
)
    n, p = size(x)
    classes = sort(unique(y))
    subfeatures = n_subfeatures < 0 ? _default_subfeatures(p) : n_subfeatures
    folds = _kfold(n, n_splits; shuffle = true, rng = MersenneTwister(random_state))

    drops = Matrix{Float64}(undef, n_splits, p)
    for (f, (train, test)) in enumerate(folds)
        forest = build_forest(
            y[train], x[train, :], subfeatures, n_trees, 0.7, max_depth, 1, 2, 0.0;
            rng = random_state + f,
        )
        baseline = -_log_loss(y[test], apply_forest_proba(forest, x[test, :], classes), classes)
        rng = MersenneTwister(random_state + f)
        for j = 1:p
            shuffled = copy(x[test, :])
            column = shuffled[:, j]
            shuffle!(rng, column)
            shuffled[:, j] = column
            score = -_log_loss(y[test], apply_forest_proba(forest, shuffled, classes), classes)
            drops[f, j] = baseline - score
        end
    end

    means = [sum(view(drops, :, j)) / n_splits for j = 1:p]
    stds = [_nan_std(view(drops, :, j)) * (n_splits^-0.5) for j = 1:p]
    return (mean = means, std = stds)
end

"""
    feature_importance_sfi(x, y; n_splits=10, n_trees=100, max_depth=-1, scoring=:log_loss, random_state=0)
        -> (; mean, std)

Single-Feature Importance: cross-validated score of a forest trained on each
feature alone (`scoring` `:log_loss` → negative log-loss, or `:accuracy`).
Behavioural. Mirrors Python's `FeatureImportanceSFI`.
"""
function feature_importance_sfi(
    x::AbstractMatrix{<:Real},
    y::AbstractVector;
    n_splits::Integer = 10,
    n_trees::Integer = 100,
    max_depth::Integer = -1,
    scoring::Symbol = :log_loss,
    random_state::Integer = 0,
)
    n, p = size(x)
    classes = sort(unique(y))
    folds = _kfold(n, n_splits; shuffle = false)

    means = Float64[]
    stds = Float64[]
    for j = 1:p
        scores = Float64[]
        for (f, (train, test)) in enumerate(folds)
            feature_train = reshape(x[train, j], :, 1)
            feature_test = reshape(x[test, j], :, 1)
            forest = build_forest(
                y[train], feature_train, -1, n_trees, 0.7, max_depth, 1, 2, 0.0;
                rng = random_state + f,
            )
            if scoring == :log_loss
                proba = apply_forest_proba(forest, feature_test, classes)
                push!(scores, -_log_loss(y[test], proba, classes))
            elseif scoring == :accuracy
                push!(scores, sum(apply_forest(forest, feature_test) .== y[test]) / length(test))
            else
                throw(ArgumentError("scoring must be :log_loss or :accuracy"))
            end
        end
        push!(means, sum(scores) / length(scores))
        push!(stds, _nan_std(scores) * (length(scores)^-0.5))
    end
    return (mean = means, std = stds)
end

"""
    get_test_dataset(; n_features=40, n_informative=10, n_redundant=10, n_samples=1000, random_state=0, sigma_std=0.0)
        -> (x, y, feature_names)

Synthetic classification dataset of informative, redundant (noisy copies of
informative) and noise features. Stochastic. Mirrors Python's `get_test_dataset`
(behavioural; `sklearn.make_classification` replaced by a native generator).
"""
function get_test_dataset(;
    n_features::Integer = 40,
    n_informative::Integer = 10,
    n_redundant::Integer = 10,
    n_samples::Integer = 1000,
    random_state::Integer = 0,
    sigma_std::Real = 0.0,
)
    rng = MersenneTwister(random_state)
    n_noise = n_features - n_informative - n_redundant
    y = rand(rng, 0:1, n_samples)
    signed = 2.0 .* y .- 1.0

    columns = Matrix{Float64}(undef, n_samples, n_informative + n_noise)
    for j = 1:n_informative
        columns[:, j] = randn(rng, n_samples) .+ signed .* (0.5 + rand(rng))
    end
    for j = (n_informative+1):(n_informative+n_noise)
        columns[:, j] = randn(rng, n_samples)
    end

    redundant = Matrix{Float64}(undef, n_samples, n_redundant)
    for i = 1:n_redundant
        source = rand(rng, 1:n_informative)
        base = columns[:, source]
        noise = sigma_std .* randn(rng, n_samples) .* std(base)
        redundant[:, i] = base .+ noise
    end

    x = hcat(columns, redundant)
    names = vcat(
        ["I_$(i)" for i = 0:(n_informative-1)],
        ["N_$(i)" for i = 0:(n_noise-1)],
        ["R_$(i)" for i = 0:(n_redundant-1)],
    )
    return x, y, names
end
