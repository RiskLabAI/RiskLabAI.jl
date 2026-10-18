using Random: randperm

function _importance_summary(values)
    size(values, 1) >= 1 || throw(ArgumentError("at least one importance observation is required"))
    means = [_nan_mean(view(values, :, column)) for column in axes(values, 2)]
    errors = [_nan_std(view(values, :, column)) / sqrt(size(values, 1)) for column in axes(values, 2)]
    return (mean = means, std = errors)
end

function _importance_clusters(clusters, n_features)
    named = clusters isa AbstractDict || clusters isa AbstractVector{<:Pair}
    entries = named ? collect(clusters) : [index => group for (index, group) in enumerate(clusters)]
    groups = [Int.(collect(last(entry))) for entry in entries]
    all(all(index -> 1 <= index <= n_features, group) for group in groups) || throw(ArgumentError("cluster feature positions must index the data columns"))
    return groups, ["C_" * string(first(entry)) for entry in entries]
end

"""
    feature_importance_mdi(per_tree_importances) -> (; mean, std)

Aggregate a tree-by-feature importance matrix using the released zero-to-NaN
rule, sample standard deviation divided by the square root of all trees, and
normalization by the sum of feature means. Undefined columns remain NaN.
"""
function feature_importance_mdi(per_tree_importances::AbstractMatrix{<:Real})
    values = Float64.(per_tree_importances)
    values[values .== 0] .= NaN
    result = _importance_summary(values)
    total = sum(filter(!isnan, result.mean))
    return (mean = result.mean ./ total, std = result.std ./ total)
end

"""
    clustered_feature_importance_mdi(per_tree_importances, clusters)

Sum member-feature importances within each tree before computing normalized
cluster means and standard errors. NaNs and zero importances are skipped;
an all-missing or empty cluster contributes zero for that tree. Clusters may
overlap. Provide ordered vectors of one-based feature positions, ordered
`name => positions` pairs, or a dictionary. Returned `names` identify the
output order with the public `C_` prefix.
"""
function clustered_feature_importance_mdi(per_tree_importances::AbstractMatrix{<:Real}, clusters)
    values = Float64.(per_tree_importances)
    values[values .== 0] .= NaN
    groups, names = _importance_clusters(clusters, size(values, 2))
    grouped = [sum((values[row, column] for column in group if !isnan(values[row, column])); init = 0.0) for row in axes(values, 1), group in groups]
    result = _importance_summary(grouped)
    total = sum(result.mean)
    return (mean = result.mean ./ total, std = result.std ./ total, names = names)
end

function _importance_weights(weights, count)
    values = weights === nothing ? ones(count) : Float64.(weights)
    length(values) == count || throw(DimensionMismatch("weights must align with observations"))
    all(value -> isfinite(value) && value >= 0, values) || throw(ArgumentError("weights must be finite and nonnegative"))
    return values
end

function _importance_input_shape(x, y)
    length(y) == size(x, 1) || throw(DimensionMismatch("labels must align with observations"))
    size(x, 1) > 0 && size(x, 2) > 0 || throw(ArgumentError("feature data must be nonempty"))
    return nothing
end

"""
    feature_importance_mdi(estimator, x, y; fit_estimator, tree_importances,
                           sample_weights=nothing)

Fit the caller's ensemble through `fit_estimator(model, x, y, weights)` and
aggregate `tree_importances(fitted_model)`. The callback receives `nothing`
when no weights were supplied. Return mean, standard error and the fitted
`estimator`; no tree backend or alternative weighting rule is selected.
"""
function feature_importance_mdi(estimator, x::AbstractMatrix, y::AbstractVector; fit_estimator, tree_importances, sample_weights = nothing)
    _importance_input_shape(x, y)
    weights = sample_weights === nothing ? nothing : _importance_weights(sample_weights, length(y))
    fitted = fit_estimator(estimator, copy(x), copy(y), weights)
    values = tree_importances(fitted)
    size(values, 2) == size(x, 2) || throw(DimensionMismatch("tree importances must align with features"))
    return merge(feature_importance_mdi(values), (estimator = fitted,))
end

"""
    clustered_feature_importance_mdi(estimator, x, y, clusters;
        fit_estimator, tree_importances, sample_weights=nothing)

Fit the caller ensemble with aligned observation weights, then aggregate the
supplied feature clusters. Return cluster means, standard errors, names and
the fitted estimator.
"""
function clustered_feature_importance_mdi(estimator, x::AbstractMatrix, y::AbstractVector, clusters; fit_estimator, tree_importances, sample_weights = nothing)
    _importance_input_shape(x, y)
    _importance_clusters(clusters, size(x, 2))
    weights = sample_weights === nothing ? nothing : _importance_weights(sample_weights, length(y))
    fitted = fit_estimator(estimator, copy(x), copy(y), weights)
    values = tree_importances(fitted)
    size(values, 2) == size(x, 2) || throw(DimensionMismatch("tree importances must align with features"))
    return merge(clustered_feature_importance_mdi(values, clusters), (estimator = fitted,))
end

function _importance_folds(n_samples, n_splits, folds, shuffle, random_state)
    if folds === nothing
        2 <= n_splits <= n_samples || throw(ArgumentError("n_splits must be between two and the sample count"))
        result = [(sort(train), sort(test)) for (train, test) in _kfold(n_samples, n_splits; shuffle, rng = MersenneTwister(random_state))]
    else
        result = [(Int.(collect(train)), Int.(collect(test))) for (train, test) in folds]
        length(result) >= 2 || throw(ArgumentError("at least two folds are required"))
    end
    for (train, test) in result
        !isempty(train) && !isempty(test) && allunique(train) && allunique(test) || throw(ArgumentError("folds must have nonempty distinct positions"))
        all(index -> 1 <= index <= n_samples, [train; test]) || throw(ArgumentError("fold positions must index the observations"))
        sort(train) == setdiff(1:n_samples, test) || throw(ArgumentError("training positions must complement each test fold"))
    end
    sort(reduce(vcat, [test for (_, test) in result])) == collect(1:n_samples) || throw(ArgumentError("test folds must partition all observations"))
    return result
end

function _importance_log_loss(labels, probabilities, weights, classes)
    ordered_classes = sort(unique(classes))
    length(ordered_classes) >= 2 || throw(ArgumentError("log loss requires at least two classes"))
    values = float.(probabilities)
    ndims(values) == 1 && (values = reshape(values, :, 1))
    size(values, 2) == 1 && length(ordered_classes) == 2 && (values = hcat(1 .- values, values))
    size(values) == (length(labels), length(ordered_classes)) || throw(DimensionMismatch("probabilities must align with observations and classes"))
    all(value -> isfinite(value) && 0 <= value <= 1, values) || throw(ArgumentError("probabilities must lie in [0, 1]"))
    sum(weights) > 0 || throw(ArgumentError("scoring weights must have positive sum"))
    epsilon = eps(eltype(values))
    columns = Dict(label => index for (index, label) in enumerate(ordered_classes))
    all(haskey(columns, label) for label in labels) || throw(ArgumentError("observed labels must belong to the declared classes"))
    losses = [-log(clamp(values[row, columns[label]], epsilon, one(epsilon)-epsilon)) for (row, label) in enumerate(labels)]
    return sum(weights .* losses) / sum(weights)
end

function _importance_permutations(permutations, folds, groups)
    permutations === nothing && return nothing
    length(permutations) == length(folds) || throw(DimensionMismatch("permutations must align with folds"))
    for (fold, (_, test)) in enumerate(folds)
        length(permutations[fold]) == length(groups) || throw(DimensionMismatch("permutations must align with feature groups"))
        all(sort(Int.(permutation)) == collect(1:length(test)) for permutation in permutations[fold]) || throw(ArgumentError("each permutation must index every test observation once"))
    end
    return permutations
end

"""
    clustered_feature_importance_mda(estimator, x, y, clusters;
        fit_estimator, predict_proba, estimator_classes, train_sample_weights=nothing,
        score_sample_weights=nothing, n_splits=10, random_state=42,
        folds=nothing, permutations=nothing)

Compute the released weighted decrease in negative log loss after jointly
permuting each cluster's test rows. Empty clusters have exactly zero decrease.
Callbacks fit and update the same estimator sequentially across folds, retaining
the source warm-start semantics. Training and scoring weights are separate.
Supplied one-based folds and `[fold][cluster]` row permutations enable exact
cross-language replay; native random streams are not NumPy seed equivalents.
Return means, standard errors, cluster names and the final fitted estimator.
"""
function clustered_feature_importance_mda(estimator, x::AbstractMatrix, y::AbstractVector, clusters; fit_estimator, predict_proba, estimator_classes, train_sample_weights = nothing, score_sample_weights = nothing, n_splits::Integer = 10, random_state::Integer = 42, folds = nothing, permutations = nothing)
    _importance_input_shape(x, y)
    groups, names = _importance_clusters(clusters, size(x, 2))
    training_weights = _importance_weights(train_sample_weights, length(y))
    scoring_weights = _importance_weights(score_sample_weights, length(y))
    splits = _importance_folds(length(y), n_splits, folds, true, random_state)
    _importance_permutations(permutations, splits, groups)
    decreases = zeros(length(splits), length(groups))
    fitted = estimator
    for (fold, (train, test)) in enumerate(splits)
        fitted = fit_estimator(fitted, x[train, :], y[train], training_weights[train])
        classes = estimator_classes(fitted)
        baseline = -_importance_log_loss(y[test], predict_proba(fitted, x[test, :]), scoring_weights[test], classes)
        rng = MersenneTwister(random_state + fold - 1)
        for (column, group) in enumerate(groups)
            isempty(group) && continue
            permutation = permutations === nothing ? randperm(rng, length(test)) : Int.(permutations[fold][column])
            shuffled = x[test, :]
            shuffled[:, group] = shuffled[permutation, group]
            loss = _importance_log_loss(y[test], predict_proba(fitted, shuffled), scoring_weights[test], classes)
            decreases[fold, column] = baseline + loss
        end
    end
    return merge(_importance_summary(decreases), (names = names, estimator = fitted))
end

"""
    feature_importance_mda(estimator, x, y; fit_estimator, predict_proba,
        estimator_classes, train_sample_weights=nothing, score_sample_weights=nothing,
        n_splits=10, random_state=42, folds=nothing, permutations=nothing)

Caller-estimator single-column permutation importance. Uses the same weighted
loss, sequential fit state and supplied-draw contract as clustered MDA.
"""
function feature_importance_mda(estimator, x::AbstractMatrix, y::AbstractVector; kwargs...)
    result = clustered_feature_importance_mda(estimator, x, y, [[column] for column in axes(x, 2)]; kwargs...)
    return (mean = result.mean, std = result.std, estimator = result.estimator)
end

"""
    feature_importance_sfi(estimator, x, y; fit_estimator, predict=nothing,
        predict_proba=nothing, estimator_classes=nothing, scoring=:log_loss,
        train_sample_weights=nothing, score_sample_weights=nothing,
        n_splits=10, folds=nothing)

Fit the caller estimator to each single feature over contiguous K-fold splits
and aggregate weighted accuracy or negative log loss. Fit state advances in
feature-major, fold-minor order as in the reference. Supplied folds can replay
the reference partition. Return mean scores, standard errors and the final
fitted estimator; no forest or alternative score is selected implicitly.
"""
function feature_importance_sfi(estimator, x::AbstractMatrix, y::AbstractVector; fit_estimator, predict = nothing, predict_proba = nothing, estimator_classes = nothing, scoring::Symbol = :log_loss, train_sample_weights = nothing, score_sample_weights = nothing, n_splits::Integer = 10, folds = nothing)
    _importance_input_shape(x, y)
    scoring in (:log_loss, :accuracy) || throw(ArgumentError("scoring must be :log_loss or :accuracy"))
    scoring == :accuracy && predict === nothing && throw(ArgumentError("accuracy requires a prediction callback"))
    scoring == :log_loss && (predict_proba === nothing || estimator_classes === nothing) && throw(ArgumentError("log loss requires probability and class callbacks"))
    training_weights = _importance_weights(train_sample_weights, length(y))
    scoring_weights = _importance_weights(score_sample_weights, length(y))
    splits = _importance_folds(length(y), n_splits, folds, false, 0)
    scores = zeros(length(splits), size(x, 2))
    fitted = estimator
    for column in axes(x, 2)
        for (fold, (train, test)) in enumerate(splits)
            fitted = fit_estimator(fitted, x[train, column:column], y[train], training_weights[train])
            if scoring == :log_loss
                scores[fold, column] = -_importance_log_loss(y[test], predict_proba(fitted, x[test, column:column]), scoring_weights[test], estimator_classes(fitted))
            else
                predictions = predict(fitted, x[test, column:column])
                predictions isa AbstractVector && length(predictions) == length(test) || throw(DimensionMismatch("predictions must align with test observations"))
                sum(scoring_weights[test]) > 0 || throw(ArgumentError("scoring weights must have positive sum"))
                scores[fold, column] = sum(scoring_weights[test] .* (predictions .== y[test])) / sum(scoring_weights[test])
            end
        end
    end
    return merge(_importance_summary(scores), (estimator = fitted,))
end
