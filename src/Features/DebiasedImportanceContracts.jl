"""
    mdi_plus_importance(estimator, x, y; fit_estimator, tree_records, n_trees=60)

Fit the caller ensemble with `fit_estimator(model, x, y, nothing)` and obtain
ordered tree records from `tree_records(fitted_model)`. Each record contains
one-based `features`, matching `thresholds`, and one-based `bootstrap_indices`.
The released simplified MDI+ calculation uses globally centered stump bases,
unique in-bag rows, intercept ridge with penalty one, and out-of-bag partial
population variances. Trees with no splits or fewer than five out-of-bag rows
are skipped. The estimator family and bootstrap draws remain caller choices.
"""
function mdi_plus_importance(estimator, x::AbstractMatrix, y::AbstractVector; fit_estimator, tree_records, n_trees::Integer = 60)
    _importance_input_shape(x, y)
    n_trees >= 0 || throw(ArgumentError("n_trees must be nonnegative."))
    xv, yv = Matrix{Float64}(x), Vector{Float64}(y)
    n, p = size(xv)
    fitted = fit_estimator(estimator, copy(x), copy(y), nothing)
    records = collect(tree_records(fitted))
    accumulated = zeros(p)
    used = 0
    for row in Iterators.take(records, n_trees)
        features = Vector{Int}(row.features)
        thresholds = Vector{Float64}(row.thresholds)
        length(features) == length(thresholds) || throw(DimensionMismatch("Each split needs one threshold."))
        all(feature -> 1 <= feature <= p, features) || throw(ArgumentError("Split features must index the data columns."))
        isempty(features) && continue
        in_bag = sort(unique(Vector{Int}(row.bootstrap_indices)))
        !isempty(in_bag) && all(index -> 1 <= index <= n, in_bag) || throw(ArgumentError("Bootstrap positions must index the data rows."))
        out_of_bag = setdiff(1:n, in_bag)
        length(out_of_bag) < 5 && continue
        basis = Float64[xv[index, features[column]] <= thresholds[column] for index in 1:n, column in eachindex(features)]
        basis .-= mean(basis; dims = 1)
        coefficients, _, _ = _ridge_coefficients(basis[in_bag, :], yv[in_bag])
        for feature in unique(features)
            selected = features .== feature
            partial = basis[out_of_bag, selected] * coefficients[selected]
            accumulated[feature] += _var_pop(partial)
        end
        used += 1
    end
    return accumulated ./ max(used, 1)
end

function _cpi_summary(differences)
    importance = zeros(length(differences))
    p_value = ones(length(differences))
    for (feature, values) in enumerate(differences)
        length(values) >= 2 || throw(ArgumentError("CPI requires at least two held-out loss differences."))
        importance[feature] = mean(values)
        if _std_pop(values) >= 1e-12
            statistic = importance[feature] / (std(values; corrected = true) / sqrt(length(values)))
            p_value[feature] = ccdf(TDist(length(values) - 1), statistic)
        end
    end
    return (importance = importance, p_value = p_value)
end

function _cpi_splits(n, n_splits, random_state, supplied)
    if supplied === nothing
        2 <= n_splits <= n || throw(ArgumentError("n_splits must be between two and the observation count."))
        return [(sort(train), sort(test)) for (train, test) in _kfold(n, n_splits; shuffle = true, rng = MersenneTwister(random_state))]
    end
    splits = [(Vector{Int}(first(pair)), Vector{Int}(last(pair))) for pair in supplied]
    for (train, test) in splits
        !isempty(train) && !isempty(test) || throw(ArgumentError("Every CPI fold needs training and test observations."))
        sort(vcat(train, test)) == collect(1:n) || throw(ArgumentError("Each CPI fold must partition all observations without overlap."))
    end
    sort(reduce(vcat, (test for (_, test) in splits); init = Int[])) == collect(1:n) || throw(ArgumentError("CPI test folds must cover each observation exactly once."))
    return splits
end

"""
    conditional_predictive_impact(estimator, x, y; clone_estimator,
        fit_estimator, predict_proba, estimator_classes, random_state=0,
        n_splits=4, splits=nothing, gaussian_draws=nothing)

Clone and fit the caller classifier once per fold, retaining its class-column
order. Callbacks are `clone_estimator(model)`, `fit_estimator(model, x, y, nothing)`,
`predict_proba(model, x)`, and `estimator_classes(model)`. Optional one-based
`(train, test)` splits and per-fold test-row by feature Gaussian matrices replay
the public computation. Conditional ridge uses penalty one and an intercept;
residual population deviation receives the public additive `1e-9`. Losses are
clipped to `[1e-15, 1-1e-15]` without renormalization. Pooled loss increases use
the public population-deviation gate and one-sided paired Student test.
This interface validates arithmetic conditional on the caller model and draws;
it does not certify empirical calibration or default forest equivalence.
"""
function conditional_predictive_impact(estimator, x::AbstractMatrix, y::AbstractVector; clone_estimator, fit_estimator, predict_proba, estimator_classes, random_state::Integer = 0, n_splits::Integer = 4, splits = nothing, gaussian_draws = nothing)
    _importance_input_shape(x, y)
    xv = Matrix{Float64}(x)
    n, p = size(xv)
    p >= 2 || throw(ArgumentError("Conditional ridge requires at least two feature columns."))
    folds = _cpi_splits(n, n_splits, random_state, splits)
    gaussian_draws === nothing || length(gaussian_draws) == length(folds) || throw(DimensionMismatch("Supply one Gaussian matrix per fold."))
    differences = [Float64[] for _ in 1:p]
    for (fold, (train, test)) in enumerate(folds)
        model = fit_estimator(clone_estimator(estimator), copy(xv[train, :]), copy(y[train]), nothing)
        classes = collect(estimator_classes(model))
        original = Matrix{Float64}(predict_proba(model, copy(xv[test, :])))
        size(original) == (length(test), length(classes)) || throw(DimensionMismatch("Probability columns must match the supplied classes."))
        loss_true = _per_sample_log_loss(y[test], original, classes)
        rng = MersenneTwister(random_state + 100 + fold - 1)
        draws = gaussian_draws === nothing ? randn(rng, length(test), p) : Matrix{Float64}(gaussian_draws[fold])
        size(draws) == (length(test), p) || throw(DimensionMismatch("Gaussian draws must match test observations and features."))
        for feature in 1:p
            others = [column for column in 1:p if column != feature]
            coefficients, xbar, ybar = _ridge_coefficients(xv[train, others], xv[train, feature])
            residual = xv[train, feature] .- _ridge_predict(coefficients, xbar, ybar, xv[train, others])
            sigma = _std_pop(residual) + 1e-9
            knockoff = copy(xv[test, :])
            knockoff[:, feature] = _ridge_predict(coefficients, xbar, ybar, xv[test, others]) .+ sigma .* draws[:, feature]
            probabilities = Matrix{Float64}(predict_proba(model, knockoff))
            size(probabilities) == size(original) || throw(DimensionMismatch("Knockoff probability dimensions changed."))
            append!(differences[feature], _per_sample_log_loss(y[test], probabilities, classes) .- loss_true)
        end
    end
    return _cpi_summary(differences)
end
