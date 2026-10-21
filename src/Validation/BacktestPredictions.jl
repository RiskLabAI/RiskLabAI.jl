function _prediction_layout(cv::Union{CombinatorialPurgedCV,AdaptiveCombinatorialPurgedCV}, n_samples; permutation = nothing)
    permutation === nothing || throw(ArgumentError("permutation is only supported for shuffled K-fold"))
    length(cv.event_starts) == n_samples || throw(DimensionMismatch("events must align with observations"))
    _validate_event_ranges(cv.event_starts, cv.event_ends; embargo = cv.embargo)
    1 <= cv.n_test_groups < cv.n_splits <= n_samples || throw(ArgumentError("invalid combinatorial group counts"))
    segments = _validation_segments(cv)
    splits = _combinatorial_cv_split(cv, segments)
    locations = _path_locations(cv.n_splits, collect(combinations(1:cv.n_splits, cv.n_test_groups)))
    paths = Dict(path => [(collect(segments[group]), split) for (group, split) in coordinates] for (path, coordinates) in locations)
    return splits, paths
end

function _prediction_layout(cv::Union{KFoldCV,PurgedKFoldCV,WalkForwardCV}, n_samples; permutation = nothing)
    if cv isa KFoldCV
        splits = cv_split(cv, n_samples; permutation)
    elseif cv isa WalkForwardCV
        permutation === nothing || throw(ArgumentError("permutation is only supported for shuffled K-fold"))
        splits = cv_split(cv, n_samples)
    else
        permutation === nothing || throw(ArgumentError("permutation is only supported for shuffled K-fold"))
        length(cv.event_starts) == n_samples || throw(DimensionMismatch("events must align with observations"))
        _validate_event_ranges(cv.event_starts, cv.event_ends; embargo = cv.embargo)
        1 <= cv.n_splits <= n_samples || throw(ArgumentError("invalid fold count"))
        splits = cv_split(cv)
    end
    return splits, Dict(1 => [(test, split) for (split, (_, test)) in enumerate(splits)])
end

function _checked_path_prediction(model, data, predict_function, probability)
    result = predict_function(model, data)
    expected_dimensions = probability ? 2 : 1
    result isa AbstractArray && ndims(result) == expected_dimensions || throw(DimensionMismatch("prediction callback returned the wrong rank"))
    size(result, 1) == size(data, 1) || throw(DimensionMismatch("predictions must align with test observations"))
    return copy(result)
end

"""
    backtest_predictions(cv, estimator, x, y; clone_estimator, fit_estimator,
        predict, sample_weights=nothing, predict_probability=false,
        predict_proba=nothing, permutation=nothing)

Fit caller-selected estimators and assemble the released validation paths.
`clone_estimator(estimator)` must return an independent estimator with the desired
reset semantics. `fit_estimator(model, x_train, y_train, weights)` returns its
fitted model. `predict(model, x_test)` returns a vector; `predict_proba` returns a
matrix with a consistent class-column order. Each fit receives the aligned
observation weights, or ones when omitted; callback failures propagate.

Combinatorial validators fit once per combination and reuse that model for all
of its path segments. Paths have integer keys and chronological group order.
Shuffled K-fold restores original observation order and accepts a supplied
one-based permutation for cross-language comparisons. Walk-forward folds with
no training observations produce NaNs. Execution is serial. This interface does
not choose an estimator, replace observation weights, or silently retry an
unweighted fit. A dictionary of validators, estimators, matrices and label
vectors supports independent datasets under identical keys.
"""
function backtest_predictions(cv, estimator, x::AbstractMatrix, y::AbstractVector; clone_estimator, fit_estimator, predict, sample_weights = nothing, predict_probability::Bool = false, predict_proba = nothing, permutation = nothing)
    n_samples = size(x, 1)
    length(y) == n_samples || throw(DimensionMismatch("labels must align with observations"))
    weights = sample_weights === nothing ? ones(n_samples) : Float64.(sample_weights)
    length(weights) == n_samples || throw(DimensionMismatch("weights must align with observations"))
    all(value -> isfinite(value) && value >= 0, weights) || throw(ArgumentError("sample weights must be finite and nonnegative"))
    probability_function = predict_probability ? predict_proba : predict
    probability_function === nothing && throw(ArgumentError("probability predictions require predict_proba"))
    splits, paths = _prediction_layout(cv, n_samples; permutation)
    models = map(splits) do (train, _)
        cv isa WalkForwardCV && isempty(train) && return nothing
        model = clone_estimator(estimator)
        return fit_estimator(model, x[train, :], y[train], weights[train])
    end
    predictions = Dict{Int,Any}()
    for (path, coordinates) in paths
        parts = map(coordinates) do (test, split)
            if cv isa WalkForwardCV && isempty(splits[split][1])
                return predict_probability ? fill(NaN, length(test), length(unique(y))) : fill(NaN, length(test))
            end
            return _checked_path_prediction(models[split], x[test, :], probability_function, predict_probability)
        end
        values = reduce(vcat, parts)
        if cv isa KFoldCV && cv.shuffle
            positions = reduce(vcat, [test for (test, _) in coordinates])
            ordering = sortperm(positions)
            values = predict_probability ? values[ordering, :] : values[ordering]
        end
        predictions[path] = values
    end
    return predictions
end

function backtest_predictions(validators::AbstractDict, estimators::AbstractDict, x::AbstractDict, y::AbstractDict; sample_weights = nothing, kwargs...)
    keyset = Set(keys(x))
    all(Set(keys(value)) == keyset for value in (validators, estimators, y)) || throw(ArgumentError("dataset keys must match"))
    sample_weights === nothing || sample_weights isa AbstractDict && Set(keys(sample_weights)) == keyset || throw(ArgumentError("sample-weight keys must match"))
    return Dict(key => backtest_predictions(validators[key], estimators[key], x[key], y[key]; sample_weights = sample_weights === nothing ? nothing : sample_weights[key], kwargs...) for key in keys(x))
end
