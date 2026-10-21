using ..Features: _importance_log_loss

function _pipeline_grids(grid)
    grids = grid isa AbstractDict || grid isa NamedTuple ? [grid] : collect(grid)
    !isempty(grids) || throw(ArgumentError("at least one parameter grid is required"))
    return [_pipeline_options(part) for part in grids]
end

_pipeline_parameter_list(value) = value isa AbstractVector || value isa Tuple

"""Enumerate finite parameter grids with sorted keys and the last key varying fastest."""
function pipeline_parameter_grid(grid)
    candidates = Dict{String,Any}[]
    for part in _pipeline_grids(grid)
        all(_pipeline_parameter_list(value) && !isempty(value) for value in values(part)) || throw(ArgumentError("grid parameters must be nonempty vectors or tuples"))
        rows = [Dict{String,Any}()]
        for key in sort(collect(keys(part)))
            rows = [merge(row, Dict{String,Any}(key => deepcopy(value))) for row in rows for value in part[key]]
        end
        append!(candidates, rows)
    end
    return candidates
end

function _pipeline_candidates(grid, iterations, indices, draws, rng)
    iterations >= 0 || throw(ArgumentError("rnd_search_iter must be nonnegative"))
    indices !== nothing && draws !== nothing && throw(ArgumentError("supply candidate indices or draws, not both"))
    if iterations == 0
        indices === nothing && draws === nothing || throw(ArgumentError("supplied random candidates require rnd_search_iter > 0"))
        return pipeline_parameter_grid(grid)
    end
    grids = _pipeline_grids(grid)
    finite = all(_pipeline_parameter_list(value) for part in grids for value in values(part))
    if draws !== nothing
        length(draws) == iterations || throw(DimensionMismatch("candidate_draws must contain rnd_search_iter parameter dictionaries"))
        return _pipeline_options.(draws)
    elseif finite
        candidates = pipeline_parameter_grid(grid)
        count = min(iterations, length(candidates))
        positions = indices === nothing ? randperm(rng, length(candidates))[1:count] : Int.(indices)
        length(positions) == count && allunique(positions) && all(position -> 1 <= position <= length(candidates), positions) || throw(ArgumentError("candidate_indices must be distinct valid positions in the finite grid"))
        return candidates[positions]
    end
    indices === nothing || throw(ArgumentError("candidate_indices require a finite parameter grid"))
    return map(1:iterations) do _
        part = rand(rng, grids)
        Dict{String,Any}(key => deepcopy(rand(rng, part[key])) for key in sort(collect(keys(part))))
    end
end

function _pipeline_slice_options(options, rows, count)
    return Dict{String,Any}(key => value isa AbstractArray && size(value, 1) == count ? (ndims(value) == 1 ? value[rows] : value[rows, ntuple(_ -> Colon(), ndims(value)-1)...]) : deepcopy(value) for (key, value) in options)
end

function _pipeline_search_score(pipeline, x, y, weights, scoring)
    if scoring == :f1
        predicted = predict_pipeline(pipeline, x)
        length(predicted) == length(y) && all(value -> value in (0, 1), predicted) || throw(ArgumentError("binary F1 requires one binary prediction per observation"))
        true_positive = sum(weights[(y .== 1) .& (predicted .== 1)])
        false_positive = sum(weights[(y .!= 1) .& (predicted .== 1)])
        false_negative = sum(weights[(y .== 1) .& (predicted .!= 1)])
        denominator = 2true_positive + false_positive + false_negative
        return denominator == 0 ? 0.0 : 2true_positive / denominator
    end
    classes = pipeline_classes(pipeline)
    probability = predict_pipeline_proba(pipeline, x)
    length(classes) >= 2 && classes == sort(unique(classes)) || throw(ArgumentError("log-loss estimators must expose at least two sorted distinct classes"))
    size(probability) == (length(y), length(classes)) || throw(DimensionMismatch("probabilities must align with estimator classes"))
    response = length(classes) == 2 ? probability[:, 2] : probability
    return -_importance_log_loss(y, response, weights, sort(unique(y)))
end

function _pipeline_bagging(best, bagging, x, y, weights; sample_indices, feature_indices, estimator_seeds, rng)
    length(bagging) == 3 || throw(ArgumentError("bagging must contain estimator count, sample fraction and feature fraction"))
    count = trunc(Int, bagging[1])
    sample_fraction = bagging[2] > 0 ? Float64(bagging[2]) : 1.0
    feature_fraction = Float64(bagging[3])
    count >= 1 && 0 < sample_fraction <= 1 && 0 < feature_fraction <= 1 || throw(ArgumentError("invalid bagging count or fractions"))
    options = (classifier = true, n_estimators = count, max_samples = sample_fraction, max_features = feature_fraction, bootstrap = true, bootstrap_features = false, random_seed = nothing)
    prototype = clone_pipeline(best)
    fit_bag = function(name, _, data, labels, params)
        all(key == "sample_weight" for key in keys(params)) || throw(ArgumentError("post-search bagging accepts only global sample_weight without supplier metadata routing"))
        fitted_weights = get(params, "sample_weight", ones(length(labels)))
        splits = [(collect(1:length(labels)), Int[])]
        plans = _bagging_draw_plan(options, splits, fitted_weights, size(data, 2); sample_indices = sample_indices === nothing ? nothing : [sample_indices], feature_indices = feature_indices === nothing ? nothing : [feature_indices], estimator_seeds = estimator_seeds === nothing ? nothing : [estimator_seeds], rng)
        return _fit_bagged_path(options, prototype, data, labels, only(plans), (model, seed) -> clone_pipeline(model; seed), (model, tx, ty, tw) -> fit_pipeline!(model, tx, ty; sample_weight = tw), true, pipeline_classes)
    end
    probability_callback = best.callbacks.predict_proba_step === nothing ? nothing : predict_pipeline_proba
    probabilities = (_, fitted, data) -> _bagged_probability(fitted, data, predict_pipeline, probability_callback)
    predictions = (_, fitted, data) -> [fitted.classes[argmax(row)] for row in eachrow(probabilities(nothing, fitted, data))]
    bag = SampleWeightedPipeline(["bag" => nothing]; clone_step = (args...) -> nothing, fit_step = fit_bag, transform_step = (_, _, data) -> data, predict_step = predictions, predict_proba_step = probabilities, classes_step = (_, fitted) -> fitted.classes)
    return fit_pipeline!(bag, x, y; sample_weight = weights)
end

"""
    clf_hyper_fit(cv, x, y, pipeline, param_grid; sample_weight=nothing,
        fit_params=Dict(), rnd_search_iter=0, bagging=nothing,
        candidate_indices=nothing, candidate_draws=nothing, folds=nothing,
        return_search=false, error_score=NaN, rng=default_rng(),
        bagging_sample_indices=nothing, bagging_feature_indices=nothing,
        bagging_estimator_seeds=nothing)

Search a caller-selected native pipeline with explicit cross-validation. Binary
labels exactly `{0,1}` use F1 with positive label 1; other labels use negative log
loss. As in the validated scikit-learn 1.9.1 supplier, top-level sample weights
route to both fitting and scoring. Step-prefixed weights affect fitting only.
Every observation-aligned fitting array is sliced to each training fold.
Candidates and folds have independent unfitted pipeline clones. Fold means are
unweighted arithmetic means; ties select the first candidate. The winner is
refitted on every observation. Individual fit/scoring failures default to NaN;
all failed fits raise an error. `error_score=:raise` propagates the first error.

Finite randomized grids sample without replacement. Native distribution values
use `rand(rng, distribution)` and sample with replacement. One-based candidate
indices or explicit drawn parameter dictionaries replay external random draws;
equal seeds across languages do not imply identical draws. Native execution is
serial. Optional `[count, sample_fraction, feature_fraction]` bagging returns a
one-step pipeline fitted to the full data under the validated weighted bagging
contract. Supplied bagging draws are nested by member, with one-based original
row/feature positions. Extra step-prefixed fit options with bagging are rejected,
matching the supplier's default metadata-routing configuration.

The default return value is the fitted pipeline, matching the public function's
actual return. `return_search=true` additionally exposes candidate parameters,
fold scores, mean scores, first-best position and failure diagnostics.
"""
function clf_hyper_fit(cv, x::AbstractMatrix, y::AbstractVector, pipeline::SampleWeightedPipeline, param_grid; sample_weight = nothing, fit_params = Dict{String,Any}(), rnd_search_iter::Integer = 0, bagging = nothing, candidate_indices = nothing, candidate_draws = nothing, folds = nothing, return_search::Bool = false, error_score = NaN, rng::AbstractRNG = default_rng(), bagging_sample_indices = nothing, bagging_feature_indices = nothing, bagging_estimator_seeds = nothing)
    count = length(y)
    count == size(x, 1) && count >= 2 || throw(DimensionMismatch("search requires aligned observations and labels"))
    error_score === :raise || error_score isa Real || throw(ArgumentError("error_score must be numeric or :raise"))
    options = _pipeline_options(fit_params)
    haskey(options, "sample_weight") && (sample_weight === nothing && (sample_weight = options["sample_weight"]); delete!(options, "sample_weight"))
    pop!(options, "groups", nothing)
    weights = sample_weight === nothing ? ones(count) : Float64.(sample_weight)
    length(weights) == count && all(value -> isfinite(value) && value >= 0, weights) && sum(weights) > 0 || throw(ArgumentError("search weights must be finite, nonnegative and observation-aligned with positive sum"))
    sample_weight === nothing || (options["sample_weight"] = weights)
    splits = folds === nothing ? collect(_cv_splits(cv, count)) : [(Int.(first(fold)), Int.(last(fold))) for fold in folds]
    !isempty(splits) || throw(ArgumentError("search requires at least one split"))
    all(allunique(train) && allunique(test) && isempty(intersect(train, test)) && !isempty(test) && all(index -> 1 <= index <= count, [train; test]) for (train, test) in splits) || throw(ArgumentError("folds must contain disjoint valid observation positions and nonempty test sets"))
    candidates = _pipeline_candidates(param_grid, rnd_search_iter, candidate_indices, candidate_draws, rng)
    scoring = Set(y) == Set([0, 1]) ? :f1 : :neg_log_loss
    scores = fill(NaN, length(candidates), length(splits))
    failures = NamedTuple[]
    successful_fits = 0
    for (candidate, parameters) in enumerate(candidates), (fold, (train, test)) in enumerate(splits)
        model = configure_pipeline!(clone_pipeline(pipeline), parameters)
        phase = :fit
        try
            fit_pipeline!(model, x[train, :], y[train]; fit_params = _pipeline_slice_options(options, train, count))
            successful_fits += 1
            phase = :score
            scores[candidate, fold] = _pipeline_search_score(model, x[test, :], y[test], weights[test], scoring)
        catch error
            error isa InterruptException && rethrow()
            error_score === :raise && rethrow()
            scores[candidate, fold] = Float64(error_score)
            push!(failures, (candidate = candidate, fold = fold, phase = phase, message = sprint(showerror, error)))
        end
    end
    successful_fits > 0 || throw(ArgumentError("all $(length(candidates)*length(splits)) candidate fits failed"))
    means = vec(mean(scores; dims = 2))
    ranked = [isnan(value) ? -Inf : value for value in means]
    best_index = argmax(ranked)
    best = configure_pipeline!(clone_pipeline(pipeline), candidates[best_index])
    fit_pipeline!(best, x, y; fit_params = options)
    estimator = if bagging !== nothing && first(bagging) > 0
        all(key == "sample_weight" for key in keys(options)) || throw(ArgumentError("post-search bagging accepts only global sample_weight without supplier metadata routing"))
        _pipeline_bagging(best, bagging, x, y, sample_weight === nothing ? nothing : weights; sample_indices = bagging_sample_indices, feature_indices = bagging_feature_indices, estimator_seeds = bagging_estimator_seeds, rng)
    else
        best
    end
    evidence = (estimator = estimator, best_estimator = best, best_index = best_index, best_params = deepcopy(candidates[best_index]), candidates = candidates, fold_scores = scores, mean_scores = means, scoring = scoring, failures = failures)
    return return_search ? evidence : estimator
end

"""Use the public default five-fold purged search with one-percent embargo."""
function clf_hyper_fit(x::AbstractMatrix, y::AbstractVector, event_starts::AbstractVector, event_ends::AbstractVector, pipeline::SampleWeightedPipeline, param_grid; n_splits::Integer = 5, embargo::Real = 0.01, kwargs...)
    cv = PurgedKFoldCV(n_splits, event_starts, event_ends; embargo)
    return clf_hyper_fit(cv, x, y, pipeline, param_grid; kwargs...)
end
