"""
    _purged_cv_score(estimator_factory, params, x, y, cv; fit_estimator,
        predict=nothing, scoring=nothing)

Score a fresh caller estimator on each eligible purged fold. The public contract
skips folds with fewer than 25 training rows, fewer than five testing rows or
fewer than two training classes. Fits receive no observation weights. A custom
`scoring(model, x_test, y_test)` replaces accuracy; an empty score list returns
negative infinity. Exceptions propagate.
"""
function _purged_cv_score(estimator_factory, params, x, y, cv::PurgedKFoldCV;
    fit_estimator, predict = nothing, scoring = nothing)
    size(x, 1) == length(y) == length(cv.event_starts) == length(cv.event_ends) || throw(DimensionMismatch("features, labels and event intervals must align"))
    scoring === nothing && predict === nothing && throw(ArgumentError("provide predict or a custom scoring callback"))
    scores = Float64[]
    for (train, test) in cv_split(cv)
        (length(train) < 25 || length(test) < 5 || length(unique(y[train])) < 2) && continue
        model = estimator_factory(params)
        fitted = fit_estimator(model, x[train, :], y[train])
        fitted !== nothing && (model = fitted)
        value = if scoring === nothing
            predicted = predict(model, x[test, :])
            length(predicted) == length(test) || throw(DimensionMismatch("prediction count must match the testing fold"))
            mean(predicted .== y[test])
        else
            scoring(model, x[test, :], y[test])
        end
        push!(scores, Float64(value))
    end
    return isempty(scores) ? -Inf : mean(scores)
end

function _hpo_trial_summary(history, n_trials)
    completed = filter(trial -> trial.state == :complete, history)
    isempty(completed) && throw(ArgumentError("no completed trial is available"))
    best = first(completed)
    for trial in Iterators.drop(completed, 1)
        trial.score > best.score && (best = trial)
    end
    scores = Float64[trial.score for trial in history if isfinite(trial.score)]
    return (best_params = deepcopy(best.params), best_score = best.score,
        n_trials = n_trials, trial_scores = scores,
        mean_trial_score = isempty(scores) ? NaN : mean(scores),
        std_trial_score = isempty(scores) ? NaN : std(scores; corrected = false))
end

"""
    leakage_aware_hpo(suggest_params, estimator_factory, x, y; event_starts,
        event_ends, ask_trial, fit_estimator, predict=nothing, scoring=nothing,
        trial_parameters=trial->trial.params, tell_trial! = (trial,score)->nothing,
        n_trials=50, n_splits=5, embargo=0.0, random_state=0)

Run leakage-controlled optimization with an explicit caller-supplied sampler.
`ask_trial(number, history, rng)` returns a trial consumed by `suggest_params`.
History contains completed/failed scores and sampled parameter dictionaries;
`tell_trial!` can update a stateful sampler. The sampled `trial_parameters` define
`best_params`, while the dictionary returned by `suggest_params` configures the
fresh estimator and may also contain fixed settings. Nonfinite scores are absent
from finite-score summaries; NaN trials fail, while infinities remain completed.
Ties retain the first completed maximum and finite-score deviation is population
deviation, including zero for a singleton.

This overload implements the public optimization orchestration and purged
objective. It requires an actual TPE/CMA-ES provider when that search family is
requested; it does not substitute random search or claim native sampler parity.
Recorded proposals support exact validation but are not fresh optimization.
"""
function leakage_aware_hpo(suggest_params, estimator_factory,
    x::AbstractMatrix, y::AbstractVector;
    event_starts::AbstractVector, event_ends::AbstractVector,
    ask_trial, fit_estimator, predict = nothing, scoring = nothing,
    trial_parameters = trial -> trial.params,
    tell_trial! = (trial, score) -> nothing,
    n_trials::Integer = 50, n_splits::Integer = 5,
    embargo::Real = 0., random_state::Integer = 0)
    n_trials >= 0 || throw(ArgumentError("n_trials must be nonnegative"))
    1 <= n_splits <= size(x, 1) || throw(ArgumentError("n_splits must be between one and the sample count"))
    cv = PurgedKFoldCV(n_splits, event_starts, event_ends; embargo = embargo)
    rng = MersenneTwister(random_state)
    history = NamedTuple[]
    for number in 1:n_trials
        trial = ask_trial(number, deepcopy(history), rng)
        configuration = suggest_params(trial)
        score = _purged_cv_score(estimator_factory, configuration, x, y, cv;
            fit_estimator = fit_estimator, predict = predict, scoring = scoring)
        params = deepcopy(trial_parameters(trial))
        tell_trial!(trial, score)
        push!(history, (number = number, params = params, score = score,
            state = isnan(score) ? :failed : :complete))
    end
    return _hpo_trial_summary(history, n_trials)
end
