using Random: randperm

"""
    BaggedCombinatorialPurgedCV(n_splits, n_test_groups, event_starts, event_ends;
        embargo=0.0, classifier=true, n_estimators=10, max_samples=1.0,
        max_features=1.0, bootstrap=true, bootstrap_features=false,
        random_seed=nothing)

Fit a caller-selected bagged estimator for each combinatorial purged split.
Integer sample/feature sizes are counts; floating sizes are fractions. Under
the validated scikit-learn 1.9.1 contract, fractional sample counts use the sum
of training weights. Sample weights determine row-draw probabilities. A
weight-aware base estimator receives draw multiplicities, while an explicitly
unweighted base estimator receives resampled rows. The wrapped index validator
is available as `base`.
"""
struct BaggedCombinatorialPurgedCV
    base::CombinatorialPurgedCV
    classifier::Bool
    n_estimators::Int
    max_samples::Union{Int,Float64}
    max_features::Union{Int,Float64}
    bootstrap::Bool
    bootstrap_features::Bool
    random_seed::Union{Nothing,Int}
end

function BaggedCombinatorialPurgedCV(n_splits::Integer, n_test_groups::Integer, event_starts::AbstractVector, event_ends::AbstractVector; embargo::Real = 0.0, classifier::Bool = true, n_estimators::Integer = 10, max_samples::Real = 1.0, max_features::Real = 1.0, bootstrap::Bool = true, bootstrap_features::Bool = false, random_seed::Union{Nothing,Integer} = nothing)
    _validate_event_ranges(event_starts, event_ends; embargo)
    1 <= n_test_groups < n_splits <= length(event_starts) || throw(ArgumentError("invalid combinatorial group counts"))
    n_estimators >= 1 || throw(ArgumentError("n_estimators must be positive"))
    for (name, value) in (("max_samples", max_samples), ("max_features", max_features))
        (value isa Integer ? value >= 1 : isfinite(value) && 0 < value <= 1) || throw(ArgumentError("$name must be a positive count or a fraction in (0, 1]"))
    end
    base = CombinatorialPurgedCV(n_splits, n_test_groups, event_starts, event_ends; embargo)
    return BaggedCombinatorialPurgedCV(base, classifier, Int(n_estimators), max_samples isa Integer ? Int(max_samples) : Float64(max_samples), max_features isa Integer ? Int(max_features) : Float64(max_features), bootstrap, bootstrap_features, random_seed === nothing ? nothing : Int(random_seed))
end

cv_split(cv::BaggedCombinatorialPurgedCV) = cv_split(cv.base)
backtest_paths(cv::BaggedCombinatorialPurgedCV) = backtest_paths(cv.base)
get_n_splits(cv::BaggedCombinatorialPurgedCV) = get_n_splits(cv.base)
_cv_splits(cv::BaggedCombinatorialPurgedCV, _) = cv_split(cv)

function _bagging_sizes(cv, weights, n_features)
    isempty(weights) && throw(ArgumentError("bagging requires nonempty training data"))
    all(value -> isfinite(value) && value >= 0, weights) && sum(weights) > 0 || throw(ArgumentError("training weights must be finite, nonnegative and have positive sum"))
    sample_count = cv.max_samples isa Integer ? cv.max_samples : max(1, floor(Int, cv.max_samples * sum(weights)))
    feature_count = cv.max_features isa Integer ? cv.max_features : max(1, floor(Int, cv.max_features * n_features))
    feature_count <= n_features || throw(ArgumentError("max_features exceeds the available features"))
    cv.bootstrap || sample_count <= count(>(0), weights) || throw(ArgumentError("sampling without replacement needs enough positive-weight observations"))
    return sample_count, feature_count
end

function _bagging_weighted_draws(rng, weights, count, replace)
    available = copy(weights)
    indices = Int[]
    for _ in 1:count
        cumulative = cumsum(available)
        total = cumulative[end]
        threshold = min(rand(rng) * total, prevfloat(total))
        position = searchsortedlast(cumulative, threshold) + 1
        push!(indices, position)
        replace || (available[position] = 0.0)
    end
    return indices
end

function _bagging_supplied_plan(values, n_combinations, n_estimators, name)
    values === nothing && return nothing
    length(values) == n_combinations && all(length(value) == n_estimators for value in values) || throw(DimensionMismatch("$name must provide one entry per estimator in every combination"))
    return values
end

function _bagging_draw_plan(cv, splits, weights, n_features; sample_indices, feature_indices, estimator_seeds, rng)
    sample_indices = _bagging_supplied_plan(sample_indices, length(splits), cv.n_estimators, "sample_indices")
    feature_indices = _bagging_supplied_plan(feature_indices, length(splits), cv.n_estimators, "feature_indices")
    estimator_seeds = _bagging_supplied_plan(estimator_seeds, length(splits), cv.n_estimators, "estimator_seeds")
    return map(enumerate(splits)) do (split, (train, _))
        local_rng = cv.random_seed === nothing ? rng : MersenneTwister(cv.random_seed)
        train_weights = weights[train]
        sample_count, feature_count = _bagging_sizes(cv, train_weights, n_features)
        map(1:cv.n_estimators) do member
            seed = estimator_seeds === nothing ? rand(local_rng, 0:(typemax(Int32)-1)) : Int(estimator_seeds[split][member])
            features = if feature_indices !== nothing
                Int.(feature_indices[split][member])
            elseif cv.bootstrap_features
                rand(local_rng, 1:n_features, feature_count)
            elseif feature_count == n_features
                collect(1:n_features)
            else
                randperm(local_rng, n_features)[1:feature_count]
            end
            samples = sample_indices === nothing ? _bagging_weighted_draws(local_rng, train_weights, sample_count, cv.bootstrap) : Int.(sample_indices[split][member])
            length(features) == feature_count && all(index -> 1 <= index <= n_features, features) || throw(ArgumentError("invalid supplied feature indices"))
            length(samples) == sample_count && all(index -> 1 <= index <= length(train), samples) || throw(ArgumentError("invalid supplied sample indices"))
            cv.bootstrap_features || allunique(features) || throw(ArgumentError("feature draws without replacement must be distinct"))
            cv.bootstrap || allunique(samples) || throw(ArgumentError("row draws without replacement must be distinct"))
            all(train_weights[index] > 0 for index in samples) || throw(ArgumentError("a zero-weight observation cannot be sampled"))
            return (samples = samples, features = features, seed = seed)
        end
    end
end

function _fit_bagged_path(cv, prototype, x, y, plan, clone_estimator, fit_estimator, fit_sample_weights, estimator_classes)
    classes = cv.classifier ? sort(unique(y)) : []
    encoded = cv.classifier ? [searchsortedfirst(classes, label)-1 for label in y] : y
    members = map(plan) do draw
        model = applicable(clone_estimator, prototype, draw.seed) ? clone_estimator(prototype, draw.seed) : clone_estimator(prototype)
        if fit_sample_weights
            multiplicities = zeros(Float64, length(y))
            for position in draw.samples
                multiplicities[position] += 1
            end
            train_x, train_y, train_weights = x[:, draw.features], copy(encoded), multiplicities
        else
            train_x, train_y, train_weights = x[draw.samples, draw.features], encoded[draw.samples], nothing
        end
        fitted = fit_estimator(model, train_x, train_y, train_weights)
        codes = cv.classifier ? (estimator_classes === nothing ? sort(unique(train_y)) : collect(estimator_classes(fitted))) : []
        if cv.classifier
            all(code -> code isa Integer && 0 <= code < length(classes), codes) && allunique(codes) || throw(ArgumentError("estimator class codes must index the zero-based encoded labels"))
        end
        return (model = fitted, features = copy(draw.features), codes = Int.(codes))
    end
    return (members = members, classes = classes)
end

function _bagged_probability(fitted, x, predict_function, predict_proba)
    probability = zeros(size(x, 1), length(fitted.classes))
    for member in fitted.members
        data = x[:, member.features]
        if predict_proba === nothing
            predicted = _checked_path_prediction(member.model, data, predict_function, false)
            for (row, code) in enumerate(predicted)
                code isa Integer && 0 <= code < length(fitted.classes) || throw(ArgumentError("classifier predictions must be zero-based encoded labels"))
                probability[row, code+1] += 1
            end
        else
            values = _checked_path_prediction(member.model, data, predict_proba, true)
            size(values, 2) == length(member.codes) || throw(DimensionMismatch("probability columns must align with estimator class codes"))
            probability[:, member.codes .+ 1] .+= values
        end
    end
    return probability ./ length(fitted.members)
end

function _bagged_regression(fitted, x, predict_function)
    values = zeros(size(x, 1))
    for member in fitted.members
        values .+= _checked_path_prediction(member.model, x[:, member.features], predict_function, false)
    end
    return values ./ length(fitted.members)
end

"""
    backtest_predictions(cv::BaggedCombinatorialPurgedCV, estimator, x, y;
        clone_estimator, fit_estimator, predict, predict_proba=nothing,
        estimator_classes=nothing, fit_sample_weights=true, sample_weights=nothing,
        predict_probability=false, sample_indices=nothing, feature_indices=nothing,
        estimator_seeds=nothing, rng=default_rng())

Assemble serial bagged CPCV paths using explicit caller-estimator callbacks.
`clone_estimator(prototype, seed)` can receive a fresh native seed; a one-argument
callback is also accepted for deterministic estimators. Shared draws and seeds
are nested as `[combination][estimator]`, with one-based positions relative to
the combination's training rows and the original feature columns. Supplied
draws must follow the requested replacement rules and avoid zero-weight rows.
Cross-language seeds alone do not specify identical random streams.

Classifier targets are encoded as `0:k-1`, matching the public bagging wrapper;
returned labels use the original sorted classes. `predict_proba` columns default
to sorted encoded classes present in each member's fit; `estimator_classes`
can declare a different order. When `predict_proba` is absent, hard class votes
are averaged. Regressors average their predictions and reject probability
requests. Set `fit_sample_weights=false` only for an intentionally unweighted
estimator: the fitting callback then receives sampled rows and `nothing` as its
weight argument. All fitting failures propagate without an unweighted retry.
"""
function backtest_predictions(cv::BaggedCombinatorialPurgedCV, estimator, x::AbstractMatrix, y::AbstractVector; clone_estimator, fit_estimator, predict, predict_proba = nothing, estimator_classes = nothing, fit_sample_weights::Bool = true, sample_weights = nothing, predict_probability::Bool = false, sample_indices = nothing, feature_indices = nothing, estimator_seeds = nothing, rng::AbstractRNG = default_rng())
    !cv.classifier && predict_probability && throw(ArgumentError("regressors do not produce class probabilities"))
    length(y) == size(x, 1) || throw(DimensionMismatch("labels must align with observations"))
    weights = sample_weights === nothing ? ones(size(x, 1)) : Float64.(sample_weights)
    length(weights) == size(x, 1) || throw(DimensionMismatch("weights must align with observations"))
    splits, _ = _prediction_layout(cv.base, size(x, 1))
    plans = _bagging_draw_plan(cv, splits, weights, size(x, 2); sample_indices, feature_indices, estimator_seeds, rng)
    split = Ref(0)
    fit_bag = function(prototype, train_x, train_y, train_weights)
        split[] += 1
        return _fit_bagged_path(cv, prototype, train_x, train_y, plans[split[]], clone_estimator, fit_estimator, fit_sample_weights, estimator_classes)
    end
    probability = (fitted, data) -> _bagged_probability(fitted, data, predict, predict_proba)
    prediction = if cv.classifier
        (fitted, data) -> [fitted.classes[argmax(row)] for row in eachrow(probability(fitted, data))]
    else
        (fitted, data) -> _bagged_regression(fitted, data, predict)
    end
    return backtest_predictions(cv.base, estimator, x, y; clone_estimator = identity, fit_estimator = fit_bag, predict = prediction, predict_proba = probability, predict_probability, sample_weights = weights)
end
