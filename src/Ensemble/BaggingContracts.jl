function _ensemble_accuracy(predictions, labels)
    length(predictions) == length(labels) || throw(DimensionMismatch("Predictions must align with labels."))
    isempty(labels) && throw(ArgumentError("Accuracy requires observations."))
    return sum(predictions .== labels) / length(labels)
end

function _bagging_member_weights(predictions::AbstractMatrix, labels::AbstractVector)
    size(predictions, 1) == length(labels) || throw(DimensionMismatch("Member predictions must align with training labels."))
    count = size(predictions, 2)
    count > 0 || throw(ArgumentError("At least one fitted member is required."))
    accuracy = [_ensemble_accuracy(view(predictions, :, member), labels) for member in 1:count]
    uniform = fill(1.0 / count, count)
    total = sum(accuracy)
    variability = 1 .- accuracy .^ 2
    variance_total = sum(variability)
    return Dict("uniform" => uniform, "c_i" => (total == 0 ? copy(uniform) : accuracy ./ total), "variance" => (variance_total == 0 ? copy(uniform) : variability ./ variance_total))
end

function _bagging_member_prediction(predictions::AbstractMatrix, weights, classes)
    length(classes) == 2 || throw(ArgumentError("Bagging accuracy supports two external classes."))
    size(predictions, 2) == length(weights) || throw(DimensionMismatch("Weights must align with fitted members."))
    signed = ifelse.(predictions .== classes[2], 1.0, -1.0)
    votes = signed * weights
    tolerance = eps(Float64) * max(1.0, sum(abs, weights)) * max(1, length(weights))
    return [value > tolerance ? classes[2] : classes[1] for value in votes]
end

"""
    bagging_evaluate_schemes(estimator, x_train, y_train, x_test, y_test;
        fit_estimator, tree_predictions, estimator_classes)

Fit the caller-selected ensemble once through `fit_estimator(model, x, y,
nothing)`. `tree_predictions(fitted_model, x)` returns an observation-by-member
matrix; `estimator_classes(fitted_model)` returns the two ordered external
labels. Recompute fresh training accuracies and the public uniform, accuracy
and variance weights, then evaluate signed votes with zero assigned to the
first class. Member outputs are compared directly with external labels, as in
the released wrapper; encoded-label mismatches are not silently remapped.
The supplied ensemble retains responsibility for the public entropy criterion,
balanced class weights and fitted feature subsets. Existing native training
conveniences and mutable Python evaluator cache reuse are separate contracts.
"""
function bagging_evaluate_schemes(estimator, x_train::AbstractMatrix, y_train::AbstractVector, x_test::AbstractMatrix, y_test::AbstractVector; fit_estimator, tree_predictions, estimator_classes)
    size(x_train, 1) == length(y_train) && size(x_test, 1) == length(y_test) || throw(DimensionMismatch("Feature rows must align with labels."))
    fitted = fit_estimator(estimator, copy(x_train), copy(y_train), nothing)
    classes = collect(estimator_classes(fitted))
    train_predictions = tree_predictions(fitted, copy(x_train))
    schemes = _bagging_member_weights(train_predictions, y_train)
    accuracies = Dict{String,Float64}()
    for name in ("uniform", "c_i", "variance")
        predictions = tree_predictions(fitted, copy(x_test))
        accuracies[name] = _ensemble_accuracy(_bagging_member_prediction(predictions, schemes[name], classes), y_test)
    end
    return accuracies
end

"""
    calculate_bootstrap_accuracy(estimator, x, y; predict,
        n_bootstraps=1000, random_state=nothing, bootstrap_rows=nothing)

Evaluate a fitted caller predictor through `predict(model, sampled_x)` on each
resampled row plan. Optional `bootstrap_rows` is a vector of one-based row
vectors; variable lengths can replay explicit expansion of duplicated pandas
index labels. Inputs must already be aligned by position. Return all accuracies,
their mean and sample deviation; zero draws produce NaN summaries and one draw
produces a NaN sample deviation. Native random seeds are not NumPy streams.
"""
function calculate_bootstrap_accuracy(estimator, x::AbstractMatrix, y::AbstractVector; predict, n_bootstraps::Integer = 1000, random_state = nothing, bootstrap_rows = nothing)
    n = length(y)
    size(x, 1) == n || throw(DimensionMismatch("Feature rows must align with labels."))
    n > 0 && n_bootstraps >= 0 || throw(ArgumentError("Require observations and a nonnegative bootstrap count."))
    rng = _ensemble_rng(random_state)
    plans = bootstrap_rows === nothing ? [rand(rng, 1:n, n) for _ in 1:n_bootstraps] : [Vector{Int}(rows) for rows in bootstrap_rows]
    length(plans) == n_bootstraps || throw(DimensionMismatch("Bootstrap plans must match n_bootstraps."))
    values = Float64[]
    for rows in plans
        !isempty(rows) && all(index -> 1 <= index <= n, rows) || throw(ArgumentError("Bootstrap rows must be nonempty valid positions."))
        predictions = predict(estimator, copy(x[rows, :]))
        push!(values, _ensemble_accuracy(predictions, y[rows]))
    end
    average = isempty(values) ? NaN : sum(values) / length(values)
    deviation = length(values) < 2 ? NaN : std(values; corrected = true)
    return values, average, deviation
end
