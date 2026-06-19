"""
Bagging accuracy — native Julia port mirroring the Python `RiskLabAI.ensemble`
sub-package (López de Prado, AFML Ch. 6): the theoretical accuracy of a
majority-vote bagging classifier, plus an empirical weighted-bagging evaluator.

Parity notes:
  * `bagging_classifier_accuracy` is **deterministic** and matches Python exactly
    (binomial survival function; verified in `test/runtests.jl`).
  * `bagging_evaluate_schemes` and `calculate_bootstrap_accuracy` are
    **behavioural** — they build a bagging ensemble on the `DecisionTree.jl`
    backend (not bit-identical to scikit-learn) and are validated structurally.

Deliberate divergence: the sklearn `BaggingClassifierAccuracy` class becomes a
small functional API; `class_weight="balanced"` and the Matplotlib plotting
helper are dropped (no class weights / plotting in the backend).

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 6.
"""

using Statistics: std
using Random: AbstractRNG, MersenneTwister, default_rng
using Distributions: Binomial, cdf
using DecisionTree: build_tree, apply_tree

"""
    bagging_classifier_accuracy(N, p) -> Float64

Theoretical accuracy of a majority vote of `N` independent classifiers each with
accuracy `p`: `P(X > ⌊N/2⌋)` for `X ~ Binomial(N, p)`. `N` must be odd.
Deterministic. Mirrors Python's `bagging_classifier_accuracy`.
"""
function bagging_classifier_accuracy(N::Integer, p::Real)
    isodd(N) || throw(ArgumentError("Number of estimators N must be odd. Got $N."))
    k = (N - 1) ÷ 2
    return 1.0 - cdf(Binomial(N, p), k)
end

_ensemble_rng(random_state) =
    random_state === nothing ? default_rng() :
    random_state isa AbstractRNG ? random_state : MersenneTwister(random_state)

"""
    fit_bagging(x, y; n_estimators=1000, max_samples=100, max_features=1, random_state=nothing)
        -> (trees, classes)

Fit a bagging ensemble: `n_estimators` decision trees, each trained on a
bootstrap of `max_samples` rows using `max_features` random features per split.
Returns the trees and the sorted class labels. Behavioural (binary
classification). Mirrors the fit step of Python's `BaggingClassifierAccuracy`.
"""
function fit_bagging(
    x::AbstractMatrix{<:Real},
    y::AbstractVector;
    n_estimators::Integer = 1000,
    max_samples::Integer = 100,
    max_features::Integer = 1,
    random_state = nothing,
)
    n = size(x, 1)
    classes = sort(unique(y))
    length(classes) == 2 || throw(ArgumentError("only binary classification is supported"))
    rng = _ensemble_rng(random_state)
    draw = min(max_samples, n)
    trees = Any[]
    for _ = 1:n_estimators
        sample = rand(rng, 1:n, draw)
        push!(trees, build_tree(y[sample], x[sample, :], max_features, -1, 1, 2, 0.0; rng = rng))
    end
    return trees, classes
end

_tree_accuracy(tree, x, y) = sum(apply_tree(tree, x) .== y) / length(y)

# Weighted signed-vote prediction: class_1 → +1, class_0 → −1, sum, threshold 0.
function _bagging_predict(trees, x, weights, classes)
    n = size(x, 1)
    votes = zeros(Float64, n)
    for (j, tree) in enumerate(trees)
        predictions = apply_tree(tree, x)
        for i = 1:n
            votes[i] += weights[j] * (predictions[i] == classes[2] ? 1.0 : -1.0)
        end
    end
    return [v > 0 ? classes[2] : classes[1] for v in votes]
end

"""
    bagging_evaluate_schemes(x_train, y_train, x_test, y_test; kwargs...) -> Dict{String,Float64}

Fit a bagging ensemble and return its test accuracy under three estimator
weighting schemes: `"uniform"`, `"c_i"` (proportional to each tree's training
accuracy) and `"variance"` (proportional to `1 − cᵢ²`). Behavioural. Mirrors
Python's `BaggingClassifierAccuracy.evaluate_all_schemes`.
"""
function bagging_evaluate_schemes(
    x_train::AbstractMatrix{<:Real},
    y_train::AbstractVector,
    x_test::AbstractMatrix{<:Real},
    y_test::AbstractVector;
    n_estimators::Integer = 1000,
    max_samples::Integer = 100,
    max_features::Integer = 1,
    random_state = nothing,
)
    trees, classes = fit_bagging(
        x_train, y_train;
        n_estimators = n_estimators,
        max_samples = max_samples,
        max_features = max_features,
        random_state = random_state,
    )

    c_i = [_tree_accuracy(tree, x_train, y_train) for tree in trees]
    n = length(c_i)
    uniform = fill(1.0 / n, n)
    sum_c = sum(c_i)
    weights_c = sum_c == 0 ? uniform : c_i ./ sum_c
    variance = 1.0 .- c_i .^ 2
    sum_v = sum(variance)
    weights_v = sum_v == 0 ? uniform : variance ./ sum_v

    schemes = Dict("uniform" => uniform, "c_i" => weights_c, "variance" => weights_v)
    accuracies = Dict{String,Float64}()
    for (name, weights) in schemes
        predictions = _bagging_predict(trees, x_test, weights, classes)
        accuracies[name] = sum(predictions .== y_test) / length(y_test)
    end
    return accuracies
end

"""
    calculate_bootstrap_accuracy(trees, classes, x, y; weights=uniform, n_bootstraps=1000, random_state=nothing)
        -> (values, mean, std)

Bootstrap the test set `n_bootstraps` times and return the ensemble accuracy on
each resample, with its mean and (sample) standard deviation. Behavioural.
Mirrors Python's `calculate_bootstrap_accuracy`.
"""
function calculate_bootstrap_accuracy(
    trees,
    classes,
    x::AbstractMatrix{<:Real},
    y::AbstractVector;
    weights::AbstractVector{<:Real} = fill(1.0 / length(trees), length(trees)),
    n_bootstraps::Integer = 1000,
    random_state = nothing,
)
    rng = _ensemble_rng(random_state)
    n = length(y)
    values = Float64[]
    for _ = 1:n_bootstraps
        sample = rand(rng, 1:n, n)
        predictions = _bagging_predict(trees, x[sample, :], weights, classes)
        push!(values, sum(predictions .== y[sample]) / n)
    end
    return values, sum(values) / length(values), std(values; corrected = true)
end
