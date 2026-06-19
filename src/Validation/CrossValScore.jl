"""
Cross-validation scoring — the estimator-driven companion to the cross-validators
(López de Prado, AFML Ch. 7). `cross_val_score` trains a random forest on each
train/test split produced by any of the cross-validators and returns the per-fold
score. This realises the scoring half of the Python `backtest_predictions`
machinery on the `DecisionTree.jl` backend.

Behavioural: the random forest is not bit-identical to scikit-learn, so results
are validated structurally (a separable dataset scores well; an unpredictable one
does not).
"""

using DecisionTree: build_forest, apply_forest, apply_forest_proba
using Statistics: mean

# Train/test splits for every cross-validator (KFold/WalkForward need the sample
# count; the purged validators carry their own `event_starts`).
_cv_splits(cv::KFoldCV, n_samples) = cv_split(cv, n_samples)
_cv_splits(cv::WalkForwardCV, n_samples) = cv_split(cv, n_samples)
_cv_splits(cv::PurgedKFoldCV, _) = cv_split(cv)
_cv_splits(cv::CombinatorialPurgedCV, _) = cv_split(cv)

function _accuracy(forest, x_test, y_test)
    return sum(apply_forest(forest, x_test) .== y_test) / length(y_test)
end

function _neg_log_loss(forest, x_test, y_test, classes)
    proba = apply_forest_proba(forest, x_test, classes)
    column_of = Dict(c => i for (i, c) in enumerate(classes))
    total = 0.0
    for n in eachindex(y_test)
        p = clamp(proba[n, column_of[y_test[n]]], 1e-15, 1 - 1e-15)
        total += log(p)
    end
    return total / length(y_test)
end

"""
    cross_val_score(cv, x, y; n_trees=100, n_subfeatures=-1, max_depth=-1,
                    scoring=:accuracy, random_state=0) -> Vector{Float64}

Per-fold score of a random forest under the cross-validator `cv` (any of
`KFoldCV`, `PurgedKFoldCV`, `CombinatorialPurgedCV`, `WalkForwardCV`). `scoring`
is `:accuracy` or `:neg_log_loss`. Folds whose training set is empty are skipped.
Behavioural.
"""
function cross_val_score(
    cv,
    x::AbstractMatrix{<:Real},
    y::AbstractVector;
    n_trees::Integer = 100,
    n_subfeatures::Integer = -1,
    max_depth::Integer = -1,
    scoring::Symbol = :accuracy,
    random_state::Integer = 0,
)
    classes = sort(unique(y))
    scores = Float64[]
    for (f, (train, test)) in enumerate(_cv_splits(cv, size(x, 1)))
        (isempty(train) || isempty(test)) && continue
        forest = build_forest(
            y[train], x[train, :], n_subfeatures, n_trees, 0.7, max_depth, 1, 2, 0.0;
            rng = random_state + f,
        )
        if scoring == :accuracy
            push!(scores, _accuracy(forest, x[test, :], y[test]))
        elseif scoring == :neg_log_loss
            push!(scores, _neg_log_loss(forest, x[test, :], y[test], classes))
        else
            throw(ArgumentError("scoring must be :accuracy or :neg_log_loss"))
        end
    end
    return scores
end
