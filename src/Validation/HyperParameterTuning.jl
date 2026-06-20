"""
Hyper-parameter tuning — native Julia port mirroring the Python
`RiskLabAI.optimization.hyper_parameter_tuning` API (López de Prado, AFML Ch. 9):
grid / randomised search over a random-forest hyper-parameter grid, scored by a
purged (or any) cross-validator via `cross_val_score`.

Deliberate divergence: scikit-learn's `GridSearchCV` / `RandomizedSearchCV`,
`Pipeline` / `SampleWeightedPipeline` / `MyPipeline`, and the `f1` scorer are
replaced by a small functional API over the `DecisionTree.jl` backend; the tuned
estimator is the random forest. Behavioural — validated structurally.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 9.
"""

using Statistics: mean
using Random: MersenneTwister
using DecisionTree: build_forest

# Build a forest from a hyper-parameter dictionary (keys among
# :n_trees, :n_subfeatures, :max_depth), using sensible defaults otherwise.
function _forest_from_params(x, y, params; random_state = 0)
    return build_forest(
        y, x,
        get(params, :n_subfeatures, -1),
        get(params, :n_trees, 100),
        0.7,
        get(params, :max_depth, -1),
        1, 2, 0.0;
        rng = random_state,
    )
end

function _score_params(cv, x, y, params; scoring, random_state)
    scores = cross_val_score(
        cv, x, y;
        n_trees = get(params, :n_trees, 100),
        n_subfeatures = get(params, :n_subfeatures, -1),
        max_depth = get(params, :max_depth, -1),
        scoring = scoring,
        random_state = random_state,
    )
    return isempty(scores) ? -Inf : mean(scores)
end

"""
    grid_search_cv(cv, x, y, param_grid; scoring=:accuracy, random_state=0)
        -> (; best_params, best_score, model, results)

Exhaustive grid search over `param_grid` (a `Dict{Symbol,<:AbstractVector}` with
keys among `:n_trees`, `:n_subfeatures`, `:max_depth`), each configuration scored
by `cross_val_score` under cross-validator `cv`. Returns the best parameters, its
mean CV score, a forest refit on all data with those parameters, and every
`(params, score)` pair. Behavioural. Mirrors Python's `clf_hyper_fit`
(`rnd_search_iter == 0`).
"""
function grid_search_cv(
    cv,
    x::AbstractMatrix{<:Real},
    y::AbstractVector,
    param_grid::AbstractDict;
    scoring::Symbol = :accuracy,
    random_state::Integer = 0,
)
    keys_ordered = collect(keys(param_grid))
    value_lists = [collect(param_grid[k]) for k in keys_ordered]

    best_params = Dict{Symbol,Any}()
    best_score = -Inf
    results = Tuple{Dict{Symbol,Any},Float64}[]
    for combination in Iterators.product(value_lists...)
        params = Dict{Symbol,Any}(keys_ordered[i] => combination[i] for i in eachindex(keys_ordered))
        score = _score_params(cv, x, y, params; scoring = scoring, random_state = random_state)
        push!(results, (params, score))
        if score > best_score
            best_score = score
            best_params = params
        end
    end

    model = _forest_from_params(x, y, best_params; random_state = random_state)
    return (best_params = best_params, best_score = best_score, model = model, results = results)
end

"""
    random_search_cv(cv, x, y, param_grid; n_iter=10, scoring=:accuracy, random_state=0)
        -> (; best_params, best_score, model, results)

Randomised search: sample `n_iter` configurations from `param_grid` (one random
value per key) and score each by `cross_val_score`. Behavioural. Mirrors Python's
`clf_hyper_fit` with `rnd_search_iter > 0`.
"""
function random_search_cv(
    cv,
    x::AbstractMatrix{<:Real},
    y::AbstractVector,
    param_grid::AbstractDict;
    n_iter::Integer = 10,
    scoring::Symbol = :accuracy,
    random_state::Integer = 0,
)
    rng = MersenneTwister(random_state)
    keys_ordered = collect(keys(param_grid))

    best_params = Dict{Symbol,Any}()
    best_score = -Inf
    results = Tuple{Dict{Symbol,Any},Float64}[]
    for _ = 1:n_iter
        params = Dict{Symbol,Any}(k => rand(rng, collect(param_grid[k])) for k in keys_ordered)
        score = _score_params(cv, x, y, params; scoring = scoring, random_state = random_state)
        push!(results, (params, score))
        if score > best_score
            best_score = score
            best_params = params
        end
    end

    model = _forest_from_params(x, y, best_params; random_state = random_state)
    return (best_params = best_params, best_score = best_score, model = model, results = results)
end
