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

# --------------------------------------------------------------------------- #
# Leakage-aware HPO methodology (Akiba 2019 Optuna + de Prado purged CV / DSR).
# Admitted in Appraisal 20 as methodology/infrastructure, NOT a performance claim:
# principled search reaches the optimum in fewer trials and purged CV removes the
# leakage a naive k-fold inflates, but tuning yields NO out-of-sample edge after
# deflation, so the selected model must be gated by the Deflated Sharpe at the HPO
# trial count. `deflated_sharpe_gate` is the decisive control (deterministic, parity-
# matched in `test/runtests.jl`).
#
# Deliberate divergence: the Optuna TPE/CMA-ES sampler is an OPTIONAL analogue not
# bundled in the Julia port (no de-facto Optuna). `leakage_aware_hpo` wires random
# sampling through the repo's `PurgedKFoldCV` (the leakage-controlled per-trial
# score) — the admitted methodology — and the DSR gate; a Bayesian/evolutionary
# sampler can be substituted where available. Appraisal 20
# (`library_extension/appraisals/20_verdict.md`).
# --------------------------------------------------------------------------- #

using Statistics: std
using ..Backtest: probabilistic_sharpe_ratio, expected_max_sharpe_ratio

"""
    leakage_aware_hpo(x, y, param_grid; event_starts, event_ends, n_trials=50,
        n_splits=5, embargo=0.0, scoring=:accuracy, random_state=0)

Leakage-aware hyper-parameter search: every sampled configuration is scored under
`PurgedKFoldCV` (leakage-controlled, not a leaky shuffled k-fold). Returns a
`NamedTuple` `(best_params, best_score, n_trials, trial_scores, mean_trial_score,
std_trial_score)`.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
use leakage-aware HPO (Optuna wired through PurgedKFold/CPCV, selection gated by
PBO/DSR at the HPO trial count) as the correct, efficient, leakage-safe tuning
methodology — preferred over grid/random for search efficiency and over naive-CV
tuning for leakage-safety. It does not improve out-of-sample performance after
deflation (tuning does not create edge — a documented finding); admitted as
methodology/infrastructure, not a performance claim. Always gate the selected model
with `deflated_sharpe_gate` at `n_trials`; never trust `best_score` directly.

Deliberate divergence: the Optuna TPE/CMA-ES sampler is an optional analogue (not
bundled); this uses random sampling through `PurgedKFoldCV`. Mirrors the
methodology of Python's `leakage_aware_hpo`.
"""
function leakage_aware_hpo(
    x::AbstractMatrix{<:Real},
    y::AbstractVector,
    param_grid::AbstractDict;
    event_starts::AbstractVector,
    event_ends::AbstractVector,
    n_trials::Integer = 50,
    n_splits::Integer = 5,
    embargo::Real = 0.0,
    scoring::Symbol = :accuracy,
    random_state::Integer = 0,
)
    cv = PurgedKFoldCV(n_splits, event_starts, event_ends; embargo = embargo)
    rs = random_search_cv(
        cv, x, y, param_grid;
        n_iter = n_trials, scoring = scoring, random_state = random_state,
    )
    scores = [s for (_, s) in rs.results if isfinite(s)]
    return (
        best_params = rs.best_params,
        best_score = rs.best_score,
        n_trials = n_trials,
        trial_scores = scores,
        mean_trial_score = isempty(scores) ? NaN : mean(scores),
        std_trial_score = length(scores) < 2 ? NaN : std(scores; corrected = false),
    )
end

"""
    deflated_sharpe_gate(out_of_sample_returns, n_trials, trial_sharpe_std; threshold=0.5)

Gate a selected model's out-of-sample Sharpe by the Deflated Sharpe at the HPO trial
count: the benchmark is the expected maximum Sharpe over `n_trials` trials (with
cross-trial Sharpe dispersion `trial_sharpe_std`), and the Deflated Sharpe is the
probability the realized OOS Sharpe exceeds it. A selection passes only if the
Deflated Sharpe exceeds `threshold`. Returns a `NamedTuple` `(observed_sharpe,
benchmark_sharpe, n_trials, deflated_sharpe, passes)`. This is the decisive control:
in the appraisal the principled-HPO gain did not pass it. Mirrors Python's
`deflated_sharpe_gate`.
"""
function deflated_sharpe_gate(
    out_of_sample_returns::AbstractVector{<:Real},
    n_trials::Integer,
    trial_sharpe_std::Real;
    threshold::Real = 0.5,
)
    r = float.(out_of_sample_returns)
    n = length(r)
    m = mean(r)
    m2 = sum((r .- m) .^ 2) / n
    sd = sqrt(m2)                                    # population std (numpy ddof=0)
    observed = sd > 0 ? m / sd : 0.0
    benchmark = expected_max_sharpe_ratio(n_trials, 0.0, trial_sharpe_std)
    skewness = n > 2 ? (sum((r .- m) .^ 3) / n) / m2^1.5 : 0.0
    kurtosis = n > 3 ? (sum((r .- m) .^ 4) / n) / m2^2 : 3.0
    deflated = probabilistic_sharpe_ratio(
        observed, benchmark, n;
        skewness_of_returns = skewness, kurtosis_of_returns = kurtosis,
    )
    return (
        observed_sharpe = observed,
        benchmark_sharpe = benchmark,
        n_trials = n_trials,
        deflated_sharpe = deflated,
        passes = deflated > threshold,
    )
end
