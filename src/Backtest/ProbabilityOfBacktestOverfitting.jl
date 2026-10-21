"""
Probability of backtest overfitting — native Julia port mirroring the Python
`RiskLabAI.backtest.probability_of_backtest_overfitting` API (López de Prado,
AFML Ch. 11–12): the combinatorially-symmetric cross-validation (CSCV) estimate
of PBO.

The metric defaults to the population-std Sharpe ratio (`Backtest.sharpe_ratio`);
shared checks cover distinct scores and supplied reference rank orders. Native
stable sorting can assign different ordinal ranks to tied scores than NumPy's
default sort, including different overfit decisions.
The combinatorial train/test splits use `Combinatorics.combinations`, mirroring
Python's `itertools.combinations`, and the row partitioning mirrors
`numpy.array_split` (the first `T mod S` partitions take one extra row).

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 11.
"""

function _first_metric_maximum(values)
    isempty(values) && throw(ArgumentError("At least one strategy is required."))
    selected = firstindex(values)
    for index in (selected+1):lastindex(values)
        if !isnan(values[selected]) && (isnan(values[index]) || values[index] > values[selected])
            selected = index
        end
    end
    return selected
end

"""
    performance_evaluation(train_partition, test_partition, n_strategies, metric, risk_free_return)
        -> (is_overfit, logit_value)

Pick the best strategy in-sample (by `metric`), find its out-of-sample rank, and
return whether it landed in the bottom half (`logit ≤ 0`) together with the logit
of its relative rank. Optional `rank_order(scores)` supplies the one-based
ascending strategy permutation, including exact tie order. Without a provider,
native stable sorting is retained and is not equivalent to NumPy tie ordering.
"""
function performance_evaluation(
    train_partition::AbstractMatrix{<:Real},
    test_partition::AbstractMatrix{<:Real},
    n_strategies::Integer,
    metric,
    risk_free_return::Real;
    rank_order = nothing,
)
    evaluate_train =
        [metric(view(train_partition, :, i), risk_free_return) for i = 1:n_strategies]
    best_strategy_idx = _first_metric_maximum(evaluate_train)
    evaluate_test =
        [metric(view(test_partition, :, i), risk_free_return) for i = 1:n_strategies]
    # 1-based ordinal ranks (argsort∘argsort); equals Python's rank + 1.
    order = rank_order === nothing ? sortperm(evaluate_test) : Vector{Int}(rank_order(copy(evaluate_test)))
    sort(order) == collect(1:n_strategies) || throw(ArgumentError("Rank order must be a permutation of all strategies."))
    ordered = evaluate_test[order]
    any((isnan(ordered[index]) && !isnan(ordered[index+1])) || (!isnan(ordered[index+1]) && ordered[index] > ordered[index+1]) for index in 1:(length(ordered)-1)) && throw(ArgumentError("Rank order must be nondecreasing with NaNs last."))
    ranks = invperm(order)
    rank_of_best = ranks[best_strategy_idx]
    w_bar = rank_of_best / (n_strategies + 1)
    logit_value = log(w_bar / (1 - w_bar))
    return (logit_value <= 0.0, logit_value)
end

"""
    probability_of_backtest_overfitting(performances; n_partitions=16,
        risk_free_return=0.0, metric=nothing) -> (pbo, logit_values)

Probability of backtest overfitting over all `C(S, S/2)` combinatorial
train/test splits of the `T×N` `performances` matrix (`S = n_partitions`, must be
even). `pbo` is the share of splits whose best in-sample strategy underperforms
out-of-sample; `logit_values` are the per-split logits. `metric` defaults to the
population-std Sharpe ratio. Mirrors Python's
`probability_of_backtest_overfitting`. Optional `rank_order` is forwarded to
each path's performance evaluation for exact tied-score replay.
"""
function probability_of_backtest_overfitting(
    performances::AbstractMatrix{<:Real};
    n_partitions::Integer = 16,
    risk_free_return::Real = 0.0,
    metric = nothing,
    rank_order = nothing,
)
    isodd(n_partitions) && throw(ArgumentError("Number of partitions must be even."))
    chosen_metric =
        metric === nothing ? (r, rf) -> sharpe_ratio(r; risk_free_rate = rf) : metric
    n_observations, n_strategies = size(performances)

    # numpy.array_split: first (T mod S) partitions get one extra row.
    base, remainder = divrem(n_observations, n_partitions)
    ranges = UnitRange{Int}[]
    start = 1
    for p = 1:n_partitions
        size_p = base + (p <= remainder ? 1 : 0)
        push!(ranges, start:(start+size_p-1))
        start += size_p
    end

    indices = collect(1:n_partitions)
    logit_values = Float64[]
    overfits = Bool[]
    for train_indices in combinations(indices, n_partitions ÷ 2)
        test_indices = [i for i in indices if !(i in train_indices)]
        train = reduce(vcat, (performances[ranges[i], :] for i in train_indices))
        test = reduce(vcat, (performances[ranges[i], :] for i in test_indices))
        is_overfit, logit_value = performance_evaluation(
            train,
            test,
            n_strategies,
            chosen_metric,
            risk_free_return;
            rank_order = rank_order,
        )
        push!(overfits, is_overfit)
        push!(logit_values, logit_value)
    end

    return (mean(overfits), logit_values)
end
