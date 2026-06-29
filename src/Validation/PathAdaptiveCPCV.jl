"""
Path-level Adaptive Combinatorial Purged Cross-Validation PBO (Arian–Norouzi–Seco
2024). de Prado's CPCV / CSCV weights every combinatorial backtest path equally;
when the train/test regime shifts, that equal weighting biases selection toward the
wrong regime. Adaptive CPCV weights each path by how much its TEST block resembles
the FORWARD regime, read from an observable, point-in-time volatility signal (the
cross-configuration dispersion). Down-weighting paths that test on a
non-representative regime lowers selection error when the regime is detectable, and
the weights collapse to uniform in a stationary regime, so the estimate converges to
plain CPCV with no over-adaptation.

This is the path-level mechanism; it does not touch the existing split-boundary
adaptive cross-validator. The per-path overfit indicator comes from
`Backtest.performance_evaluation`, so with uniform weights it reproduces
`Backtest.probability_of_backtest_overfitting` exactly. The port is fully
deterministic (no RNG); numeric parity asserted in `test/runtests.jl`. Admitted in
Appraisal 09b (`library_extension/appraisals/09_verdict.md`; in-house method, COI,
real-data regime-shift confirmation a tracked obligation).

Reference: Arian, H., Norouzi, M. L. & Seco, L. (2024). Bagged and Adaptive
Combinatorial Purged Cross-Validation. Bailey, Borwein, López de Prado & Zhu
(2017), Journal of Computational Finance 20(4).
"""

using ..Backtest: performance_evaluation, sharpe_ratio
using Statistics: mean, median, quantile
using Combinatorics: combinations

# numpy.array_split row ranges: the first (T mod S) partitions take one extra row.
function _array_split_ranges(t_len::Integer, n_partitions::Integer)
    base, remainder = divrem(t_len, n_partitions)
    ranges = UnitRange{Int}[]
    start = 1
    for p = 1:n_partitions
        size_p = base + (p <= remainder ? 1 : 0)
        push!(ranges, start:(start+size_p-1))
        start += size_p
    end
    return ranges
end

# Most frequent non-negative integer label (ties → smallest label), like np.bincount(.).argmax().
function _mode_nonneg(v::AbstractVector{<:Integer})
    m = maximum(v)
    counts = zeros(Int, m + 1)
    for x in v
        counts[x+1] += 1
    end
    return argmax(counts) - 1
end

"""
    estimate_volatility_regimes(performances; n_regimes=2, window=nothing) -> Vector{Int}

Observable regime label per period from the cross-configuration volatility: the
per-period dispersion across configurations (a market-stress proxy), smoothed by a
trailing (causal) moving average, then split at the observed median (`n_regimes ≤ 2`)
or the interior quantiles (`n_regimes > 2`). Uses only returns up to each period, so
it carries no look-ahead. `window` defaults to `max(T ÷ 20, 3)`. Mirrors Python's
`estimate_volatility_regimes`.
"""
function estimate_volatility_regimes(
    performances::AbstractMatrix{<:Real};
    n_regimes::Integer = 2,
    window::Union{Integer,Nothing} = nothing,
)
    P = float.(performances)
    t_len = size(P, 1)
    dispersion = [std_population(@view P[i, :]) for i = 1:t_len]
    w = window === nothing ? max(t_len ÷ 20, 3) : window
    w = max(Int(w), 1)
    cumulative = zeros(t_len + 1)
    for k = 1:t_len
        cumulative[k+1] = cumulative[k] + dispersion[k]
    end
    smooth = [(cumulative[j+1] - cumulative[max(j - w, 0)+1]) / min(j, w) for j = 1:t_len]
    if n_regimes <= 2
        med = median(smooth)
        return Int.(smooth .> med)
    end
    levels = range(0.0, 1.0; length = n_regimes + 1)[2:(end-1)]
    edges = [quantile(smooth, q) for q in levels]
    return [sum(edges .<= x) for x in smooth]      # np.digitize, right=False
end

# Population (ddof=0) standard deviation, matching numpy `.std(axis=1)`.
function std_population(x)
    m = mean(x)
    return sqrt(sum((xi - m)^2 for xi in x) / length(x))
end

# Per CSCV path: (is_overfit, test_config_metrics, test_partition_ids).
function _cscv_path_stats(
    P::AbstractMatrix{<:Real},
    ranges::Vector{UnitRange{Int}},
    metric,
    risk_free_return::Real,
)
    n_strategies = size(P, 2)
    all_ids = collect(1:length(ranges))
    stats = Tuple{Float64,Vector{Float64},Vector{Int}}[]
    for train_ids in combinations(all_ids, length(ranges) ÷ 2)
        test_ids = [i for i in all_ids if !(i in train_ids)]
        train = reduce(vcat, (P[ranges[i], :] for i in train_ids))
        test = reduce(vcat, (P[ranges[i], :] for i in test_ids))
        is_overfit, _ =
            performance_evaluation(train, test, n_strategies, metric, risk_free_return)
        test_metrics = [metric(view(test, :, j), risk_free_return) for j = 1:n_strategies]
        push!(stats, (is_overfit ? 1.0 : 0.0, test_metrics, test_ids))
    end
    return stats
end

"""
    adaptive_probability_of_backtest_overfitting(performances; n_partitions=16,
        target_fraction=0.25, risk_free_return=0.0, metric=nothing, n_regimes=2)

Adaptive (path-level) Probability of Backtest Overfitting and regime-aware
configuration selection. Each CSCV path is weighted by how much its test block
resembles the forward regime (the regime of the most recent `target_fraction` of
the sample). Returns `(regime_weighted_pbo, selected_config)` with `selected_config`
the 1-based index of the configuration with the highest regime-weighted OOS metric.
In a stationary regime the weights are uniform, so the PBO converges to plain CPCV.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer Adaptive CPCV over plain CPCV for model selection when the train/test regime
may shift, the shift is identifiable at decision time from observable volatility,
and there is adequate data for selection (it does not help where selection is
noise-dominated by too little data); it converges to plain CPCV in a stationary
regime with no over-adaptation.

Mirrors Python's `adaptive_probability_of_backtest_overfitting`.
"""
function adaptive_probability_of_backtest_overfitting(
    performances::AbstractMatrix{<:Real};
    n_partitions::Integer = 16,
    target_fraction::Real = 0.25,
    risk_free_return::Real = 0.0,
    metric = nothing,
    n_regimes::Integer = 2,
)
    P = float.(performances)
    isodd(n_partitions) && throw(ArgumentError("Number of partitions must be even."))
    t_len, n_configs = size(P)
    t_len < n_partitions &&
        throw(ArgumentError("Too few observations ($t_len) for $n_partitions partitions."))
    (0.0 < target_fraction <= 1.0) ||
        throw(ArgumentError("target_fraction must be in (0, 1]."))
    chosen_metric =
        metric === nothing ? ((r, rf) -> sharpe_ratio(r; risk_free_rate = rf)) : metric

    regimes = estimate_volatility_regimes(P; n_regimes = n_regimes)
    forward = regimes[(Int(floor((1.0 - target_fraction) * t_len))+1):end]
    forward_label = isempty(forward) ? 0 : _mode_nonneg(forward)

    ranges = _array_split_ranges(t_len, n_partitions)
    regime_partitions = [regimes[r] for r in ranges]
    stats = _cscv_path_stats(P, ranges, chosen_metric, risk_free_return)

    overfit = [s[1] for s in stats]
    test_metrics = [s[2] for s in stats]
    n_paths = length(stats)
    weights = Float64[]
    for (_, _, test_ids) in stats
        test_regime = reduce(vcat, (regime_partitions[i] for i in test_ids))
        push!(weights, mean(test_regime .== forward_label) + 1e-3)
    end

    wsum = sum(weights)
    pbo = sum(weights .* overfit) / wsum
    col_means =
        [sum(weights[k] * test_metrics[k][j] for k = 1:n_paths) / wsum for j = 1:n_configs]
    selected = argmax(col_means)
    return (pbo, selected)
end
