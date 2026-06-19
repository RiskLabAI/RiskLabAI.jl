"""
Cross-validators — native Julia port mirroring the Python
`RiskLabAI.backtest.validation` sub-package (López de Prado, AFML Ch. 7):
standard K-Fold, Purged K-Fold with embargo, Combinatorial Purged
Cross-Validation (CPCV), and Walk-Forward. These produce **train/test index
splits**; the estimator-driven `backtest_predictions` (which needs an ML backend)
is wired with the cross-validation-scoring slice.

Representation note (deliberate divergence): pandas `times` (a Series indexed by
observation-start, valued by observation-end) becomes two parallel vectors
`event_starts` / `event_ends`; all indices are 1-based integer positions (Python
returns 0-based). The purge + embargo index logic matches Python exactly
(verified in `test/runtests.jl`).

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 7.
"""

using Combinatorics: combinations
using Random: AbstractRNG, MersenneTwister, default_rng, shuffle!

"""
    _array_split(n, k) -> Vector{UnitRange{Int}}

Partition `1:n` into `k` contiguous folds, mirroring `numpy.array_split`: the
first `n % k` folds get one extra element.
"""
function _array_split(n::Integer, k::Integer)
    base, rem = divrem(n, k)
    folds = Vector{UnitRange{Int}}(undef, k)
    start = 1
    for i = 1:k
        size = base + (i <= rem ? 1 : 0)
        folds[i] = start:(start+size-1)
        start += size
    end
    return folds
end

# Training positions left after purging (and embargoing) overlap with the test
# set. Mirrors Python's `filtered_training_indices_with_embargo`.
function _purged_train_indices(
    event_starts::AbstractVector,
    event_ends::AbstractVector,
    test_positions::AbstractVector{<:Integer},
    embargo::Real,
    continuous::Bool,
)
    n = length(event_starts)
    isempty(test_positions) && return collect(1:n)
    embargo_length = floor(Int, n * embargo)

    order = sort(collect(test_positions); by = i -> event_starts[i])
    sorted_starts = event_starts[order]
    sorted_ends = event_ends[order]

    T = promote_type(eltype(event_starts), eltype(event_ends))
    block_starts = T[]
    block_ends = T[]
    if continuous
        push!(block_starts, sorted_starts[1])
        push!(block_ends, sorted_ends[end])
    else
        current_start = sorted_starts[1]
        current_end = sorted_ends[1]
        for k = 2:length(sorted_starts)
            if sorted_starts[k] > current_end
                push!(block_starts, current_start)
                push!(block_ends, current_end)
                current_start = sorted_starts[k]
                current_end = sorted_ends[k]
            else
                current_end = max(current_end, sorted_ends[k])
            end
        end
        push!(block_starts, current_start)
        push!(block_ends, current_end)
    end

    if embargo_length != 0
        for b in eachindex(block_ends)
            position = searchsortedfirst(event_starts, block_ends[b]) - 1   # 0-based count
            if position < n
                block_ends[b] = event_starts[min(position + embargo_length, n - 1)+1]
            end
        end
    end

    drop = falses(n)
    for b in eachindex(block_starts)
        low = block_starts[b]
        high = block_ends[b]
        for i = 1:n
            overlap_start = low <= event_starts[i] <= high
            overlap_end = low <= event_ends[i] <= high
            envelops = event_starts[i] <= low && event_ends[i] >= high
            (overlap_start || overlap_end || envelops) && (drop[i] = true)
        end
    end
    return [i for i = 1:n if !drop[i]]
end

# --------------------------------------------------------------------------- #
# K-Fold
# --------------------------------------------------------------------------- #
"""
    KFoldCV(n_splits; shuffle=false, random_seed=nothing)

Standard K-Fold cross-validator over `n_splits` contiguous folds, optionally
shuffled. Mirrors Python's `KFold`.
"""
struct KFoldCV
    n_splits::Int
    shuffle::Bool
    rng::AbstractRNG
end

KFoldCV(n_splits::Integer; shuffle::Bool = false, random_seed = nothing) = KFoldCV(
    n_splits,
    shuffle,
    random_seed === nothing ? default_rng() : MersenneTwister(random_seed),
)

get_n_splits(cv::KFoldCV) = cv.n_splits

"""
    cv_split(cv, n_samples) -> Vector{Tuple{Vector{Int},Vector{Int}}}

Train/test index splits (1-based) for `n_samples` observations.
"""
function cv_split(cv::KFoldCV, n_samples::Integer)
    indices = collect(1:n_samples)
    cv.shuffle && shuffle!(cv.rng, indices)
    splits = Tuple{Vector{Int},Vector{Int}}[]
    for fold in _array_split(n_samples, cv.n_splits)
        test = indices[fold]
        train = setdiff(indices, test)
        push!(splits, (train, test))
    end
    return splits
end

# --------------------------------------------------------------------------- #
# Purged K-Fold (with embargo)
# --------------------------------------------------------------------------- #
"""
    PurgedKFoldCV(n_splits, event_starts, event_ends; embargo=0.0)

Purged K-Fold cross-validator: each contiguous test fold is purged of overlapping
training observations, with an optional embargo (fraction of the dataset removed
after the test set). `event_starts`/`event_ends` give each observation's
information range. Mirrors Python's `PurgedKFold`.
"""
struct PurgedKFoldCV
    n_splits::Int
    event_starts::Vector
    event_ends::Vector
    embargo::Float64
end

PurgedKFoldCV(
    n_splits::Integer,
    event_starts::AbstractVector,
    event_ends::AbstractVector;
    embargo::Real = 0.0,
) = PurgedKFoldCV(n_splits, collect(event_starts), collect(event_ends), Float64(embargo))

get_n_splits(cv::PurgedKFoldCV) = cv.n_splits

function cv_split(cv::PurgedKFoldCV)
    n = length(cv.event_starts)
    splits = Tuple{Vector{Int},Vector{Int}}[]
    for fold in _array_split(n, cv.n_splits)
        test = collect(fold)
        train =
            _purged_train_indices(cv.event_starts, cv.event_ends, test, cv.embargo, true)
        push!(splits, (train, test))
    end
    return splits
end

# --------------------------------------------------------------------------- #
# Combinatorial Purged Cross-Validation (CPCV)
# --------------------------------------------------------------------------- #
"""
    CombinatorialPurgedCV(n_splits, n_test_groups, event_starts, event_ends; embargo=0.0)

Combinatorial Purged Cross-Validation: forms all `C(n_splits, n_test_groups)`
test combinations, each purged + embargoed. `cv_split` yields the combinatorial
splits; `backtest_paths` assembles the `n_test_groups · C / n_splits` backtest
paths. Mirrors Python's `CombinatorialPurged`.
"""
struct CombinatorialPurgedCV
    n_splits::Int
    n_test_groups::Int
    event_starts::Vector
    event_ends::Vector
    embargo::Float64
end

function CombinatorialPurgedCV(
    n_splits::Integer,
    n_test_groups::Integer,
    event_starts::AbstractVector,
    event_ends::AbstractVector;
    embargo::Real = 0.0,
)
    n_test_groups >= n_splits &&
        throw(ArgumentError("n_test_groups must be strictly less than n_splits"))
    return CombinatorialPurgedCV(
        n_splits,
        n_test_groups,
        collect(event_starts),
        collect(event_ends),
        Float64(embargo),
    )
end

get_n_splits(cv::CombinatorialPurgedCV) = binomial(cv.n_splits, cv.n_test_groups)

_combination_test(segments, combination) =
    reduce(vcat, (collect(segments[i]) for i in combination))

function cv_split(cv::CombinatorialPurgedCV)
    n = length(cv.event_starts)
    segments = _array_split(n, cv.n_splits)
    splits = Tuple{Vector{Int},Vector{Int}}[]
    for combination in combinations(1:cv.n_splits, cv.n_test_groups)
        test = _combination_test(segments, combination)
        train = _purged_train_indices(
            cv.event_starts,
            cv.event_ends,
            test,
            cv.embargo,
            false,
        )
        push!(splits, (train, test))
    end
    return splits
end

# Map each backtest path to its (group, combination) coordinates.
# Mirrors Python's `_path_locations`.
function _path_locations(n_splits::Integer, combination_list)
    counters = ones(Int, n_splits)
    locations = Dict{Int,Vector{Tuple{Int,Int}}}()
    for (j, combination) in enumerate(combination_list)
        for group in combination
            path_id = counters[group]
            counters[group] += 1
            push!(get!(locations, path_id, Tuple{Int,Int}[]), (group, j))
        end
    end
    return locations
end

"""
    backtest_paths(cv::CombinatorialPurgedCV) -> Dict{Int,Vector{Tuple{Vector{Int},Vector{Int}}}}

For each backtest path, a list of `(train_indices, test_segment)` pairs (all
1-based). The training set of each segment is purged against its full
combinatorial test set, while the test segment is the single group walked by that
path. Mirrors Python's `_single_backtest_paths`.
"""
function backtest_paths(cv::CombinatorialPurgedCV)
    n = length(cv.event_starts)
    segments = _array_split(n, cv.n_splits)
    combination_list = collect(combinations(1:cv.n_splits, cv.n_test_groups))
    locations = _path_locations(cv.n_splits, combination_list)

    paths = Dict{Int,Vector{Tuple{Vector{Int},Vector{Int}}}}()
    for (path_id, coordinates) in locations
        path_data = Tuple{Vector{Int},Vector{Int}}[]
        for (group, j) in coordinates
            test_combination = _combination_test(segments, combination_list[j])
            train = _purged_train_indices(
                cv.event_starts,
                cv.event_ends,
                test_combination,
                cv.embargo,
                false,
            )
            push!(path_data, (train, collect(segments[group])))
        end
        paths[path_id] = path_data
    end
    return paths
end

# --------------------------------------------------------------------------- #
# Walk-Forward
# --------------------------------------------------------------------------- #
"""
    WalkForwardCV(n_splits; max_train_size=nothing, gap=0)

Walk-Forward cross-validator: the training window grows (capped at
`max_train_size`) while the test fold walks forward, with an optional `gap`
between train and test. Mirrors Python's `WalkForward`.
"""
struct WalkForwardCV
    n_splits::Int
    max_train_size::Union{Nothing,Int}
    gap::Int
end

WalkForwardCV(n_splits::Integer; max_train_size = nothing, gap::Integer = 0) =
    WalkForwardCV(n_splits, max_train_size, gap)

get_n_splits(cv::WalkForwardCV) = cv.n_splits

function cv_split(cv::WalkForwardCV, n_samples::Integer)
    splits = Tuple{Vector{Int},Vector{Int}}[]
    for fold in _array_split(n_samples, cv.n_splits)
        test = collect(fold)
        train_end = first(fold) - 1 - cv.gap   # count of usable training observations
        if train_end <= 0
            train = Int[]
        elseif cv.max_train_size !== nothing && train_end > cv.max_train_size
            train = collect((train_end-cv.max_train_size+1):train_end)
        else
            train = collect(1:train_end)
        end
        push!(splits, (train, test))
    end
    return splits
end
