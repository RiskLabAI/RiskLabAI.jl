using Statistics: quantile

function _validate_event_ranges(event_starts, event_ends; embargo = 0.0)
    length(event_starts) == length(event_ends) || throw(DimensionMismatch("event starts and ends must align"))
    issorted(event_starts) && allunique(event_starts) || throw(ArgumentError("event starts must be strictly increasing"))
    all(start <= stop for (start, stop) in zip(event_starts, event_ends)) || throw(ArgumentError("event ends must not precede their starts"))
    isfinite(embargo) && 0 <= embargo <= 1 || throw(ArgumentError("embargo must lie in [0, 1]"))
    return nothing
end

"""
    filtered_training_indices_with_embargo(event_starts, event_ends, test_positions;
                                          embargo=0.0, continuous=false)

Return one-based training positions after inclusive overlap removal and the
released observation-count embargo. Events must have increasing distinct starts
and aligned ends. For `continuous=true`, the interval ends at the last test
event's end, preserving the public continuous-range convention.
"""
function filtered_training_indices_with_embargo(event_starts::AbstractVector, event_ends::AbstractVector, test_positions::AbstractVector{<:Integer}; embargo::Real = 0.0, continuous::Bool = false)
    _validate_event_ranges(event_starts, event_ends; embargo)
    all(position -> 1 <= position <= length(event_starts), test_positions) || throw(ArgumentError("test positions must index the event vectors"))
    return _purged_train_indices(event_starts, event_ends, test_positions, embargo, continuous)
end

"""
    AdaptiveCombinatorialPurgedCV(n_splits, n_test_groups, event_starts, event_ends;
        external_feature, embargo=0.0, n_subsplits=3, lower_quantile=0.25,
        upper_quantile=0.75, subtract_border_adjustments=true)

Combinatorial purged validation with the released supplied-feature boundary
adjustments. Quantiles use the full supplied feature vector and linear
interpolation. A value below the lower quantile gives adjustment -1; a value
above the upper quantile gives +1. The adjustment is subtracted by default.
Feature values are copied and must align with events. At least
`n_splits*n_subsplits` observations are required. Very small subsplit counts can
produce empty or overlapping segments under the public boundary rule; these
segments are retained without inventing a different partitioning rule.
"""
struct AdaptiveCombinatorialPurgedCV
    n_splits::Int
    n_test_groups::Int
    event_starts::Vector
    event_ends::Vector
    embargo::Float64
    external_feature::Vector{Float64}
    n_subsplits::Int
    lower_quantile::Float64
    upper_quantile::Float64
    subtract_border_adjustments::Bool
end

function AdaptiveCombinatorialPurgedCV(n_splits::Integer, n_test_groups::Integer, event_starts::AbstractVector, event_ends::AbstractVector; external_feature::AbstractVector, embargo::Real = 0.0, n_subsplits::Integer = 3, lower_quantile::Real = 0.25, upper_quantile::Real = 0.75, subtract_border_adjustments::Bool = true)
    _validate_event_ranges(event_starts, event_ends; embargo)
    1 <= n_test_groups < n_splits || throw(ArgumentError("test groups must be positive and less than all groups"))
    n_subsplits >= 1 || throw(ArgumentError("n_subsplits must be positive"))
    length(event_starts) >= n_splits * n_subsplits || throw(ArgumentError("each fine subsplit requires an observation"))
    length(external_feature) == length(event_starts) || throw(DimensionMismatch("external feature must align with events"))
    all(isfinite, external_feature) || throw(ArgumentError("external feature must be finite"))
    0 <= lower_quantile <= upper_quantile <= 1 || throw(ArgumentError("quantiles must be ordered within [0, 1]"))
    return AdaptiveCombinatorialPurgedCV(Int(n_splits), Int(n_test_groups), collect(event_starts), collect(event_ends), Float64(embargo), Float64.(external_feature), Int(n_subsplits), Float64(lower_quantile), Float64(upper_quantile), subtract_border_adjustments)
end

function _validation_segments(cv::AdaptiveCombinatorialPurgedCV)
    count = length(cv.event_starts)
    fine = _array_split(count, cv.n_splits * cv.n_subsplits)
    starts = first.(fine)
    lower, upper = quantile(cv.external_feature, [cv.lower_quantile, cv.upper_quantile])
    boundaries = Int[]
    for border in cv.n_subsplits:cv.n_subsplits:(length(fine)-1)
        value = cv.external_feature[starts[border+1]]
        adjustment = value < lower ? -1 : value > upper ? 1 : 0
        shifted = border + (cv.subtract_border_adjustments ? -adjustment : adjustment)
        push!(boundaries, starts[clamp(shifted, 1, length(starts)-1)+1])
    end
    edges = [1; boundaries; count+1]
    return [collect(edges[index]:(edges[index+1]-1)) for index in 1:cv.n_splits]
end

_validation_segments(cv::CombinatorialPurgedCV) = _array_split(length(cv.event_starts), cv.n_splits)
get_n_splits(cv::AdaptiveCombinatorialPurgedCV) = binomial(cv.n_splits, cv.n_test_groups)
cv_split(cv::AdaptiveCombinatorialPurgedCV) = _combinatorial_cv_split(cv, _validation_segments(cv))
backtest_paths(cv::AdaptiveCombinatorialPurgedCV) = _combinatorial_backtest_paths(cv, _validation_segments(cv))
backtest_paths(cv::PurgedKFoldCV) = Dict(1 => cv_split(cv))
backtest_paths(cv::KFoldCV, n_samples::Integer; permutation = nothing) = Dict(1 => cv_split(cv, n_samples; permutation))
backtest_paths(cv::WalkForwardCV, n_samples::Integer) = Dict(1 => cv_split(cv, n_samples))
backtest_paths(validators::AbstractDict) = Dict(key => backtest_paths(cv) for (key, cv) in validators)

function backtest_paths(validators::AbstractDict, data::AbstractDict)
    Set(keys(validators)) == Set(keys(data)) || throw(ArgumentError("validator and data keys must match"))
    return Dict(key => cv isa Union{KFoldCV,WalkForwardCV} ? backtest_paths(cv, size(data[key], 1)) : backtest_paths(cv) for (key, cv) in validators)
end
