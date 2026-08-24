const _STAGE_NAMES = (
    "Variable Selection",
    "Causal Discovery",
    "Causal Adjustment Set",
    "Causal Explanatory and Predictive Power",
    "Causal Portfolio Construction",
    "Backtest",
    "Multiple Testing Adjustments",
)

const _VARIABLE_SELECTION_METHODS = Set((
    "MUTUAL_INFORMATION",
    "SHAPLEY_VALUES",
    "MEAN_DECREASE_IMPURITY",
    "PERMUTATION_FEATURE_IMPORTANCE",
))
const _RESEARCH_PURPOSES = Set(("CAUSAL_ATTRIBUTION", "RISK_PREMIA_HARVESTING"))
const _DISCOVERY_METHODS = Set((
    "PC",
    "LINGAM",
    "ECONOMIC_REASONING",
    "EX_ANTE_VIEWS",
    "OBSERVED_OUTCOMES",
    "PEER_REVIEWED_ASSUMPTIONS",
    "DOMAIN_EXPERTISE",
))
const _ADJUSTMENT_METHODS =
    Set(("BACKDOOR_ADJUSTMENT", "FRONT_DOOR_ADJUSTMENT", "INSTRUMENTAL_VARIABLES"))
const _VALIDATION_METHODS = Set((
    "PURGED_CROSS_VALIDATION",
    "WALK_FORWARD",
    "RESAMPLING",
    "COMBINATORIAL_PURGED_CROSS_VALIDATION",
))
const _PURGED_VALIDATION_METHODS =
    Set(("PURGED_CROSS_VALIDATION", "COMBINATORIAL_PURGED_CROSS_VALIDATION"))
const _METRICS_BY_TASK = Dict(
    "PROBABILITY" => Set(("LOG_LOSS", "BRIER_SCORE")),
    "RANKING" => Set((
        "ROC_CURVE",
        "PRECISION_RECALL_CURVE",
        "MEAN_RECIPROCAL_RANK",
        "CLASSIFICATION_ACCURACY",
    )),
    "RETURN_SIZE" => Set(("MEAN_SQUARED_ERROR", "R_SQUARED", "SPEARMAN_CORRELATION")),
)
const _PORTFOLIO_METHODS = Set((
    "POSITION_SIZING",
    "EXPOSURE_CONTROL",
    "ECONOMIC_RATIONALE",
    "FRAGILITY_STRESS_TEST",
    "TRANSACTION_COST_OPTIMIZATION",
    "TRANSFER_COEFFICIENT",
))
const _BACKTEST_METHODS = Set((
    "WALK_FORWARD",
    "RESAMPLING",
    "COMBINATORIAL_PURGED_CROSS_VALIDATION",
    "MONTE_CARLO",
))
const _MULTIPLE_TESTING_METHODS =
    Set(("HOLM", "HOCHBERG", "BENJAMINI_HOCHBERG", "DEFLATED_SHARPE_RATIO"))
const _P_VALUE_METHODS = Set(("HOLM", "HOCHBERG", "BENJAMINI_HOCHBERG"))
const _REFIT_COMPONENTS =
    Set(("VARIABLE_SELECTION", "CAUSAL_ESTIMATION", "PORTFOLIO_CONSTRUCTION"))

Base.@kwdef struct EventHorizon
    observation_id::String
    start::Real
    stop::Real
end

Base.@kwdef struct FoldEvidence
    fold_id::String
    path_id::String
    train_ids::Tuple{Vararg{String}}
    test_ids::Tuple{Vararg{String}}
    fit_ids::Tuple{Vararg{String}}
    prediction_ids::Tuple{Vararg{String}}
    refit_id::String
    refit_components::Tuple{Vararg{String}}
end

Base.@kwdef struct ValidationEvidence
    method_labels::Tuple{Vararg{String}}
    event_horizons::Tuple{Vararg{EventHorizon}}
    folds::Tuple{Vararg{FoldEvidence}}
    embargo::Real = 0.0
end

Base.@kwdef struct VariableSelectionStage
    purpose::String
    selected_variables::Tuple{Vararg{String}}
    method_labels::Tuple{Vararg{String}}
    overlapping_returns::Bool
    strong_time_dependence::Bool
    validation::Union{Nothing,ValidationEvidence}
end

Base.@kwdef struct CausalDiscoveryStage
    method_labels::Tuple{Vararg{String}}
    graph_id::String
    graph_nodes::Tuple{Vararg{String}}
    directed_edges::Tuple{Vararg{Tuple{String,String}}}
    graph_kind::String
    ambiguous_edges::Tuple{Vararg{Tuple{String,String}}}
    assumptions::Tuple{Vararg{String}}
end

Base.@kwdef struct CausalAdjustmentSetStage
    treatment::String
    outcome::String
    method_label::String
    identified::Bool
    admissible_adjustment_sets::Tuple
    selected_adjustment_set::Tuple{Vararg{String}}
    confounders::Tuple{Vararg{String}}
    descendants::Tuple{Vararg{String}}
    mediators::Tuple{Vararg{String}}
    colliders::Tuple{Vararg{String}}
    instruments::Tuple{Vararg{String}}
    control_justifications::Tuple
    open_backdoor_paths::Tuple{Vararg{String}}
    frontdoor_criteria_satisfied::Union{Nothing,Bool} = nothing
    instrument_relevance_satisfied::Union{Nothing,Bool} = nothing
    instrument_exclusion_satisfied::Union{Nothing,Bool} = nothing
    instrument_exogeneity_satisfied::Union{Nothing,Bool} = nothing
end

Base.@kwdef struct CausalExplanatoryAndPredictivePowerStage
    task_types::Tuple{Vararg{String}}
    estimator_label::String
    explanatory_metric_labels::Tuple{Vararg{String}}
    predictive_metric_labels::Tuple{Vararg{String}}
    explanatory_evidence_id::Union{Nothing,String}
    predictive_evidence_id::Union{Nothing,String}
    naive_benchmark_id::String
    validation::ValidationEvidence
    multiclass_encoding::Union{Nothing,String} = nothing
    averaging_method::Union{Nothing,String} = nothing
end

Base.@kwdef struct CausalPortfolioConstructionStage
    method_labels::Tuple{Vararg{String}}
    causal_exposures::Tuple{Vararg{String}}
    controlled_unintended_exposures::Tuple{Vararg{String}}
    cost_model_id::String
    constraint_set_id::String
    economic_rationale::String
    fragility_scenarios::Tuple{Vararg{String}}
    transfer_coefficient::Real
end

Base.@kwdef struct BacktestStage
    method_labels::Tuple{Vararg{String}}
    trial_family_id::String
    declared_trial_ids::Tuple{Vararg{String}}
    validation::Union{Nothing,ValidationEvidence} = nothing
    monte_carlo_dgp::Union{Nothing,String} = nothing
end

Base.@kwdef struct MultipleTestingAdjustmentsStage
    method_labels::Tuple{Vararg{String}}
    trial_family_id::String
    declared_trial_ids::Tuple{Vararg{String}}
    family_partitions::Tuple
    alpha::Real
    backtests_independent::Bool
    p_value_estimator::Union{Nothing,String} = nothing
    time_dependence_model::Union{Nothing,String} = nothing
    p_value_inputs_assume_independence::Union{Nothing,Bool} = nothing
    sharpe_variance::Union{Nothing,Real} = nothing
    effective_trials::Union{Nothing,Real} = nothing
    sample_length::Union{Nothing,Integer} = nothing
    skewness::Union{Nothing,Real} = nothing
    kurtosis::Union{Nothing,Real} = nothing
    selection_bias_evidence_id::Union{Nothing,String} = nothing
end

struct CausalFactorProtocolReport
    trial_family_id::String
    declared_trial_ids::Tuple{Vararg{String}}
    variable_selection::VariableSelectionStage
    causal_discovery::CausalDiscoveryStage
    causal_adjustment_set::CausalAdjustmentSetStage
    causal_explanatory_and_predictive_power::CausalExplanatoryAndPredictivePowerStage
    causal_portfolio_construction::CausalPortfolioConstructionStage
    backtest::BacktestStage
    multiple_testing_adjustments::MultipleTestingAdjustmentsStage
    source_oracle::String
    source_locator::String

    function CausalFactorProtocolReport(
        trial_family_id::String,
        declared_trial_ids::Tuple{Vararg{String}},
        variable_selection::VariableSelectionStage,
        causal_discovery::CausalDiscoveryStage,
        causal_adjustment_set::CausalAdjustmentSetStage,
        causal_explanatory_and_predictive_power::CausalExplanatoryAndPredictivePowerStage,
        causal_portfolio_construction::CausalPortfolioConstructionStage,
        backtest::BacktestStage,
        multiple_testing_adjustments::MultipleTestingAdjustmentsStage,
        ::Val{:validated},
    )
        return new(
            trial_family_id,
            declared_trial_ids,
            variable_selection,
            causal_discovery,
            causal_adjustment_set,
            causal_explanatory_and_predictive_power,
            causal_portfolio_construction,
            backtest,
            multiple_testing_adjustments,
            "PROTOCOL-01",
            "PDF pages 18-22 (journal pages 27-31), Exhibit 10",
        )
    end
end

function CausalFactorProtocolReport(;
    trial_family_id,
    declared_trial_ids,
    variable_selection,
    causal_discovery,
    causal_adjustment_set,
    causal_explanatory_and_predictive_power,
    causal_portfolio_construction,
    backtest,
    multiple_testing_adjustments,
)
    return CausalFactorProtocolReport(
        String(trial_family_id),
        Tuple(declared_trial_ids),
        variable_selection,
        causal_discovery,
        causal_adjustment_set,
        causal_explanatory_and_predictive_power,
        causal_portfolio_construction,
        backtest,
        multiple_testing_adjustments,
        Val(:validated),
    )
end

function Base.getproperty(report::CausalFactorProtocolReport, name::Symbol)
    if name === :stage_names
        return _STAGE_NAMES
    elseif name === :stages
        return (
            (_STAGE_NAMES[1], getfield(report, :variable_selection)),
            (_STAGE_NAMES[2], getfield(report, :causal_discovery)),
            (_STAGE_NAMES[3], getfield(report, :causal_adjustment_set)),
            (_STAGE_NAMES[4], getfield(report, :causal_explanatory_and_predictive_power)),
            (_STAGE_NAMES[5], getfield(report, :causal_portfolio_construction)),
            (_STAGE_NAMES[6], getfield(report, :backtest)),
            (_STAGE_NAMES[7], getfield(report, :multiple_testing_adjustments)),
        )
    end
    return getfield(report, name)
end

function Base.propertynames(::CausalFactorProtocolReport, private::Bool = false)
    fields = fieldnames(CausalFactorProtocolReport)
    return private ? (fields..., :stages, :stage_names) : (fields..., :stages, :stage_names)
end

function _add_protocol_error!(errors::Vector{String}, message::String)
    message in errors || push!(errors, message)
    return nothing
end

_nonblank(value) = value isa AbstractString && !isempty(strip(value))

function _finite_protocol_real(value)
    value isa Bool && return false
    value isa Real || return false
    result = try
        Float64(value)
    catch
        return false
    end
    return isfinite(result)
end

function _validate_string_tuple(
    value,
    stage::String,
    subject::String,
    errors::Vector{String};
    allow_empty::Bool = false,
)
    if !(value isa Tuple)
        _add_protocol_error!(errors, "$stage: $subject must be a tuple.")
        return false
    end
    if !allow_empty && isempty(value)
        _add_protocol_error!(errors, "$stage: $subject must not be empty.")
        return false
    end
    if !all(_nonblank, value)
        _add_protocol_error!(errors, "$stage: $subject contains an invalid identifier.")
        return false
    end
    if length(unique(value)) != length(value)
        _add_protocol_error!(errors, "$stage: $subject contains duplicate identifiers.")
        return false
    end
    return true
end

function _validate_labels(
    value,
    allowed,
    stage::String,
    subject::String,
    errors::Vector{String};
    allow_empty::Bool = false,
)
    valid = _validate_string_tuple(value, stage, subject, errors; allow_empty = allow_empty)
    if valid && !issubset(Set(value), allowed)
        _add_protocol_error!(errors, "$stage: $subject contains a non-source label.")
        return false
    end
    return valid
end

function _validate_event_horizons(evidence, stage, errors)
    isempty(evidence.event_horizons) && begin
        _add_protocol_error!(errors, "$stage: event horizons must be nonempty.")
        return Dict{String,Tuple{Float64,Float64}}()
    end
    result = Dict{String,Tuple{Float64,Float64}}()
    for horizon in evidence.event_horizons
        if !_nonblank(horizon.observation_id)
            _add_protocol_error!(
                errors,
                "$stage: an event horizon has an invalid identifier.",
            )
            continue
        end
        if haskey(result, horizon.observation_id)
            _add_protocol_error!(
                errors,
                "$stage: event-horizon identifiers must be unique.",
            )
            continue
        end
        if !_finite_protocol_real(horizon.start) || !_finite_protocol_real(horizon.stop)
            _add_protocol_error!(
                errors,
                "$stage: event-horizon bounds must be finite numbers.",
            )
            continue
        end
        start = Float64(horizon.start)
        stop = Float64(horizon.stop)
        if !(start < stop)
            _add_protocol_error!(
                errors,
                "$stage: every event horizon must have positive length.",
            )
            continue
        end
        result[horizon.observation_id] = (start, stop)
    end
    return result
end

_intervals_overlap(left, right) = left[1] < right[2] && right[1] < left[2]

function _has_any_overlap(horizons)
    intervals = collect(values(horizons))
    for left_index in eachindex(intervals)
        for right_index = (left_index+1):length(intervals)
            _intervals_overlap(intervals[left_index], intervals[right_index]) && return true
        end
    end
    return false
end

function _validate_fold_evidence(
    evidence::ValidationEvidence,
    stage::String,
    errors::Vector{String};
    required_refit_components,
)
    _validate_labels(
        evidence.method_labels,
        _VALIDATION_METHODS,
        stage,
        "validation method labels",
        errors,
    )
    horizon_map = _validate_event_horizons(evidence, stage, errors)
    if !_finite_protocol_real(evidence.embargo) || Float64(evidence.embargo) < 0.0
        _add_protocol_error!(errors, "$stage: embargo must be a finite nonnegative number.")
        embargo = 0.0
    else
        embargo = Float64(evidence.embargo)
    end

    if isempty(evidence.folds)
        _add_protocol_error!(errors, "$stage: validation folds must be nonempty.")
        return horizon_map
    end

    fold_ids = Set{String}()
    refit_training_sets = Dict{String,Set{String}}()
    path_predictions = Set{Tuple{String,String}}()
    path_test_ids = Dict{String,Set{String}}()
    path_partitions = Dict{String,Vector{Tuple}}()
    nonresampled_test_ids = Set{String}()
    known_ids = Set(keys(horizon_map))
    validation_methods = Set(evidence.method_labels)

    for fold in evidence.folds
        if !_nonblank(fold.fold_id) || !_nonblank(fold.path_id)
            _add_protocol_error!(
                errors,
                "$stage: fold and path identifiers must be nonblank.",
            )
        elseif fold.fold_id in fold_ids
            _add_protocol_error!(errors, "$stage: fold identifiers must be unique.")
        else
            push!(fold_ids, fold.fold_id)
        end
        _nonblank(fold.refit_id) || _add_protocol_error!(
            errors,
            "$stage: every fold must record a refit identifier.",
        )
        components_valid = _validate_labels(
            fold.refit_components,
            _REFIT_COMPONENTS,
            stage,
            "refit components",
            errors,
        )
        if components_valid &&
           !issubset(required_refit_components, Set(fold.refit_components))
            _add_protocol_error!(
                errors,
                "$stage: every fold must refit all required pipeline components.",
            )
        end

        train_valid =
            _validate_string_tuple(fold.train_ids, stage, "training identifiers", errors)
        test_valid =
            _validate_string_tuple(fold.test_ids, stage, "test identifiers", errors)
        fit_valid = _validate_string_tuple(fold.fit_ids, stage, "fit identifiers", errors)
        prediction_valid = _validate_string_tuple(
            fold.prediction_ids,
            stage,
            "prediction identifiers",
            errors,
        )
        train_valid && test_valid && fit_valid && prediction_valid || continue

        train_set = Set(fold.train_ids)
        test_set = Set(fold.test_ids)
        fit_set = Set(fold.fit_ids)
        prediction_set = Set(fold.prediction_ids)
        !isempty(intersect(train_set, test_set)) && _add_protocol_error!(
            errors,
            "$stage: training and test membership must be disjoint.",
        )
        if fit_set != train_set || !isempty(intersect(fit_set, test_set))
            _add_protocol_error!(
                errors,
                "$stage: every fold must refit on its complete training membership only.",
            )
        end
        prediction_set == test_set || _add_protocol_error!(
            errors,
            "$stage: predictions must match the test membership.",
        )
        issubset(union(train_set, test_set, fit_set, prediction_set), known_ids) ||
            _add_protocol_error!(
                errors,
                "$stage: every fold identifier must have an event horizon.",
            )

        if !isempty(
            intersect(validation_methods, Set(("PURGED_CROSS_VALIDATION", "WALK_FORWARD"))),
        )
            !isempty(intersect(nonresampled_test_ids, test_set)) && _add_protocol_error!(
                errors,
                "$stage: ordinary temporal validation cannot duplicate test predictions.",
            )
            union!(nonresampled_test_ids, intersect(test_set, known_ids))
        end

        if _nonblank(fold.refit_id)
            previous = get(refit_training_sets, fold.refit_id, nothing)
            previous !== nothing &&
                previous != train_set &&
                _add_protocol_error!(
                    errors,
                    "$stage: a refit identifier cannot represent different training sets.",
                )
            refit_training_sets[fold.refit_id] = copy(train_set)
        end

        if _nonblank(fold.path_id)
            get!(path_test_ids, fold.path_id, Set{String}())
            get!(path_partitions, fold.path_id, Tuple[])
            push!(path_partitions[fold.path_id], Tuple(sort(collect(test_set))))
            for observation_id in fold.test_ids
                contribution = (fold.path_id, observation_id)
                contribution in path_predictions && _add_protocol_error!(
                    errors,
                    "$stage: a path cannot contain duplicate test contributions.",
                )
                push!(path_predictions, contribution)
                push!(path_test_ids[fold.path_id], observation_id)
            end
        end

        issubset(union(train_set, test_set), known_ids) || continue
        for train_id in fold.train_ids, test_id in fold.test_ids
            train_horizon = horizon_map[train_id]
            test_horizon = horizon_map[test_id]
            _intervals_overlap(train_horizon, test_horizon) && _add_protocol_error!(
                errors,
                "$stage: train and test event horizons must be purged.",
            )
            if embargo > 0.0 &&
               train_horizon[1] >= test_horizon[2] &&
               train_horizon[1] < test_horizon[2] + embargo
                _add_protocol_error!(
                    errors,
                    "$stage: post-test embargo observations must be excluded.",
                )
            end
        end
        if "WALK_FORWARD" in validation_methods && !isempty(train_set) && !isempty(test_set)
            latest_train_stop = maximum(horizon_map[item][2] for item in train_set)
            earliest_test_start = minimum(horizon_map[item][1] for item in test_set)
            latest_train_stop > earliest_test_start && _add_protocol_error!(
                errors,
                "$stage: walk-forward training must precede the test interval.",
            )
        end
    end

    if "COMBINATORIAL_PURGED_CROSS_VALIDATION" in validation_methods
        length(path_test_ids) >= 2 ||
            _add_protocol_error!(errors, "$stage: CPCV requires multiple complete paths.")
        if !isempty(known_ids) && any(ids -> ids != known_ids, values(path_test_ids))
            _add_protocol_error!(
                errors,
                "$stage: every CPCV path must predict each declared observation once.",
            )
        end
        signatures = [Tuple(sort(partition)) for partition in values(path_partitions)]
        length(unique(signatures)) == length(signatures) || _add_protocol_error!(
            errors,
            "$stage: CPCV paths must have distinct test partitions.",
        )
    end
    if "PURGED_CROSS_VALIDATION" in validation_methods &&
       !isempty(known_ids) &&
       nonresampled_test_ids != known_ids
        _add_protocol_error!(
            errors,
            "$stage: purged cross-validation must predict every observation once.",
        )
    end
    return horizon_map
end

function _protocol_adjacency(nodes, edges)
    result = Dict(node => Set{String}() for node in nodes)
    for (source, target) in edges
        if haskey(result, source) && haskey(result, target)
            push!(result[source], target)
        end
    end
    return result
end

function _protocol_acyclic(nodes, edges)
    adjacency = _protocol_adjacency(nodes, edges)
    indegree = Dict(node => 0 for node in nodes)
    for targets in values(adjacency), target in targets
        indegree[target] += 1
    end
    ready = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while !isempty(ready)
        node = pop!(ready)
        visited += 1
        for target in adjacency[node]
            indegree[target] -= 1
            indegree[target] == 0 && push!(ready, target)
        end
    end
    return visited == length(nodes)
end

function _protocol_reachable(adjacency, start)
    visited = Set{String}()
    pending = collect(get(adjacency, start, Set{String}()))
    while !isempty(pending)
        node = pop!(pending)
        node in visited && continue
        push!(visited, node)
        append!(pending, get(adjacency, node, Set{String}()))
    end
    return visited
end

function _protocol_reachable_avoiding(adjacency, start, blocked)
    start in blocked && return Set{String}()
    visited = Set{String}()
    pending = [node for node in get(adjacency, start, Set{String}()) if !(node in blocked)]
    while !isempty(pending)
        node = pop!(pending)
        if node in visited || node in blocked
            continue
        end
        push!(visited, node)
        append!(
            pending,
            [
                target for
                target in get(adjacency, node, Set{String}()) if !(target in blocked)
            ],
        )
    end
    return visited
end

function _protocol_d_separated(edges, left, right, conditioned)
    nodes = union(Set((left, right)), Set(conditioned))
    for (source, target) in edges
        push!(nodes, source)
        push!(nodes, target)
    end
    parents = Dict(node => Set{String}() for node in nodes)
    for (source, target) in edges
        push!(parents[target], source)
    end
    ancestors = union(Set((left, right)), Set(conditioned))
    pending = collect(ancestors)
    while !isempty(pending)
        node = pop!(pending)
        for parent in parents[node]
            if !(parent in ancestors)
                push!(ancestors, parent)
                push!(pending, parent)
            end
        end
    end
    moral = Dict(node => Set{String}() for node in ancestors)
    for (source, target) in edges
        if source in ancestors && target in ancestors
            push!(moral[source], target)
            push!(moral[target], source)
        end
    end
    for child in ancestors
        child_parents = collect(intersect(parents[child], ancestors))
        for first_index in eachindex(child_parents)
            for second_index = (first_index+1):length(child_parents)
                first = child_parents[first_index]
                second = child_parents[second_index]
                push!(moral[first], second)
                push!(moral[second], first)
            end
        end
    end
    if left in conditioned || right in conditioned
        return true
    end
    visited = Set(conditioned)
    pending = String[left]
    while !isempty(pending)
        node = pop!(pending)
        node == right && return false
        node in visited && continue
        push!(visited, node)
        append!(pending, [item for item in moral[node] if !(item in visited)])
    end
    return true
end

function _validate_discovery(stage, selected_variables, outcome, errors)
    label = "Stage 2"
    _validate_labels(
        stage.method_labels,
        _DISCOVERY_METHODS,
        label,
        "causal-discovery method labels",
        errors,
    )
    _nonblank(stage.graph_id) ||
        _add_protocol_error!(errors, "$label: graph identifier must be nonblank.")
    nodes_valid = _validate_string_tuple(stage.graph_nodes, label, "graph nodes", errors)
    nodes = nodes_valid ? Set(stage.graph_nodes) : Set{String}()
    if nodes_valid && nodes != union(selected_variables, Set((outcome,)))
        _add_protocol_error!(
            errors,
            "$label: graph nodes must match the selected variables and outcome.",
        )
    end
    edges_valid = true
    if length(unique(stage.directed_edges)) != length(stage.directed_edges)
        _add_protocol_error!(errors, "$label: directed edges contain duplicates.")
        edges_valid = false
    end
    for (source, target) in stage.directed_edges
        if !(source in nodes && target in nodes)
            _add_protocol_error!(
                errors,
                "$label: every directed edge must use graph nodes.",
            )
            edges_valid = false
        end
        if source == target
            _add_protocol_error!(errors, "$label: directed self-loops are not admissible.")
            edges_valid = false
        end
    end
    if edges_valid && !isempty(nodes) && !_protocol_acyclic(nodes, stage.directed_edges)
        _add_protocol_error!(errors, "$label: the resolved graph must be acyclic.")
    end
    stage.graph_kind == "DAG" ||
        _add_protocol_error!(errors, "$label: the accepted graph must be a resolved DAG.")
    isempty(stage.ambiguous_edges) || _add_protocol_error!(
        errors,
        "$label: identification-relevant ambiguity must be resolved.",
    )
    _validate_string_tuple(stage.assumptions, label, "graph assumptions", errors)
    safe_edges = edges_valid ? stage.directed_edges : ()
    return nodes, _protocol_adjacency(nodes, safe_edges), safe_edges
end

function _validate_adjustment_sets(value, stage, errors)
    if !(value isa Tuple) || isempty(value)
        _add_protocol_error!(errors, "$stage: admissible adjustment sets must be nonempty.")
        return Set{Set{String}}(), false
    end
    result = Set{Set{String}}()
    valid = true
    for adjustment_set in value
        if !_validate_string_tuple(
            adjustment_set,
            stage,
            "an admissible adjustment set",
            errors;
            allow_empty = true,
        )
            valid = false
            continue
        end
        frozen = Set(adjustment_set)
        if frozen in result
            _add_protocol_error!(
                errors,
                "$stage: admissible adjustment sets must be distinct.",
            )
            valid = false
        end
        push!(result, frozen)
    end
    return result, valid
end

function _validate_justifications(value, stage, errors)
    result = Dict{String,String}()
    for item in value
        if !(item isa Tuple) ||
           length(item) != 2 ||
           !_nonblank(item[1]) ||
           !_nonblank(item[2])
            _add_protocol_error!(errors, "$stage: a control justification is invalid.")
            continue
        end
        haskey(result, item[1]) && _add_protocol_error!(
            errors,
            "$stage: every control must have one justification.",
        )
        result[item[1]] = item[2]
    end
    return result
end

function _validate_adjustment(stage, graph_nodes, adjacency, directed_edges, errors)
    label = "Stage 3"
    treatment = _nonblank(stage.treatment) ? stage.treatment : ""
    outcome = _nonblank(stage.outcome) ? stage.outcome : ""
    if !_nonblank(stage.treatment) || !_nonblank(stage.outcome)
        _add_protocol_error!(errors, "$label: treatment and outcome must be nonblank.")
    elseif treatment == outcome
        _add_protocol_error!(errors, "$label: treatment and outcome must be distinct.")
    end
    treatment in graph_nodes && outcome in graph_nodes ||
        _add_protocol_error!(errors, "$label: treatment and outcome must be graph nodes.")
    stage.method_label in _ADJUSTMENT_METHODS ||
        _add_protocol_error!(errors, "$label: adjustment method is not source-listed.")
    stage.identified ||
        _add_protocol_error!(errors, "$label: the causal effect must be identified.")

    selected_valid = _validate_string_tuple(
        stage.selected_adjustment_set,
        label,
        "selected adjustment set",
        errors;
        allow_empty = true,
    )
    admissible_sets, _ =
        _validate_adjustment_sets(stage.admissible_adjustment_sets, label, errors)
    selected = selected_valid ? Set(stage.selected_adjustment_set) : Set{String}()
    if selected_valid && !(selected in admissible_sets)
        _add_protocol_error!(
            errors,
            "$label: the selected adjustment set must be declared admissible.",
        )
    end

    role_names = Dict{Symbol,Set{String}}()
    for field_name in (:confounders, :descendants, :mediators, :colliders, :instruments)
        value = getproperty(stage, field_name)
        valid = _validate_string_tuple(
            value,
            label,
            String(field_name),
            errors;
            allow_empty = true,
        )
        role_names[field_name] = valid ? Set(value) : Set{String}()
        if valid && !issubset(Set(value), graph_nodes)
            _add_protocol_error!(
                errors,
                "$label: every declared causal role must be a graph node.",
            )
        end
        if valid && !isempty(intersect(Set((treatment, outcome)), Set(value)))
            _add_protocol_error!(
                errors,
                "$label: treatment and outcome cannot be role variables.",
            )
        end
    end

    computed_descendants = _protocol_reachable(adjacency, treatment)
    delete!(computed_descendants, outcome)
    role_names[:descendants] == computed_descendants || _add_protocol_error!(
        errors,
        "$label: declared descendants must exactly match the graph.",
    )
    for mediator in role_names[:mediators]
        if !(mediator in computed_descendants) ||
           !(outcome in _protocol_reachable(adjacency, mediator))
            _add_protocol_error!(
                errors,
                "$label: declared mediators must lie on a causal path.",
            )
        end
    end
    incoming = Dict(node => 0 for node in graph_nodes)
    for targets in values(adjacency), target in targets
        incoming[target] += 1
    end
    any(node -> get(incoming, node, 0) < 2, role_names[:colliders]) && _add_protocol_error!(
        errors,
        "$label: declared colliders must have converging arrows.",
    )
    outcome in _protocol_reachable(adjacency, treatment) || _add_protocol_error!(
        errors,
        "$label: the graph must retain a treatment-to-outcome path.",
    )

    graph_confounders = Set(
        node for node in setdiff(graph_nodes, Set((treatment, outcome))) if
        treatment in _protocol_reachable(adjacency, node) &&
            outcome in _protocol_reachable_avoiding(adjacency, node, Set((treatment,)))
    )
    role_names[:confounders] == graph_confounders || _add_protocol_error!(
        errors,
        "$label: declared confounders must match graph-implied common causes.",
    )
    graph_colliders = Set(
        node for
        (node, degree) in incoming if degree >= 2 && !(node in (treatment, outcome))
    )
    role_names[:colliders] == graph_colliders || _add_protocol_error!(
        errors,
        "$label: declared colliders must match graph-implied converging nodes.",
    )
    graph_mediators = Set(
        node for node in setdiff(graph_nodes, Set((treatment, outcome))) if
        node in _protocol_reachable(adjacency, treatment) &&
            outcome in _protocol_reachable(adjacency, node)
    )
    role_names[:mediators] == graph_mediators || _add_protocol_error!(
        errors,
        "$label: declared mediators must match graph-implied causal-path nodes.",
    )
    isempty(intersect(role_names[:confounders], role_names[:instruments])) ||
        _add_protocol_error!(
            errors,
            "$label: a variable cannot be both a confounder and an instrument.",
        )

    instrument_edges = Tuple(edge for edge in directed_edges if edge[1] != treatment)
    for instrument in role_names[:instruments]
        treatment in _protocol_reachable_avoiding(adjacency, instrument, selected) ||
            _add_protocol_error!(
                errors,
                "$label: every declared instrument must be relevant in the graph.",
            )
        outcome in _protocol_reachable_avoiding(adjacency, instrument, Set((treatment,))) &&
            _add_protocol_error!(
                errors,
                "$label: every declared instrument must satisfy graph-level exclusion.",
            )
        _protocol_d_separated(instrument_edges, instrument, outcome, selected) ||
            _add_protocol_error!(
                errors,
                "$label: every declared instrument must be graph-separated from outcome shocks.",
            )
    end

    forbidden_controls = union(
        computed_descendants,
        role_names[:descendants],
        role_names[:mediators],
        role_names[:colliders],
        role_names[:instruments],
        Set((treatment, outcome)),
    )
    isempty(intersect(selected, forbidden_controls)) || _add_protocol_error!(
        errors,
        "$label: ordinary controls cannot include descendants, mediators, colliders, or instruments.",
    )
    for candidate in admissible_sets
        issubset(candidate, graph_nodes) || _add_protocol_error!(
            errors,
            "$label: every declared admissible set must use graph nodes.",
        )
        isempty(intersect(candidate, forbidden_controls)) || _add_protocol_error!(
            errors,
            "$label: no declared admissible set may contain a descendant, mediator, collider, or instrument.",
        )
        if stage.method_label == "BACKDOOR_ADJUSTMENT" &&
           !_protocol_backdoor_d_separated(directed_edges, treatment, outcome, candidate)
            _add_protocol_error!(
                errors,
                "$label: every declared admissible set must block graph-implied backdoor paths.",
            )
        end
    end
    issubset(selected, graph_nodes) ||
        _add_protocol_error!(errors, "$label: every selected control must be a graph node.")
    justifications = _validate_justifications(stage.control_justifications, label, errors)
    Set(keys(justifications)) == selected || _add_protocol_error!(
        errors,
        "$label: every selected control needs one justification.",
    )
    _validate_string_tuple(
        stage.open_backdoor_paths,
        label,
        "open backdoor paths",
        errors;
        allow_empty = true,
    )
    isempty(stage.open_backdoor_paths) ||
        _add_protocol_error!(errors, "$label: no backdoor path may remain open.")

    if stage.method_label != "FRONT_DOOR_ADJUSTMENT" &&
       stage.frontdoor_criteria_satisfied !== nothing
        _add_protocol_error!(
            errors,
            "$label: front-door evidence requires the front-door method.",
        )
    end
    instrument_premises = (
        stage.instrument_relevance_satisfied,
        stage.instrument_exclusion_satisfied,
        stage.instrument_exogeneity_satisfied,
    )
    if stage.method_label != "INSTRUMENTAL_VARIABLES" &&
       any(value -> value !== nothing, instrument_premises)
        _add_protocol_error!(
            errors,
            "$label: instrumental-variable premises require the IV method.",
        )
    end

    if stage.method_label == "BACKDOOR_ADJUSTMENT"
        _protocol_backdoor_d_separated(directed_edges, treatment, outcome, selected) ||
            _add_protocol_error!(
                errors,
                "$label: selected controls must block every graph-implied backdoor path.",
            )
    elseif stage.method_label == "FRONT_DOOR_ADJUSTMENT"
        if isempty(role_names[:mediators]) || stage.frontdoor_criteria_satisfied !== true
            _add_protocol_error!(
                errors,
                "$label: front-door criteria must be explicitly satisfied.",
            )
        elseif outcome in
               _protocol_reachable_avoiding(adjacency, treatment, role_names[:mediators])
            _add_protocol_error!(
                errors,
                "$label: front-door mediators must intercept every directed causal path.",
            )
        end
        for mediator in role_names[:mediators]
            if !_protocol_backdoor_d_separated(
                directed_edges,
                treatment,
                mediator,
                selected,
            ) ||
               !_protocol_backdoor_d_separated(
                directed_edges,
                mediator,
                outcome,
                union(Set((treatment,)), selected),
            )
                _add_protocol_error!(
                    errors,
                    "$label: graph-implied front-door backdoor criteria must hold.",
                )
            end
        end
    elseif stage.method_label == "INSTRUMENTAL_VARIABLES"
        if isempty(role_names[:instruments]) ||
           any(value -> value !== true, instrument_premises)
            _add_protocol_error!(
                errors,
                "$label: instrumental-variable premises must all hold.",
            )
        end
    end
    return nothing
end

function validate_causal_factor_protocol(report::CausalFactorProtocolReport)
    errors = String[]
    _nonblank(report.trial_family_id) ||
        _add_protocol_error!(errors, "Protocol: trial-family identifier must be nonblank.")
    trials_valid = _validate_string_tuple(
        report.declared_trial_ids,
        "Protocol",
        "declared trial identifiers",
        errors,
    )

    selection = report.variable_selection
    label = "Stage 1"
    selection.purpose in _RESEARCH_PURPOSES ||
        _add_protocol_error!(errors, "$label: research purpose is not source-listed.")
    selected_valid = _validate_string_tuple(
        selection.selected_variables,
        label,
        "selected variables",
        errors,
    )
    selected_variables = selected_valid ? Set(selection.selected_variables) : Set{String}()
    selection_methods_valid = _validate_labels(
        selection.method_labels,
        _VARIABLE_SELECTION_METHODS,
        label,
        "variable-selection method labels",
        errors,
    )
    selection_methods =
        selection_methods_valid ? Set(selection.method_labels) : Set{String}()
    if selection.validation === nothing
        selection_horizons = Dict{String,Tuple{Float64,Float64}}()
        if selection.overlapping_returns ||
           selection.strong_time_dependence ||
           "PERMUTATION_FEATURE_IMPORTANCE" in selection_methods
            _add_protocol_error!(
                errors,
                "$label: the declared selection design requires temporal validation.",
            )
        end
    else
        selection_horizons = _validate_fold_evidence(
            selection.validation,
            label,
            errors;
            required_refit_components = Set(("VARIABLE_SELECTION",)),
        )
        actual_overlap = _has_any_overlap(selection_horizons)
        selection.overlapping_returns == actual_overlap || _add_protocol_error!(
            errors,
            "$label: overlap declaration must match event horizons.",
        )
        validation_methods = Set(selection.validation.method_labels)
        if actual_overlap &&
           isempty(intersect(validation_methods, _PURGED_VALIDATION_METHODS))
            _add_protocol_error!(
                errors,
                "$label: overlapping returns require purged validation.",
            )
        end
        if selection.strong_time_dependence && (
            !_finite_protocol_real(selection.validation.embargo) ||
            Float64(selection.validation.embargo) <= 0.0
        )
            _add_protocol_error!(
                errors,
                "$label: strong time dependence requires an embargo.",
            )
        end
    end

    adjustment = report.causal_adjustment_set
    safe_outcome = _nonblank(adjustment.outcome) ? adjustment.outcome : ""
    graph_nodes, graph_adjacency, graph_edges = _validate_discovery(
        report.causal_discovery,
        selected_variables,
        safe_outcome,
        errors,
    )
    _validate_adjustment(adjustment, graph_nodes, graph_adjacency, graph_edges, errors)

    stage4 = report.causal_explanatory_and_predictive_power
    label = "Stage 4"
    task_types_valid = _validate_labels(
        stage4.task_types,
        Set(keys(_METRICS_BY_TASK)),
        label,
        "task types",
        errors,
    )
    allowed_metrics = Set{String}()
    if task_types_valid
        for task_type in stage4.task_types
            union!(allowed_metrics, _METRICS_BY_TASK[task_type])
        end
    end
    _nonblank(stage4.estimator_label) ||
        _add_protocol_error!(errors, "$label: estimator label must be nonblank.")
    explanatory_valid = _validate_labels(
        stage4.explanatory_metric_labels,
        allowed_metrics,
        label,
        "explanatory metric labels",
        errors;
        allow_empty = true,
    )
    predictive_valid = _validate_labels(
        stage4.predictive_metric_labels,
        allowed_metrics,
        label,
        "predictive metric labels",
        errors;
        allow_empty = true,
    )
    has_explanatory = explanatory_valid && !isempty(stage4.explanatory_metric_labels)
    has_predictive = predictive_valid && !isempty(stage4.predictive_metric_labels)
    has_explanatory ||
        has_predictive ||
        _add_protocol_error!(errors, "$label: at least one performance aspect is required.")
    reported_metrics = union(
        explanatory_valid ? Set(stage4.explanatory_metric_labels) : Set{String}(),
        predictive_valid ? Set(stage4.predictive_metric_labels) : Set{String}(),
    )
    if task_types_valid
        for task_type in stage4.task_types
            isempty(intersect(reported_metrics, _METRICS_BY_TASK[task_type])) &&
                _add_protocol_error!(
                    errors,
                    "$label: every declared task needs a compatible metric.",
                )
        end
    end
    if has_explanatory && !_nonblank(stage4.explanatory_evidence_id)
        _add_protocol_error!(errors, "$label: explanatory evidence identifier is required.")
    elseif !has_explanatory && stage4.explanatory_evidence_id !== nothing
        _add_protocol_error!(
            errors,
            "$label: explanatory evidence must match reported metrics.",
        )
    end
    if has_predictive && !_nonblank(stage4.predictive_evidence_id)
        _add_protocol_error!(errors, "$label: predictive evidence identifier is required.")
    elseif !has_predictive && stage4.predictive_evidence_id !== nothing
        _add_protocol_error!(
            errors,
            "$label: predictive evidence must match reported metrics.",
        )
    end
    if has_explanatory &&
       has_predictive &&
       stage4.explanatory_evidence_id == stage4.predictive_evidence_id
        _add_protocol_error!(
            errors,
            "$label: explanatory and predictive evidence must be separate.",
        )
    end
    if selection.purpose == "CAUSAL_ATTRIBUTION" && !has_explanatory
        _add_protocol_error!(
            errors,
            "$label: causal attribution requires explanatory evidence.",
        )
    end
    if selection.purpose == "RISK_PREMIA_HARVESTING" && !has_predictive
        _add_protocol_error!(
            errors,
            "$label: risk-premia harvesting requires predictive evidence.",
        )
    end
    _nonblank(stage4.naive_benchmark_id) ||
        _add_protocol_error!(errors, "$label: a naive benchmark identifier is required.")
    if stage4.multiclass_encoding !== nothing || stage4.averaging_method !== nothing
        stage4.multiclass_encoding == "ONE_VS_REST" || _add_protocol_error!(
            errors,
            "$label: multiclass encoding is not source-listed.",
        )
        stage4.averaging_method in ("MICRO", "MACRO", "WEIGHTED", "SAMPLES") ||
            _add_protocol_error!(
                errors,
                "$label: multiclass averaging is not source-listed.",
            )
        isempty(intersect(Set(stage4.task_types), Set(("PROBABILITY", "RANKING")))) &&
            _add_protocol_error!(
                errors,
                "$label: multiclass settings require a classification task.",
            )
    end
    stage4_horizons = _validate_fold_evidence(
        stage4.validation,
        label,
        errors;
        required_refit_components = Set(("VARIABLE_SELECTION", "CAUSAL_ESTIMATION")),
    )
    isempty(intersect(Set(stage4.validation.method_labels), _PURGED_VALIDATION_METHODS)) &&
        _add_protocol_error!(errors, "$label: purged cross-validation is required.")
    if !isempty(selection_horizons) && selection_horizons != stage4_horizons
        _add_protocol_error!(errors, "$label: event horizons must match Stage 1.")
    end
    if selection.strong_time_dependence && (
        !_finite_protocol_real(stage4.validation.embargo) ||
        Float64(stage4.validation.embargo) <= 0.0
    )
        _add_protocol_error!(errors, "$label: strong time dependence requires an embargo.")
    end

    portfolio = report.causal_portfolio_construction
    label = "Stage 5"
    portfolio_labels_valid = _validate_labels(
        portfolio.method_labels,
        _PORTFOLIO_METHODS,
        label,
        "portfolio-construction method labels",
        errors,
    )
    if portfolio_labels_valid && Set(portfolio.method_labels) != _PORTFOLIO_METHODS
        _add_protocol_error!(
            errors,
            "$label: all source portfolio considerations are required.",
        )
    end
    causal_valid = _validate_string_tuple(
        portfolio.causal_exposures,
        label,
        "causal exposures",
        errors,
    )
    neutral_valid = _validate_string_tuple(
        portfolio.controlled_unintended_exposures,
        label,
        "controlled unintended exposures",
        errors;
        allow_empty = true,
    )
    causal_exposures = causal_valid ? Set(portfolio.causal_exposures) : Set{String}()
    neutral_exposures =
        neutral_valid ? Set(portfolio.controlled_unintended_exposures) : Set{String}()
    if causal_valid && !issubset(causal_exposures, graph_nodes)
        _add_protocol_error!(errors, "$label: causal exposures must be graph variables.")
    end
    if causal_valid && any(
        exposure -> !(safe_outcome in _protocol_reachable(graph_adjacency, exposure)),
        causal_exposures,
    )
        _add_protocol_error!(
            errors,
            "$label: every causal exposure must have a directed path to the outcome.",
        )
    end
    if neutral_valid && !issubset(neutral_exposures, graph_nodes)
        _add_protocol_error!(
            errors,
            "$label: controlled unintended exposures must be graph variables.",
        )
    end
    safe_treatment = _nonblank(adjustment.treatment) ? adjustment.treatment : ""
    safe_treatment in causal_exposures || _add_protocol_error!(
        errors,
        "$label: the target causal factor must drive position sizing.",
    )
    isempty(intersect(causal_exposures, neutral_exposures)) || _add_protocol_error!(
        errors,
        "$label: causal and neutralized exposures must be disjoint.",
    )
    issubset(Set(adjustment.colliders), neutral_exposures) ||
        _add_protocol_error!(errors, "$label: declared colliders must be neutralized.")
    for (value, subject) in (
        (portfolio.cost_model_id, "cost model identifier"),
        (portfolio.constraint_set_id, "constraint-set identifier"),
        (portfolio.economic_rationale, "economic rationale"),
    )
        _nonblank(value) ||
            _add_protocol_error!(errors, "$label: $subject must be nonblank.")
    end
    _validate_string_tuple(
        portfolio.fragility_scenarios,
        label,
        "causal-fragility scenarios",
        errors,
    )
    if !_finite_protocol_real(portfolio.transfer_coefficient) ||
       !(-1.0 <= Float64(portfolio.transfer_coefficient) <= 1.0)
        _add_protocol_error!(
            errors,
            "$label: transfer coefficient must be finite and between minus one and one.",
        )
    end

    backtest = report.backtest
    label = "Stage 6"
    backtest_labels_valid = _validate_labels(
        backtest.method_labels,
        _BACKTEST_METHODS,
        label,
        "backtest method labels",
        errors,
    )
    backtest.trial_family_id == report.trial_family_id || _add_protocol_error!(
        errors,
        "$label: trial-family identifier must match the report.",
    )
    backtest.declared_trial_ids == report.declared_trial_ids || _add_protocol_error!(
        errors,
        "$label: declared trials must match the report exactly.",
    )
    non_monte_carlo =
        backtest_labels_valid ?
        setdiff(Set(backtest.method_labels), Set(("MONTE_CARLO",))) : Set{String}()
    if !isempty(non_monte_carlo)
        if backtest.validation === nothing
            _add_protocol_error!(
                errors,
                "$label: temporal backtests require validation evidence.",
            )
        else
            _validate_fold_evidence(
                backtest.validation,
                label,
                errors;
                required_refit_components = _REFIT_COMPONENTS,
            )
            if selection.strong_time_dependence && (
                !_finite_protocol_real(backtest.validation.embargo) ||
                Float64(backtest.validation.embargo) <= 0.0
            )
                _add_protocol_error!(
                    errors,
                    "$label: strong time dependence requires a backtest embargo.",
                )
            end
            issubset(non_monte_carlo, Set(backtest.validation.method_labels)) ||
                _add_protocol_error!(
                    errors,
                    "$label: backtest and validation method labels must agree.",
                )
        end
    elseif backtest.validation !== nothing
        _add_protocol_error!(
            errors,
            "$label: Monte Carlo-only evidence cannot carry temporal folds.",
        )
    end
    if backtest_labels_valid && "MONTE_CARLO" in backtest.method_labels
        _nonblank(backtest.monte_carlo_dgp) ||
            _add_protocol_error!(errors, "$label: Monte Carlo requires an explicit DGP.")
    elseif backtest.monte_carlo_dgp !== nothing
        _add_protocol_error!(
            errors,
            "$label: a Monte Carlo DGP requires the Monte Carlo method.",
        )
    end

    multiple_testing = report.multiple_testing_adjustments
    label = "Stage 7"
    methods_valid = _validate_labels(
        multiple_testing.method_labels,
        _MULTIPLE_TESTING_METHODS,
        label,
        "multiple-testing method labels",
        errors,
    )
    multiple_testing.trial_family_id == report.trial_family_id || _add_protocol_error!(
        errors,
        "$label: trial-family identifier must match the report.",
    )
    multiple_testing.declared_trial_ids == report.declared_trial_ids ||
        _add_protocol_error!(
            errors,
            "$label: declared trials must match the report exactly.",
        )

    covered_trials = Set{String}()
    partition_ids = Set{String}()
    if isempty(multiple_testing.family_partitions)
        _add_protocol_error!(errors, "$label: family partitions must be nonempty.")
    else
        for partition in multiple_testing.family_partitions
            if !(partition isa Tuple) || length(partition) != 2 || !_nonblank(partition[1])
                _add_protocol_error!(errors, "$label: a family partition is invalid.")
                continue
            end
            partition[1] in partition_ids && _add_protocol_error!(
                errors,
                "$label: family partition identifiers must be unique.",
            )
            push!(partition_ids, partition[1])
            _validate_string_tuple(
                partition[2],
                label,
                "partition trial identifiers",
                errors,
            ) || continue
            for trial_id in partition[2]
                trial_id in covered_trials && _add_protocol_error!(
                    errors,
                    "$label: family partitions must be disjoint.",
                )
                push!(covered_trials, trial_id)
            end
        end
        if trials_valid && covered_trials != Set(report.declared_trial_ids)
            _add_protocol_error!(
                errors,
                "$label: family partitions must cover declared trials exactly.",
            )
        end
    end
    if !_finite_protocol_real(multiple_testing.alpha) ||
       !(0.0 < Float64(multiple_testing.alpha) < 1.0)
        _add_protocol_error!(
            errors,
            "$label: alpha must be a finite number between zero and one.",
        )
    end
    multiple_testing.backtests_independent &&
        _add_protocol_error!(errors, "$label: backtest dependence must be represented.")

    methods = methods_valid ? Set(multiple_testing.method_labels) : Set{String}()
    if !isempty(intersect(methods, _P_VALUE_METHODS))
        _nonblank(multiple_testing.p_value_estimator) ||
            _add_protocol_error!(errors, "$label: p-value estimator must be declared.")
        _nonblank(multiple_testing.time_dependence_model) || _add_protocol_error!(
            errors,
            "$label: time dependence treatment must be declared.",
        )
        multiple_testing.p_value_inputs_assume_independence === false ||
            _add_protocol_error!(
                errors,
                "$label: p-value inputs must explicitly reject independence assumptions.",
            )
    elseif any(
        value -> value !== nothing,
        (
            multiple_testing.p_value_estimator,
            multiple_testing.time_dependence_model,
            multiple_testing.p_value_inputs_assume_independence,
        ),
    )
        _add_protocol_error!(errors, "$label: p-value inputs require a p-value correction.")
    end
    if "DEFLATED_SHARPE_RATIO" in methods
        trial_count = length(report.declared_trial_ids)
        if !_finite_protocol_real(multiple_testing.sharpe_variance) ||
           Float64(multiple_testing.sharpe_variance) <= 0.0
            _add_protocol_error!(
                errors,
                "$label: positive Sharpe-ratio variance is required.",
            )
        end
        if !_finite_protocol_real(multiple_testing.effective_trials) ||
           !(1.0 <= Float64(multiple_testing.effective_trials) < trial_count)
            _add_protocol_error!(
                errors,
                "$label: effective trials must be below total trials.",
            )
        end
        if multiple_testing.sample_length === nothing ||
           multiple_testing.sample_length isa Bool ||
           Int(multiple_testing.sample_length) <= 1
            _add_protocol_error!(
                errors,
                "$label: sample length must be an integer above one.",
            )
        end
        _finite_protocol_real(multiple_testing.skewness) ||
            _add_protocol_error!(errors, "$label: skewness must be finite.")
        if !_finite_protocol_real(multiple_testing.kurtosis) ||
           Float64(multiple_testing.kurtosis) < 1.0
            _add_protocol_error!(
                errors,
                "$label: Pearson kurtosis must be finite and at least one.",
            )
        elseif _finite_protocol_real(multiple_testing.skewness) &&
               abs(Float64(multiple_testing.skewness)) >
               sqrt(Float64(multiple_testing.kurtosis) - 1.0)
            _add_protocol_error!(
                errors,
                "$label: skewness and Pearson kurtosis must satisfy the moment inequality.",
            )
        end
        _nonblank(multiple_testing.selection_bias_evidence_id) ||
            _add_protocol_error!(errors, "$label: selection-bias evidence is required.")
    elseif any(
        value -> value !== nothing,
        (
            multiple_testing.sharpe_variance,
            multiple_testing.effective_trials,
            multiple_testing.sample_length,
            multiple_testing.skewness,
            multiple_testing.kurtosis,
            multiple_testing.selection_bias_evidence_id,
        ),
    )
        _add_protocol_error!(
            errors,
            "$label: Sharpe-correction inputs require the deflated Sharpe ratio.",
        )
    end

    isempty(errors) ||
        throw(ArgumentError("Causal factor protocol is invalid. " * join(errors, " ")))
    return report
end



function _protocol_backdoor_d_separated(edges, exposure, outcome, conditioned)
    backdoor_edges = Tuple(edge for edge in edges if edge[1] != exposure)
    return _protocol_d_separated(backdoor_edges, exposure, outcome, conditioned)
end
