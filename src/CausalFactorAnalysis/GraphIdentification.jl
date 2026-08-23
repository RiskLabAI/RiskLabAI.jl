const _DEFAULT_MAX_PATHS = 10_000
const _DEFAULT_MAX_PATH_STATES = 100_000
const _DEFAULT_MAX_CANDIDATES = 65_536

struct CausalDAG
    nodes::Tuple{Vararg{String}}
    directed_edges::Tuple{Vararg{Tuple{String,String}}}
    observed_nodes::Tuple{Vararg{String}}

    function CausalDAG(
        nodes::Tuple{Vararg{String}},
        directed_edges::Tuple{Vararg{Tuple{String,String}}},
        observed_nodes::Tuple{Vararg{String}},
        ::Val{:validated},
    )
        return new(nodes, directed_edges, observed_nodes)
    end
end

struct PathEvidence
    nodes::Tuple{Vararg{String}}
    colliders::Tuple{Vararg{String}}
    noncolliders::Tuple{Vararg{String}}
    conditioned_noncolliders::Tuple{Vararg{String}}
    activated_colliders::Tuple{Vararg{String}}
    blocking_colliders::Tuple{Vararg{String}}
    is_open::Bool
    is_directed_from_left::Bool
    is_backdoor_from_left::Bool
end

struct DSeparationEvidence
    left::Tuple{Vararg{String}}
    right::Tuple{Vararg{String}}
    conditioned::Tuple{Vararg{String}}
    separated::Bool
    paths::Tuple{Vararg{PathEvidence}}
end

struct BackdoorAdjustmentEvidence
    treatment::String
    outcome::String
    adjustment_set::Tuple{Vararg{String}}
    all_observed::Bool
    endpoints_in_set::Tuple{Vararg{String}}
    descendants_in_set::Tuple{Vararg{String}}
    open_backdoor_paths::Tuple{Vararg{PathEvidence}}
    admissible::Bool
end

struct FrontdoorAdjustmentEvidence
    treatment::String
    outcome::String
    mediator_set::Tuple{Vararg{String}}
    all_observed::Bool
    endpoints_in_set::Tuple{Vararg{String}}
    directed_path_exists::Bool
    unintercepted_directed_paths::Tuple{Vararg{PathEvidence}}
    open_treatment_mediator_backdoor_paths::Tuple{Vararg{PathEvidence}}
    open_mediator_outcome_backdoor_paths_given_treatment::Tuple{Vararg{PathEvidence}}
    graphically_admissible::Bool
    positivity_required::Bool
end

struct InstrumentEvidence
    instrument::String
    treatment::String
    outcome::String
    conditioned::Tuple{Vararg{String}}
    all_observed::Bool
    conditioning_disjoint::Bool
    conditioning_unaffected_by_treatment::Bool
    pearl_relevance::Bool
    pearl_exclusion_exogeneity::Bool
    pearl_graphical::Bool
    cfi_direct_relevance::Bool
    cfi_full_mediation::Bool
    cfi_exogeneity::Bool
    cfi_simple::Bool
end

struct NodeRoleEvidence
    node::String
    treatment::String
    outcome::String
    collider_on_paths::Tuple
    noncollider_on_paths::Tuple
    mediator_on_directed_paths::Tuple
    backdoor_noncollider_on_paths::Tuple
    common_cause_paths::Tuple
    descendant_of_treatment::Bool
    ancestor_of_outcome::Bool
end

function _node(value, subject::AbstractString)
    value isa AbstractString ||
        throw(ArgumentError("$subject must be a string node identifier"))
    result = String(value)
    isempty(strip(result)) && throw(ArgumentError("$subject must be nonblank"))
    return result
end

function _node_tuple(value, subject::AbstractString; allow_empty::Bool)
    value isa AbstractString &&
        throw(ArgumentError("$subject must be a collection, not a string"))
    items = try
        collect(value)
    catch
        throw(ArgumentError("$subject must be an iterable collection"))
    end
    nodes = String[_node(item, "$subject member") for item in items]
    allow_empty || !isempty(nodes) ||
        throw(ArgumentError("$subject must be nonempty"))
    length(unique(nodes)) == length(nodes) ||
        throw(ArgumentError("$subject must not contain duplicate nodes"))
    return Tuple(sort(nodes))
end

function _edge_tuple(value)
    value isa AbstractString &&
        throw(ArgumentError("directed_edges must be a collection"))
    items = try
        collect(value)
    catch
        throw(ArgumentError("directed_edges must be an iterable collection"))
    end
    edges = Tuple{String,String}[]
    for raw_edge in items
        raw_edge isa Tuple || raw_edge isa AbstractVector || throw(
            ArgumentError("each directed edge must be an ordered two-node sequence"),
        )
        length(raw_edge) == 2 ||
            throw(ArgumentError("each directed edge must contain two nodes"))
        push!(
            edges,
            (
                _node(raw_edge[1], "directed-edge source"),
                _node(raw_edge[2], "directed-edge target"),
            ),
        )
    end
    length(unique(edges)) == length(edges) ||
        throw(ArgumentError("directed_edges must not contain duplicate edges"))
    return Tuple(sort(edges))
end

function _children(nodes, edges)
    result = Dict(node => Set{String}() for node in nodes)
    for (source, target) in edges
        push!(result[source], target)
    end
    return result
end

function _parents(nodes, edges)
    result = Dict(node => Set{String}() for node in nodes)
    for (source, target) in edges
        push!(result[target], source)
    end
    return result
end

function _acyclic(nodes, edges)
    children = _children(nodes, edges)
    indegree = Dict(node => 0 for node in nodes)
    for (_, target) in edges
        indegree[target] += 1
    end
    ready = sort([node for node in nodes if indegree[node] == 0])
    visited = 0
    while !isempty(ready)
        node = pop!(ready)
        visited += 1
        for child in sort(collect(children[node]); rev = true)
            indegree[child] -= 1
            indegree[child] == 0 && push!(ready, child)
        end
    end
    return visited == length(nodes)
end

function CausalDAG(nodes, directed_edges, observed_nodes)
    normalized_nodes = _node_tuple(nodes, "nodes"; allow_empty = false)
    normalized_edges = _edge_tuple(directed_edges)
    observed = _node_tuple(
        observed_nodes,
        "observed_nodes";
        allow_empty = true,
    )
    node_set = Set(normalized_nodes)
    issubset(Set(observed), node_set) ||
        throw(ArgumentError("observed_nodes must be a subset of nodes"))
    for (source, target) in normalized_edges
        source in node_set && target in node_set || throw(
            ArgumentError("every directed edge endpoint must be a graph node"),
        )
        source != target ||
            throw(ArgumentError("directed self-loops are not admissible"))
    end
    _acyclic(normalized_nodes, normalized_edges) ||
        throw(ArgumentError("directed_edges must define an acyclic graph"))
    return CausalDAG(
        normalized_nodes,
        normalized_edges,
        observed,
        Val(:validated),
    )
end

function _endpoints(dag::CausalDAG, treatment, outcome)
    treatment_node = _node(treatment, "treatment")
    outcome_node = _node(outcome, "outcome")
    treatment_node != outcome_node ||
        throw(ArgumentError("treatment and outcome must be distinct"))
    treatment_node in dag.nodes && outcome_node in dag.nodes ||
        throw(ArgumentError("treatment and outcome must be known graph nodes"))
    observed = Set(dag.observed_nodes)
    treatment_node in observed && outcome_node in observed ||
        throw(ArgumentError("treatment and outcome must both be observed"))
    return treatment_node, outcome_node
end

function _known_nodes(dag, value, subject; allow_empty)
    nodes = _node_tuple(value, subject; allow_empty = allow_empty)
    issubset(Set(nodes), Set(dag.nodes)) ||
        throw(ArgumentError("$subject must contain only known graph nodes"))
    return nodes
end

function _limit(value, subject)
    value === nothing && return nothing
    value isa Bool && throw(ArgumentError("$subject must be positive or nothing"))
    value isa Integer ||
        throw(ArgumentError("$subject must be positive or nothing"))
    value >= 1 || throw(ArgumentError("$subject must be positive"))
    return Int(value)
end

function _descendants(dag::CausalDAG, node::String, edges = dag.directed_edges)
    children = _children(dag.nodes, edges)
    visited = Set{String}()
    pending = collect(children[node])
    while !isempty(pending)
        current = pop!(pending)
        current in visited && continue
        push!(visited, current)
        append!(pending, children[current])
    end
    return visited
end

function _ancestors(dag::CausalDAG, node::String)
    parents = _parents(dag.nodes, dag.directed_edges)
    visited = Set{String}()
    pending = collect(parents[node])
    while !isempty(pending)
        current = pop!(pending)
        current in visited && continue
        push!(visited, current)
        append!(pending, parents[current])
    end
    return visited
end

_without_outgoing(edges, nodes) =
    Tuple(edge for edge in edges if !(edge[1] in Set(nodes)))
_without_incoming(edges, nodes) =
    Tuple(edge for edge in edges if !(edge[2] in Set(nodes)))

function _has_directed_path(
    dag,
    left,
    right;
    edges = dag.directed_edges,
    blocked = (),
)
    blocked_nodes = Set(blocked)
    if left in blocked_nodes || right in blocked_nodes
        return false
    end
    children = _children(dag.nodes, edges)
    pending = String[left]
    visited = Set{String}()
    while !isempty(pending)
        node = pop!(pending)
        node == right && return true
        node in visited && continue
        push!(visited, node)
        append!(
            pending,
            [
                child for child in children[node] if
                !(child in visited) && !(child in blocked_nodes)
            ],
        )
    end
    return false
end

function _is_d_separated(dag, left, right, conditioned; edges = dag.directed_edges)
    parents = _parents(dag.nodes, edges)
    relevant = union(Set(left), Set(right), Set(conditioned))
    pending = collect(relevant)
    while !isempty(pending)
        node = pop!(pending)
        for parent in parents[node]
            if !(parent in relevant)
                push!(relevant, parent)
                push!(pending, parent)
            end
        end
    end
    moral = Dict(node => Set{String}() for node in relevant)
    for (source, target) in edges
        if source in relevant && target in relevant
            push!(moral[source], target)
            push!(moral[target], source)
        end
    end
    for child in relevant
        child_parents = sort([parent for parent in parents[child] if parent in relevant])
        for first_index in eachindex(child_parents)
            for second_index in first_index + 1:length(child_parents)
                first = child_parents[first_index]
                second = child_parents[second_index]
                push!(moral[first], second)
                push!(moral[second], first)
            end
        end
    end
    blocked = Set(conditioned)
    targets = Set(right)
    reachable = [node for node in left if !(node in blocked)]
    visited = Set{String}()
    while !isempty(reachable)
        node = pop!(reachable)
        node in targets && return false
        node in visited && continue
        push!(visited, node)
        append!(
            reachable,
            [
                neighbor for neighbor in moral[node] if
                !(neighbor in blocked) && !(neighbor in visited)
            ],
        )
    end
    return true
end

function _simple_paths(
    dag,
    left,
    right;
    edges = dag.directed_edges,
    max_paths = _DEFAULT_MAX_PATHS,
    max_path_states = _DEFAULT_MAX_PATH_STATES,
)
    adjacency = Dict(node => Set{String}() for node in dag.nodes)
    for (source, target) in edges
        push!(adjacency[source], target)
        push!(adjacency[target], source)
    end
    paths = Tuple[]
    states = 0
    for start in left, finish in right
        stack = Any[(start, (start,), Set((start,)))]
        while !isempty(stack)
            states += 1
            max_path_states !== nothing && states > max_path_states && throw(
                ArgumentError("complete path search exceeds max_path_states"),
            )
            node, path, visited = pop!(stack)
            if node == finish
                push!(paths, path)
                max_paths !== nothing && length(paths) > max_paths &&
                    throw(ArgumentError("complete path evidence exceeds max_paths"))
                continue
            end
            for neighbor in sort(collect(adjacency[node]); rev = true)
                if !(neighbor in visited)
                    push!(stack, (neighbor, (path..., neighbor), union(visited, (neighbor,))))
                end
            end
        end
    end
    sort!(paths; by = path -> (length(path), path))
    return Tuple(paths)
end

function _path_evidence(dag, path, conditioned; edges = dag.directed_edges)
    edge_set = Set(edges)
    conditioned_set = Set(conditioned)
    colliders = String[]
    noncolliders = String[]
    for index in 2:length(path) - 1
        previous, node, following = path[index - 1], path[index], path[index + 1]
        if (previous, node) in edge_set && (following, node) in edge_set
            push!(colliders, node)
        else
            push!(noncolliders, node)
        end
    end
    activated = String[]
    blocking = String[]
    for collider in colliders
        family = union(Set((collider,)), _descendants(dag, collider, edges))
        if !isempty(intersect(family, conditioned_set))
            push!(activated, collider)
        else
            push!(blocking, collider)
        end
    end
    conditioned_noncolliders =
        Tuple(node for node in noncolliders if node in conditioned_set)
    is_directed = all(
        index -> (path[index], path[index + 1]) in edge_set,
        1:length(path) - 1,
    )
    is_backdoor = length(path) > 1 && (path[2], path[1]) in edge_set
    return PathEvidence(
        path,
        Tuple(colliders),
        Tuple(noncolliders),
        conditioned_noncolliders,
        Tuple(activated),
        Tuple(blocking),
        isempty(conditioned_noncolliders) && isempty(blocking),
        is_directed,
        is_backdoor,
    )
end

function _d_separation(
    dag,
    left,
    right,
    conditioned;
    edges = dag.directed_edges,
    max_paths = _DEFAULT_MAX_PATHS,
    max_path_states = _DEFAULT_MAX_PATH_STATES,
)
    paths = Tuple(
        _path_evidence(dag, path, conditioned; edges = edges) for path in
        _simple_paths(
            dag,
            left,
            right;
            edges = edges,
            max_paths = max_paths,
            max_path_states = max_path_states,
        )
    )
    return DSeparationEvidence(
        left,
        right,
        conditioned,
        all(path -> !path.is_open, paths),
        paths,
    )
end

function d_separation(
    dag::CausalDAG,
    left,
    right,
    conditioned = ();
    max_paths = _DEFAULT_MAX_PATHS,
    max_path_states = _DEFAULT_MAX_PATH_STATES,
)
    left_nodes = _known_nodes(dag, left, "left"; allow_empty = false)
    right_nodes = _known_nodes(dag, right, "right"; allow_empty = false)
    conditioned_nodes = _known_nodes(
        dag,
        conditioned,
        "conditioned";
        allow_empty = true,
    )
    path_limit = _limit(max_paths, "max_paths")
    state_limit = _limit(max_path_states, "max_path_states")
    isempty(intersect(Set(left_nodes), Set(right_nodes))) &&
    isempty(intersect(Set(left_nodes), Set(conditioned_nodes))) &&
    isempty(intersect(Set(right_nodes), Set(conditioned_nodes))) || throw(
        ArgumentError("left, right, and conditioned must be pairwise disjoint"),
    )
    return _d_separation(
        dag,
        left_nodes,
        right_nodes,
        conditioned_nodes;
        max_paths = path_limit,
        max_path_states = state_limit,
    )
end

function check_backdoor_adjustment_set(
    dag::CausalDAG,
    treatment,
    outcome,
    adjustment_set = ();
    max_paths = _DEFAULT_MAX_PATHS,
    max_path_states = _DEFAULT_MAX_PATH_STATES,
)
    treatment_node, outcome_node = _endpoints(dag, treatment, outcome)
    adjustment = _known_nodes(
        dag,
        adjustment_set,
        "adjustment_set";
        allow_empty = true,
    )
    path_limit = _limit(max_paths, "max_paths")
    state_limit = _limit(max_path_states, "max_path_states")
    observed = Set(dag.observed_nodes)
    endpoints = Tuple(sort(collect(intersect(Set(adjustment), Set((treatment_node, outcome_node))))))
    descendants = Tuple(sort(collect(intersect(Set(adjustment), _descendants(dag, treatment_node)))))
    edges = _without_outgoing(dag.directed_edges, (treatment_node,))
    separation = _d_separation(
        dag,
        (treatment_node,),
        (outcome_node,),
        adjustment;
        edges = edges,
        max_paths = path_limit,
        max_path_states = state_limit,
    )
    open_paths = Tuple(path for path in separation.paths if path.is_open)
    all_observed = issubset(Set(adjustment), observed)
    admissible =
        all_observed && isempty(endpoints) && isempty(descendants) && isempty(open_paths)
    return BackdoorAdjustmentEvidence(
        treatment_node,
        outcome_node,
        adjustment,
        all_observed,
        endpoints,
        descendants,
        open_paths,
        admissible,
    )
end

function _inclusion_minimal_sets(valid_sets)
    ordered = sort(collect(valid_sets); by = item -> (length(item), item))
    result = Tuple[]
    for candidate in ordered
        candidate_set = Set(candidate)
        any(existing -> issubset(Set(existing), candidate_set), result) ||
            push!(result, candidate)
    end
    return Tuple(result)
end

mutable struct _SearchFrame
    next_index::Int
    evaluated::Bool
end

function _search_minimal_sets(candidates, is_valid, max_candidates)
    valid_sets = Tuple[]
    selected = String[]
    frames = _SearchFrame[_SearchFrame(1, false)]
    work = 0
    while !isempty(frames)
        frame = frames[end]
        candidate = Tuple(selected)
        if !frame.evaluated
            work += 1 + length(candidate)
            max_candidates !== nothing && work > max_candidates && throw(
                ArgumentError("exact minimal-set search exceeds max_candidates"),
            )
            candidate_set = Set(candidate)
            if any(existing -> issubset(Set(existing), candidate_set), valid_sets)
                pop!(frames)
                !isempty(selected) && pop!(selected)
                continue
            end
            if is_valid(candidate)
                isempty(candidate) && return ((),)
                push!(valid_sets, candidate)
                pop!(frames)
                pop!(selected)
                continue
            end
            frame.evaluated = true
            continue
        end
        if frame.next_index > length(candidates)
            pop!(frames)
            !isempty(selected) && pop!(selected)
            continue
        end
        index = frame.next_index
        frame.next_index += 1
        push!(selected, candidates[index])
        push!(frames, _SearchFrame(index + 1, false))
    end
    return _inclusion_minimal_sets(valid_sets)
end

function minimal_backdoor_adjustment_sets(
    dag::CausalDAG,
    treatment,
    outcome;
    max_candidates = _DEFAULT_MAX_CANDIDATES,
)
    treatment_node, outcome_node = _endpoints(dag, treatment, outcome)
    candidate_limit = _limit(max_candidates, "max_candidates")
    descendants = _descendants(dag, treatment_node)
    candidates = Tuple(
        node for node in dag.observed_nodes if
        !(node in (treatment_node, outcome_node)) && !(node in descendants)
    )
    edges = _without_outgoing(dag.directed_edges, (treatment_node,))
    is_valid(candidate) = _is_d_separated(
        dag,
        (treatment_node,),
        (outcome_node,),
        candidate;
        edges = edges,
    )
    return _search_minimal_sets(candidates, is_valid, candidate_limit)
end

function _directed_path_tuples(
    dag,
    left,
    right;
    max_paths = _DEFAULT_MAX_PATHS,
    max_path_states = _DEFAULT_MAX_PATH_STATES,
)
    children = _children(dag.nodes, dag.directed_edges)
    parents = _parents(dag.nodes, dag.directed_edges)
    can_reach_right = Set((right,))
    pending = String[right]
    while !isempty(pending)
        node = pop!(pending)
        for parent in parents[node]
            if !(parent in can_reach_right)
                push!(can_reach_right, parent)
                push!(pending, parent)
            end
        end
    end
    paths = Tuple[]
    stack = Any[(left, (left,))]
    states = 0
    while !isempty(stack)
        states += 1
        max_path_states !== nothing && states > max_path_states && throw(
            ArgumentError("directed-path search exceeds max_path_states"),
        )
        node, path = pop!(stack)
        if node == right
            push!(paths, path)
            max_paths !== nothing && length(paths) > max_paths &&
                throw(ArgumentError("directed-path evidence exceeds max_paths"))
            continue
        end
        for child in sort(collect(children[node]); rev = true)
            child in can_reach_right && push!(stack, (child, (path..., child)))
        end
    end
    sort!(paths; by = path -> (length(path), path))
    return Tuple(paths)
end

function _directed_paths(dag, left, right; max_paths, max_path_states)
    return Tuple(
        _path_evidence(dag, path, ()) for path in _directed_path_tuples(
            dag,
            left,
            right;
            max_paths = max_paths,
            max_path_states = max_path_states,
        )
    )
end

function check_frontdoor_adjustment_set(
    dag::CausalDAG,
    treatment,
    outcome,
    mediator_set;
    max_paths = _DEFAULT_MAX_PATHS,
    max_path_states = _DEFAULT_MAX_PATH_STATES,
)
    treatment_node, outcome_node = _endpoints(dag, treatment, outcome)
    mediators = _known_nodes(
        dag,
        mediator_set,
        "mediator_set";
        allow_empty = true,
    )
    path_limit = _limit(max_paths, "max_paths")
    state_limit = _limit(max_path_states, "max_path_states")
    observed = Set(dag.observed_nodes)
    endpoints = Tuple(sort(collect(intersect(Set(mediators), Set((treatment_node, outcome_node))))))
    directed_paths = _directed_paths(
        dag,
        treatment_node,
        outcome_node;
        max_paths = path_limit,
        max_path_states = state_limit,
    )
    unintercepted = Tuple(
        path for path in directed_paths if
        isempty(intersect(Set(path.nodes[2:end-1]), Set(mediators)))
    )
    if !isempty(mediators) && isempty(endpoints)
        treatment_edges = _without_outgoing(dag.directed_edges, (treatment_node,))
        treatment_mediator = _d_separation(
            dag,
            (treatment_node,),
            mediators,
            ();
            edges = treatment_edges,
            max_paths = path_limit,
            max_path_states = state_limit,
        )
        mediator_edges = _without_outgoing(dag.directed_edges, mediators)
        mediator_outcome = _d_separation(
            dag,
            mediators,
            (outcome_node,),
            (treatment_node,);
            edges = mediator_edges,
            max_paths = path_limit,
            max_path_states = state_limit,
        )
        open_treatment_mediator =
            Tuple(path for path in treatment_mediator.paths if path.is_open)
        open_mediator_outcome =
            Tuple(path for path in mediator_outcome.paths if path.is_open)
    else
        open_treatment_mediator = ()
        open_mediator_outcome = ()
    end
    all_observed = issubset(Set(mediators), observed)
    directed_path_exists = !isempty(directed_paths)
    admissible =
        !isempty(mediators) && all_observed && isempty(endpoints) &&
        directed_path_exists && isempty(unintercepted) &&
        isempty(open_treatment_mediator) && isempty(open_mediator_outcome)
    return FrontdoorAdjustmentEvidence(
        treatment_node,
        outcome_node,
        mediators,
        all_observed,
        endpoints,
        directed_path_exists,
        unintercepted,
        open_treatment_mediator,
        open_mediator_outcome,
        admissible,
        true,
    )
end

function minimal_frontdoor_adjustment_sets(
    dag::CausalDAG,
    treatment,
    outcome;
    max_candidates = _DEFAULT_MAX_CANDIDATES,
)
    treatment_node, outcome_node = _endpoints(dag, treatment, outcome)
    candidate_limit = _limit(max_candidates, "max_candidates")
    _has_directed_path(dag, treatment_node, outcome_node) || return ()
    candidates = Tuple(
        node for node in dag.observed_nodes if
        !(node in (treatment_node, outcome_node))
    )
    treatment_edges = _without_outgoing(dag.directed_edges, (treatment_node,))
    function is_valid(candidate)
        isempty(candidate) && return false
        _has_directed_path(
            dag,
            treatment_node,
            outcome_node;
            blocked = candidate,
        ) && return false
        _is_d_separated(
            dag,
            (treatment_node,),
            candidate,
            ();
            edges = treatment_edges,
        ) || return false
        mediator_edges = _without_outgoing(dag.directed_edges, candidate)
        return _is_d_separated(
            dag,
            candidate,
            (outcome_node,),
            (treatment_node,);
            edges = mediator_edges,
        )
    end
    return _search_minimal_sets(candidates, is_valid, candidate_limit)
end

function check_instrument(
    dag::CausalDAG,
    instrument,
    treatment,
    outcome,
    conditioned = (),
)
    treatment_node, outcome_node = _endpoints(dag, treatment, outcome)
    instrument_node = _node(instrument, "instrument")
    instrument_node in dag.nodes ||
        throw(ArgumentError("instrument must be a known graph node"))
    !(instrument_node in (treatment_node, outcome_node)) || throw(
        ArgumentError("instrument, treatment, and outcome must be distinct"),
    )
    controls = _known_nodes(
        dag,
        conditioned,
        "conditioned";
        allow_empty = true,
    )
    forbidden = Set((instrument_node, treatment_node, outcome_node))
    conditioning_disjoint = isempty(intersect(Set(controls), forbidden))
    observed = Set(dag.observed_nodes)
    all_observed = issubset(union(forbidden, Set(controls)), observed)
    unaffected = isempty(intersect(Set(controls), _descendants(dag, treatment_node)))
    if conditioning_disjoint
        treatment_edges = _without_incoming(dag.directed_edges, (treatment_node,))
        pearl_relevance = !_is_d_separated(
            dag,
            (instrument_node,),
            (treatment_node,),
            controls,
        )
        pearl_exclusion = _is_d_separated(
            dag,
            (instrument_node,),
            (outcome_node,),
            controls;
            edges = treatment_edges,
        )
    else
        pearl_relevance = false
        pearl_exclusion = false
    end
    pearl_graphical =
        all_observed && conditioning_disjoint && unaffected &&
        pearl_relevance && pearl_exclusion

    edge_set = Set(dag.directed_edges)
    cfi_direct = (instrument_node, treatment_node) in edge_set
    instrument_outcome_path = _has_directed_path(
        dag,
        instrument_node,
        outcome_node,
    )
    bypasses_treatment = _has_directed_path(
        dag,
        instrument_node,
        outcome_node;
        blocked = (treatment_node,),
    )
    cfi_full_mediation = instrument_outcome_path && !bypasses_treatment
    instrument_edges = _without_outgoing(dag.directed_edges, (instrument_node,))
    cfi_exogeneity = _is_d_separated(
        dag,
        (instrument_node,),
        (outcome_node,),
        ();
        edges = instrument_edges,
    )
    cfi_simple =
        instrument_node in observed && cfi_direct &&
        cfi_full_mediation && cfi_exogeneity
    return InstrumentEvidence(
        instrument_node,
        treatment_node,
        outcome_node,
        controls,
        all_observed,
        conditioning_disjoint,
        unaffected,
        pearl_relevance,
        pearl_exclusion,
        pearl_graphical,
        cfi_direct,
        cfi_full_mediation,
        cfi_exogeneity,
        cfi_simple,
    )
end

function causal_role_evidence(
    dag::CausalDAG,
    treatment,
    outcome,
    node;
    max_paths = _DEFAULT_MAX_PATHS,
    max_path_states = _DEFAULT_MAX_PATH_STATES,
)
    treatment_node, outcome_node = _endpoints(dag, treatment, outcome)
    role_node = _node(node, "node")
    role_node in dag.nodes ||
        throw(ArgumentError("node must be a known graph node"))
    !(role_node in (treatment_node, outcome_node)) ||
        throw(ArgumentError("node must differ from treatment and outcome"))
    path_limit = _limit(max_paths, "max_paths")
    state_limit = _limit(max_path_states, "max_path_states")
    paths = Tuple(
        _path_evidence(dag, path, ()) for path in _simple_paths(
            dag,
            (treatment_node,),
            (outcome_node,);
            max_paths = path_limit,
            max_path_states = state_limit,
        )
    )
    collider_paths = Tuple(path.nodes for path in paths if role_node in path.colliders)
    noncollider_paths =
        Tuple(path.nodes for path in paths if role_node in path.noncolliders)
    mediator_paths = Tuple(
        path.nodes for path in paths if
        path.is_directed_from_left && role_node in path.nodes[2:end-1]
    )
    backdoor_paths = Tuple(
        path.nodes for path in paths if
        path.is_backdoor_from_left && role_node in path.noncolliders
    )
    edge_set = Set(dag.directed_edges)
    common_cause_paths = Tuple[]
    for path in paths
        role_node in path.noncolliders || continue
        index = findfirst(==(role_node), path.nodes)
        directed_to_treatment = all(
            position -> (path.nodes[position], path.nodes[position - 1]) in edge_set,
            index:-1:2,
        )
        directed_to_outcome = all(
            position -> (path.nodes[position], path.nodes[position + 1]) in edge_set,
            index:length(path.nodes) - 1,
        )
        directed_to_treatment && directed_to_outcome &&
            push!(common_cause_paths, path.nodes)
    end
    return NodeRoleEvidence(
        role_node,
        treatment_node,
        outcome_node,
        collider_paths,
        noncollider_paths,
        mediator_paths,
        backdoor_paths,
        Tuple(common_cause_paths),
        role_node in _descendants(dag, treatment_node),
        role_node in _ancestors(dag, outcome_node),
    )
end
