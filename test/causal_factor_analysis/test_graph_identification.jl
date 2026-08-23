function book_dag(nodes, edges; observed = nodes)
    return CausalDAG(Tuple(nodes), Tuple(edges), Tuple(observed))
end

function oracle_acyclic(nodes, edges)
    children = Dict(node => String[] for node in nodes)
    indegree = Dict(node => 0 for node in nodes)
    for (source, target) in edges
        push!(children[source], target)
        indegree[target] += 1
    end
    pending = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while !isempty(pending)
        node = pop!(pending)
        visited += 1
        for child in children[node]
            indegree[child] -= 1
            indegree[child] == 0 && push!(pending, child)
        end
    end
    return visited == length(nodes)
end

function oracle_four_node_dags()
    nodes = ("A", "B", "C", "D")
    pairs = (
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    )
    graphs = Tuple[]
    for encoding in 0:(3^length(pairs) - 1)
        states = digits(encoding; base = 3, pad = length(pairs))
        edges = Tuple{String,String}[]
        for (pair, state) in zip(pairs, states)
            state == 1 && push!(edges, pair)
            state == 2 && push!(edges, reverse(pair))
        end
        oracle_acyclic(nodes, edges) && push!(graphs, Tuple(sort(edges)))
    end
    return nodes, Tuple(graphs)
end

function oracle_descendants(nodes, edges, start)
    children = Dict(node => String[] for node in nodes)
    for (source, target) in edges
        push!(children[source], target)
    end
    result = Set{String}()
    pending = copy(children[start])
    while !isempty(pending)
        node = pop!(pending)
        node in result && continue
        push!(result, node)
        append!(pending, children[node])
    end
    return result
end

function oracle_simple_paths(nodes, edges, left, right)
    adjacency = Dict(node => String[] for node in nodes)
    for (source, target) in edges
        push!(adjacency[source], target)
        push!(adjacency[target], source)
    end
    result = Tuple[]
    stack = Any[(left, (left,), Set((left,)))]
    while !isempty(stack)
        node, path, visited = pop!(stack)
        if node == right
            push!(result, path)
            continue
        end
        for neighbor in adjacency[node]
            !(neighbor in visited) && push!(
                stack,
                (neighbor, (path..., neighbor), union(visited, Set((neighbor,)))),
            )
        end
    end
    return result
end

function oracle_path_open(nodes, edges, path, conditioned)
    edge_set = Set(edges)
    conditioned_set = Set(conditioned)
    for index in 2:(length(path) - 1)
        previous, node, following = path[index - 1], path[index], path[index + 1]
        collider = (previous, node) in edge_set && (following, node) in edge_set
        if collider
            family = union(Set((node,)), oracle_descendants(nodes, edges, node))
            isempty(intersect(family, conditioned_set)) && return false
        elseif node in conditioned_set
            return false
        end
    end
    return true
end

function oracle_d_separated(nodes, edges, left, right, conditioned)
    return all(
        path -> !oracle_path_open(nodes, edges, path, conditioned),
        oracle_simple_paths(nodes, edges, left, right),
    )
end

function oracle_subsets(items)
    result = Tuple[]
    for mask in 0:(2^length(items) - 1)
        push!(
            result,
            Tuple(items[index] for index in eachindex(items) if !iszero(mask & (1 << (index - 1)))),
        )
    end
    return result
end

function oracle_minimal_backdoor_sets(nodes, edges, treatment, outcome)
    descendants = oracle_descendants(nodes, edges, treatment)
    candidates = Tuple(
        node for node in nodes if
        !(node in (treatment, outcome)) && !(node in descendants)
    )
    backdoor_edges = Tuple(edge for edge in edges if edge[1] != treatment)
    valid = Tuple[
        candidate for candidate in oracle_subsets(candidates) if
        oracle_d_separated(nodes, backdoor_edges, treatment, outcome, candidate)
    ]
    sort!(valid; by = item -> (length(item), item))
    minimal = Tuple[]
    for candidate in valid
        candidate_set = Set(candidate)
        any(item -> issubset(Set(item), candidate_set), minimal) || push!(minimal, candidate)
    end
    return Tuple(minimal)
end

function oracle_has_directed_path(nodes, edges, left, right; blocked = ())
    blocked_set = Set(blocked)
    if left in blocked_set || right in blocked_set
        return false
    end
    children = Dict(node => String[] for node in nodes)
    for (source, target) in edges
        push!(children[source], target)
    end
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
                !(child in blocked_set) && !(child in visited)
            ],
        )
    end
    return false
end

function oracle_sets_d_separated(nodes, edges, left, right, conditioned)
    return all(
        oracle_d_separated(nodes, edges, left_node, right_node, conditioned) for
        left_node in left for right_node in right
    )
end

function oracle_minimal_frontdoor_sets(nodes, edges, treatment, outcome)
    oracle_has_directed_path(nodes, edges, treatment, outcome) || return ()
    candidates = Tuple(node for node in nodes if !(node in (treatment, outcome)))
    treatment_edges = Tuple(edge for edge in edges if edge[1] != treatment)
    valid = Tuple[]
    for candidate in oracle_subsets(candidates)
        isempty(candidate) && continue
        oracle_has_directed_path(
            nodes,
            edges,
            treatment,
            outcome;
            blocked = candidate,
        ) && continue
        oracle_sets_d_separated(
            nodes,
            treatment_edges,
            (treatment,),
            candidate,
            (),
        ) || continue
        mediator_edges = Tuple(edge for edge in edges if !(edge[1] in candidate))
        oracle_sets_d_separated(
            nodes,
            mediator_edges,
            candidate,
            (outcome,),
            (treatment,),
        ) || continue
        push!(valid, candidate)
    end
    sort!(valid; by = item -> (length(item), item))
    minimal = Tuple[]
    for candidate in valid
        candidate_set = Set(candidate)
        any(item -> issubset(Set(item), candidate_set), minimal) || push!(minimal, candidate)
    end
    return Tuple(minimal)
end

function oracle_instrument_profile(nodes, edges, instrument, treatment, outcome, conditioned)
    observed = Set(nodes)
    controls = Set(conditioned)
    forbidden = Set((instrument, treatment, outcome))
    disjoint = isempty(intersect(controls, forbidden))
    all_observed = issubset(union(forbidden, controls), observed)
    unaffected = isempty(intersect(controls, oracle_descendants(nodes, edges, treatment)))
    if disjoint
        pearl_relevance = !oracle_d_separated(
            nodes,
            edges,
            instrument,
            treatment,
            conditioned,
        )
        incoming_removed = Tuple(edge for edge in edges if edge[2] != treatment)
        pearl_exclusion = oracle_d_separated(
            nodes,
            incoming_removed,
            instrument,
            outcome,
            conditioned,
        )
    else
        pearl_relevance = false
        pearl_exclusion = false
    end
    pearl = all_observed && disjoint && unaffected && pearl_relevance && pearl_exclusion
    cfi_direct = (instrument, treatment) in Set(edges)
    cfi_full_mediation = oracle_has_directed_path(
        nodes,
        edges,
        instrument,
        outcome,
    ) && !oracle_has_directed_path(
        nodes,
        edges,
        instrument,
        outcome;
        blocked = (treatment,),
    )
    instrument_edges = Tuple(edge for edge in edges if edge[1] != instrument)
    cfi_exogeneity = oracle_d_separated(
        nodes,
        instrument_edges,
        instrument,
        outcome,
        (),
    )
    cfi_simple = cfi_direct && cfi_full_mediation && cfi_exogeneity
    return (
        all_observed,
        disjoint,
        unaffected,
        pearl_relevance,
        pearl_exclusion,
        pearl,
        cfi_direct,
        cfi_full_mediation,
        cfi_exogeneity,
        cfi_simple,
    )
end

function oracle_role_profile(nodes, edges, treatment, outcome, role_node)
    edge_set = Set(edges)
    paths = oracle_simple_paths(nodes, edges, treatment, outcome)
    sort!(paths; by = path -> (length(path), path))
    collider_paths = Tuple[]
    noncollider_paths = Tuple[]
    mediator_paths = Tuple[]
    backdoor_paths = Tuple[]
    common_cause_paths = Tuple[]
    for path in paths
        colliders = Set{String}()
        noncolliders = Set{String}()
        for index in 2:(length(path) - 1)
            previous, node, following = path[index - 1], path[index], path[index + 1]
            if (previous, node) in edge_set && (following, node) in edge_set
                push!(colliders, node)
            else
                push!(noncolliders, node)
            end
        end
        role_node in colliders && push!(collider_paths, path)
        role_node in noncolliders && push!(noncollider_paths, path)
        directed = all(
            (path[index], path[index + 1]) in edge_set for
            index in 1:(length(path) - 1)
        )
        directed && role_node in path[2:(end - 1)] && push!(mediator_paths, path)
        (path[2], path[1]) in edge_set && role_node in noncolliders &&
            push!(backdoor_paths, path)
        if role_node in noncolliders
            index = findfirst(==(role_node), path)
            to_treatment = all(
                (path[position], path[position - 1]) in edge_set for
                position in index:-1:2
            )
            to_outcome = all(
                (path[position], path[position + 1]) in edge_set for
                position in index:(length(path) - 1)
            )
            to_treatment && to_outcome && push!(common_cause_paths, path)
        end
    end
    return (
        Tuple(collider_paths),
        Tuple(noncollider_paths),
        Tuple(mediator_paths),
        Tuple(backdoor_paths),
        Tuple(common_cause_paths),
        oracle_has_directed_path(nodes, edges, treatment, role_node),
        oracle_has_directed_path(nodes, edges, role_node, outcome),
    )
end

@testset "graph identification" begin
    fork = book_dag(("X", "Y", "Z"), (("Z", "X"), ("Z", "Y")))
    @test !d_separation(fork, ("X",), ("Y",)).separated
    @test d_separation(fork, ("X",), ("Y",), ("Z",)).separated
    @test minimal_backdoor_adjustment_sets(fork, "X", "Y") == (("Z",),)

    collider = book_dag(("X", "Y", "Z"), (("X", "Z"), ("Y", "Z")))
    @test d_separation(collider, ("X",), ("Y",)).separated
    @test !d_separation(collider, ("X",), ("Y",), ("Z",)).separated

    chain = book_dag(("X", "Y", "Z"), (("X", "Z"), ("Z", "Y")))
    @test !d_separation(chain, ("X",), ("Y",)).separated
    @test d_separation(chain, ("X",), ("Y",), ("Z",)).separated
    @test causal_role_evidence(chain, "X", "Y", "Z").mediator_on_directed_paths ==
          (("X", "Z", "Y"),)

    frontdoor = book_dag(
        ("M", "U", "X", "Y"),
        (("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y"));
        observed = ("M", "X", "Y"),
    )
    @test minimal_backdoor_adjustment_sets(frontdoor, "X", "Y") == ()
    @test minimal_frontdoor_adjustment_sets(frontdoor, "X", "Y") == (("M",),)

    factor_edges = (
        ("MOM", "HML"),
        ("MOM", "PC"),
        ("HML", "OI"),
        ("OI", "PC"),
    )
    observed_factor_graph = book_dag(("HML", "MOM", "OI", "PC"), factor_edges)
    @test minimal_backdoor_adjustment_sets(
        observed_factor_graph,
        "HML",
        "PC",
    ) == (("MOM",),)
    latent_factor_graph = book_dag(
        ("HML", "MOM", "OI", "PC"),
        factor_edges;
        observed = ("HML", "OI", "PC"),
    )
    @test minimal_backdoor_adjustment_sets(latent_factor_graph, "HML", "PC") == ()
    @test minimal_frontdoor_adjustment_sets(latent_factor_graph, "HML", "PC") ==
          (("OI",),)

    instrument_graph = book_dag(
        ("U", "W", "X", "Y"),
        (("W", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")),
    )
    instrument = check_instrument(instrument_graph, "W", "X", "Y")
    @test instrument.pearl_graphical
    @test instrument.cfi_simple

    mediator = book_dag(
        ("W", "X", "Y", "Z"),
        (("X", "Z"), ("W", "Z"), ("Z", "Y"), ("W", "Y")),
    )
    @test minimal_backdoor_adjustment_sets(mediator, "X", "Y") == ((),)
    @test check_backdoor_adjustment_set(mediator, "X", "Y", ()).admissible
    @test !check_backdoor_adjustment_set(mediator, "X", "Y", ("Z",)).admissible

    @test_throws ArgumentError CausalDAG(
        ("X", "Y"),
        (("X", "Y"), ("Y", "X")),
        ("X", "Y"),
    )
    @test_throws ArgumentError d_separation(fork, ("X",), ("X",))
end

@testset "exhaustive independent four-node graph oracle" begin
    nodes, graphs = oracle_four_node_dags()
    @test length(graphs) == 543
    d_separation_queries = 0
    backdoor_queries = 0
    mismatches = Any[]
    for edges in graphs
        dag = book_dag(nodes, edges)
        for treatment in nodes, outcome in nodes
            treatment == outcome && continue
            remaining = Tuple(node for node in nodes if !(node in (treatment, outcome)))
            for conditioned in oracle_subsets(remaining)
                expected = oracle_d_separated(
                    nodes,
                    edges,
                    treatment,
                    outcome,
                    conditioned,
                )
                actual = d_separation(
                    dag,
                    (treatment,),
                    (outcome,),
                    conditioned,
                ).separated
                d_separation_queries += 1
                if expected != actual && length(mismatches) < 10
                    push!(mismatches, (:d_separation, edges, treatment, outcome, conditioned, expected, actual))
                end
            end
            expected_sets = oracle_minimal_backdoor_sets(
                nodes,
                edges,
                treatment,
                outcome,
            )
            actual_sets = minimal_backdoor_adjustment_sets(dag, treatment, outcome)
            backdoor_queries += 1
            if expected_sets != actual_sets && length(mismatches) < 10
                push!(mismatches, (:backdoor, edges, treatment, outcome, expected_sets, actual_sets))
            end
        end
    end
    @test d_separation_queries == 26_064
    @test backdoor_queries == 6_516
    @test isempty(mismatches)
end

@testset "exhaustive independent front-door and instrument oracles" begin
    nodes, graphs = oracle_four_node_dags()
    frontdoor_queries = 0
    instrument_queries = 0
    mismatches = Any[]
    for edges in graphs
        dag = book_dag(nodes, edges)
        for treatment in nodes, outcome in nodes
            treatment == outcome && continue
            expected_frontdoor = oracle_minimal_frontdoor_sets(
                nodes,
                edges,
                treatment,
                outcome,
            )
            actual_frontdoor = minimal_frontdoor_adjustment_sets(
                dag,
                treatment,
                outcome,
            )
            frontdoor_queries += 1
            if expected_frontdoor != actual_frontdoor && length(mismatches) < 10
                push!(mismatches, (
                    :frontdoor,
                    edges,
                    treatment,
                    outcome,
                    expected_frontdoor,
                    actual_frontdoor,
                ))
            end
            remaining = Tuple(node for node in nodes if !(node in (treatment, outcome)))
            for instrument in remaining
                controls = Tuple(node for node in remaining if node != instrument)
                for conditioned in oracle_subsets(controls)
                    expected = oracle_instrument_profile(
                        nodes,
                        edges,
                        instrument,
                        treatment,
                        outcome,
                        conditioned,
                    )
                    actual = check_instrument(
                        dag,
                        instrument,
                        treatment,
                        outcome,
                        conditioned,
                    )
                    observed = (
                        actual.all_observed,
                        actual.conditioning_disjoint,
                        actual.conditioning_unaffected_by_treatment,
                        actual.pearl_relevance,
                        actual.pearl_exclusion_exogeneity,
                        actual.pearl_graphical,
                        actual.cfi_direct_relevance,
                        actual.cfi_full_mediation,
                        actual.cfi_exogeneity,
                        actual.cfi_simple,
                    )
                    instrument_queries += 1
                    if expected != observed && length(mismatches) < 10
                        push!(mismatches, (
                            :instrument,
                            edges,
                            instrument,
                            treatment,
                            outcome,
                            conditioned,
                            expected,
                            observed,
                        ))
                    end
                end
            end
        end
    end
    @test frontdoor_queries == 6_516
    @test instrument_queries == 26_064
    @test isempty(mismatches)
end

@testset "exhaustive independent path-relative role oracle" begin
    nodes, graphs = oracle_four_node_dags()
    queries = 0
    mismatches = Any[]
    for edges in graphs
        dag = book_dag(nodes, edges)
        for treatment in nodes, outcome in nodes
            treatment == outcome && continue
            for role_node in nodes
                role_node in (treatment, outcome) && continue
                expected = oracle_role_profile(
                    nodes,
                    edges,
                    treatment,
                    outcome,
                    role_node,
                )
                actual = causal_role_evidence(dag, treatment, outcome, role_node)
                observed = (
                    actual.collider_on_paths,
                    actual.noncollider_on_paths,
                    actual.mediator_on_directed_paths,
                    actual.backdoor_noncollider_on_paths,
                    actual.common_cause_paths,
                    actual.descendant_of_treatment,
                    actual.ancestor_of_outcome,
                )
                queries += 1
                if expected != observed && length(mismatches) < 10
                    push!(mismatches, (
                        edges,
                        treatment,
                        outcome,
                        role_node,
                        expected,
                        observed,
                    ))
                end
            end
        end
    end
    @test queries == 13_032
    @test isempty(mismatches)
end
