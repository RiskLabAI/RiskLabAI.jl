function graph_role_dag(edges = (); nodes = ("N", "X", "Y"), observed = nodes)
    return CausalDAG(Tuple(nodes), Tuple(edges), Tuple(observed))
end

function graph_role_transitive_closure(nodes, edges)
    index = Dict(node => position for (position, node) in enumerate(nodes))
    reachable = falses(length(nodes), length(nodes))
    for (source, target) in edges
        reachable[index[source], index[target]] = true
    end
    for middle in eachindex(nodes), left in eachindex(nodes), right in eachindex(nodes)
        reachable[left, right] |= reachable[left, middle] & reachable[middle, right]
    end
    return reachable, index
end

function graph_role_four_node_dags()
    nodes = ("A", "B", "C", "D")
    pairs = (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"))
    graphs = Tuple[]
    for encoding = 0:(3^length(pairs)-1)
        states = digits(encoding; base = 3, pad = length(pairs))
        edges = Tuple{String,String}[]
        for (pair, state) in zip(pairs, states)
            state == 1 && push!(edges, pair)
            state == 2 && push!(edges, reverse(pair))
        end
        try
            CausalDAG(nodes, Tuple(edges), nodes)
            push!(graphs, Tuple(edges))
        catch error
            error isa ArgumentError || rethrow()
        end
    end
    return nodes, graphs
end

@testset "accepted-DAG factor control roles" begin
    graph = CausalDAG(
        ("A", "D", "L_A", "L_D", "O", "T"),
        (("A", "T"), ("D", "L_D"), ("L_A", "A"), ("T", "D")),
        ("A", "D", "O", "T"),
    )
    roles = factor_control_roles(graph, "T")
    @test roles.target_factor == "T"
    @test roles.ancestor_factors == ("A",)
    @test roles.descendant_factors == ("D",)
    @test roles.other_factors == ("O",)
    @test roles.unobserved_ancestors == ("L_A",)
    @test roles.unobserved_descendants == ("L_D",)

    reversed_graph = CausalDAG(
        reverse(graph.nodes),
        reverse(graph.directed_edges),
        reverse(graph.observed_nodes),
    )
    @test factor_control_roles(reversed_graph, "T").ancestor_factors ==
          roles.ancestor_factors
    @test factor_control_roles(reversed_graph, "T").descendant_factors ==
          roles.descendant_factors

    @test_throws ArgumentError factor_control_roles(nothing, "T")
    @test_throws ArgumentError factor_control_roles(graph, :T)
    @test_throws ArgumentError factor_control_roles(graph, "missing")
    @test_throws ArgumentError factor_control_roles(graph, "L_A")
end

@testset "exhaustive four-node factor-role oracle" begin
    nodes, graphs = graph_role_four_node_dags()
    queries = 0
    for edges in graphs
        dag = CausalDAG(nodes, edges, nodes)
        reachable, index = graph_role_transitive_closure(nodes, edges)
        for target in nodes
            expected_ancestors = Tuple(
                sort([
                    node for node in nodes if
                    node != target && reachable[index[node], index[target]]
                ]),
            )
            expected_descendants = Tuple(
                sort([
                    node for node in nodes if
                    node != target && reachable[index[target], index[node]]
                ]),
            )
            expected_other = Tuple(
                sort([
                    node for node in nodes if node != target &&
                        !(node in expected_ancestors) &&
                        !(node in expected_descendants)
                ],),
            )
            actual = factor_control_roles(dag, target)
            @test actual.ancestor_factors == expected_ancestors
            @test actual.descendant_factors == expected_descendants
            @test actual.other_factors == expected_other
            @test isempty(actual.unobserved_ancestors)
            @test isempty(actual.unobserved_descendants)
            queries += 1
        end
    end
    @test length(graphs) == 543
    @test queries == 2_172
end

@testset "one-hop treatment/outcome roles" begin
    expected = Dict(
        (true, false, false, false) => :cause_of_treatment,
        (false, true, false, false) => :consequence_of_treatment,
        (false, false, true, false) => :cause_of_outcome,
        (false, false, false, true) => :consequence_of_outcome,
        (true, false, true, false) => :confounder,
        (false, true, false, true) => :collider,
        (false, true, true, false) => :mediator,
        (false, false, false, false) => :independent,
    )
    relationship_edges = ((), (("N", "X"),), (("X", "N"),))
    outcome_edges = ((), (("N", "Y"),), (("Y", "N"),))
    observed_signatures = 0
    for treatment_edges in relationship_edges, result_edges in outcome_edges
        edges = (treatment_edges..., result_edges...)
        signature = (
            ("N", "X") in edges,
            ("X", "N") in edges,
            ("N", "Y") in edges,
            ("Y", "N") in edges,
        )
        dag = graph_role_dag(edges)
        if haskey(expected, signature)
            evidence = classify_treatment_outcome_role(dag, "X", "Y", "N")
            @test evidence.node == "N"
            @test evidence.treatment == "X"
            @test evidence.outcome == "Y"
            @test evidence.role == TreatmentOutcomeRole(expected[signature])
            @test (
                evidence.node_to_treatment,
                evidence.treatment_to_node,
                evidence.node_to_outcome,
                evidence.outcome_to_node,
            ) == signature
        else
            @test signature == (true, false, false, true)
            @test_throws ArgumentError classify_treatment_outcome_role(dag, "X", "Y", "N")
        end
        observed_signatures += 1
    end
    @test observed_signatures == 9

    indirect = CausalDAG(
        ("A", "N", "X", "Y"),
        (("N", "A"), ("A", "X"), ("N", "Y")),
        ("A", "N", "X", "Y"),
    )
    @test classify_treatment_outcome_role(indirect, "X", "Y", "N").role ==
          TreatmentOutcomeRole(:cause_of_outcome)

    @test string(TreatmentOutcomeRole(:confounder)) == "confounder"
    @test TreatmentOutcomeRole("collider") == TreatmentOutcomeRole(:collider)
    @test_throws ArgumentError TreatmentOutcomeRole(:unsupported)
    @test_throws ArgumentError classify_treatment_outcome_role(
        graph_role_dag(),
        "X",
        "X",
        "N",
    )
    @test_throws ArgumentError classify_treatment_outcome_role(
        graph_role_dag(),
        "X",
        "Y",
        :N,
    )
    @test_throws ArgumentError classify_treatment_outcome_role(nothing, "X", "Y", "N")
end
