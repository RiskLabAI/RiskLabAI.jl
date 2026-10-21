using Test
using TOML
using RiskLabAI

const GraphUnderTest = RiskLabAI.CausalFactorAnalysis
const graph_fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "graphs_python_320.toml"))
function graph_plain(value)
    value isa GraphUnderTest.TreatmentOutcomeRole && return string(value)
    parentmodule(typeof(value)) === GraphUnderTest || return value
    return Dict(string(name) => graph_plain(getproperty(value, name)) for name in propertynames(value))
end
graph_plain(value::NamedTuple) = Dict(string(key) => graph_plain(item) for (key, item) in pairs(value))
graph_plain(value::Tuple) = [graph_plain(item) for item in value]
graph_plain(value::AbstractVector) = [graph_plain(item) for item in value]
graph_plain(value::AbstractMatrix) = [graph_plain(collect(row)) for row in eachrow(value)]
graph_fixture_matrix(rows) = reduce(vcat, permutedims.(rows))

@testset "Public DAG evidence against independently checked Python 3.2.0 graphs" begin
    dags = Dict{Int, GraphUnderTest.CausalDAG}()
    for row in graph_fixtures["graphs"]
        dag = GraphUnderTest.CausalDAG(row["nodes"], row["directed_edges"], row["observed_nodes"])
        @test dag.nodes == Tuple(row["nodes"])
        @test dag.directed_edges == Tuple(Tuple(edge) for edge in row["directed_edges"])
        @test dag.observed_nodes == Tuple(row["observed_nodes"])
        dags[row["id"]] = dag
    end
    for row in graph_fixtures["separation"]
        result = GraphUnderTest.d_separation(dags[row["graph"]], row["left"], row["right"], row["conditioned"])
        @test graph_plain(result) == row["expected"]
    end
    for (group, method) in (("backdoor", GraphUnderTest.check_backdoor_adjustment_set), ("frontdoor", GraphUnderTest.check_frontdoor_adjustment_set))
        for row in graph_fixtures[group]
            result = method(dags[row["graph"]], row["treatment"], row["outcome"], row["candidate"])
            @test graph_plain(result) == row["expected"]
        end
    end
    for row in graph_fixtures["minimal"]
        @test graph_plain(GraphUnderTest.minimal_backdoor_adjustment_sets(dags[row["graph"]], row["treatment"], row["outcome"])) == row["expected"]
    end
    for row in graph_fixtures["instrument"]
        result = GraphUnderTest.check_instrument(dags[row["graph"]], row["instrument"], row["treatment"], row["outcome"], row["conditioned"])
        @test graph_plain(result) == row["expected"]
    end
    for (group, method) in (("path_roles", GraphUnderTest.causal_role_evidence), ("one_hop", GraphUnderTest.classify_treatment_outcome_role))
        for row in graph_fixtures[group]
            @test graph_plain(method(dags[row["graph"]], row["treatment"], row["outcome"], row["node"])) == row["expected"]
        end
    end
end

@testset "Immutable DAG domains and complete-search limits" begin
    nodes, edges, observed = ["B", "A", "D", "C"], [["A", "B"], ["A", "C"], ["B", "D"], ["C", "D"]], ["A", "B", "C", "D"]
    diamond = GraphUnderTest.CausalDAG(nodes, edges, observed)
    nodes[1], edges[1][1], observed[1] = "Z", "Z", "Z"
    @test diamond.nodes == ("A", "B", "C", "D")
    @test diamond.directed_edges[1] == ("A", "B")
    @test diamond.observed_nodes == ("A", "B", "C", "D")
    @test_throws ArgumentError GraphUnderTest.d_separation(diamond, ("A",), ("D",); max_paths = 1)
    @test_throws ArgumentError GraphUnderTest.d_separation(diamond, ("A",), ("D",); max_path_states = 1)
    @test length(GraphUnderTest.d_separation(diamond, ("A",), ("D",); max_paths = nothing, max_path_states = nothing).paths) == 2
    @test_throws ArgumentError GraphUnderTest.check_frontdoor_adjustment_set(diamond, "A", "D", ("B", "C"); max_paths = 1)
    fork = GraphUnderTest.CausalDAG(("X", "Y", "Z"), (("Z", "X"), ("Z", "Y")), ("X", "Y", "Z"))
    @test_throws ArgumentError GraphUnderTest.minimal_backdoor_adjustment_sets(fork, "X", "Y"; max_candidates = 1)
    @test GraphUnderTest.minimal_backdoor_adjustment_sets(fork, "X", "Y"; max_candidates = nothing) == (("Z",),)
    @test GraphUnderTest.minimal_backdoor_adjustment_sets(diamond, "A", "D"; max_candidates = 1) == ((),)
    collider = GraphUnderTest.CausalDAG(("X", "Y", "M", "D"), (("X", "M"), ("Y", "M"), ("M", "D")), ("X", "Y", "D"))
    @test GraphUnderTest.d_separation(collider, ("X",), ("Y",)).separated
    @test !GraphUnderTest.d_separation(collider, ("X",), ("Y",), ("D",)).separated
    multi = GraphUnderTest.d_separation(diamond, ("A", "B"), ("D",), ("C",))
    @test !multi.separated && !isempty(multi.paths)
    @test_throws ArgumentError GraphUnderTest.CausalDAG([], [], [])
    @test_throws ArgumentError GraphUnderTest.CausalDAG(("A", "A"), (), ("A",))
    @test_throws ArgumentError GraphUnderTest.CausalDAG(("A",), (("A", "A"),), ("A",))
    @test_throws ArgumentError GraphUnderTest.CausalDAG(("A", "B"), (("A", "B"), ("B", "A")), ("A", "B"))
    @test_throws ArgumentError GraphUnderTest.CausalDAG(("A", "B"), (("A", "B"), ("A", "B")), ("A", "B"))
    @test_throws ArgumentError GraphUnderTest.CausalDAG(("A",), (("A", "B"),), ("A",))
    @test_throws ArgumentError GraphUnderTest.CausalDAG(("A",), (), ("B",))
    @test_throws ArgumentError GraphUnderTest.CausalDAG("A", (), ("A",))
    @test_throws ArgumentError GraphUnderTest.CausalDAG((" ",), (), ())
    for tolerance in (0, -1, true, 1.2)
        @test_throws ArgumentError GraphUnderTest.d_separation(diamond, ("A",), ("D",); max_paths = tolerance)
        @test_throws ArgumentError GraphUnderTest.minimal_backdoor_adjustment_sets(diamond, "A", "D"; max_candidates = tolerance)
    end
    @test_throws ArgumentError GraphUnderTest.d_separation(diamond, "A", ("D",))
    @test_throws ArgumentError GraphUnderTest.d_separation(diamond, ("A",), ("A",))
    @test_throws ArgumentError GraphUnderTest.d_separation(diamond, ("A",), ("D",), ("D",))
    @test_throws ArgumentError GraphUnderTest.d_separation(diamond, ("A",), ("Z",))
    @test_throws ArgumentError GraphUnderTest.check_backdoor_adjustment_set(collider, "X", "M")
    @test_throws ArgumentError GraphUnderTest.check_instrument(diamond, "A", "A", "D")
    @test_throws ArgumentError GraphUnderTest.causal_role_evidence(diamond, "A", "D", "A")
    reversed = GraphUnderTest.CausalDAG(("T", "Y", "Z"), (("Y", "Z"), ("Z", "T")), ("T", "Y", "Z"))
    @test_throws ArgumentError GraphUnderTest.classify_treatment_outcome_role(reversed, "T", "Y", "Z")
end

@testset "Supplied structural mechanisms and independent snapshots" begin
    for row in graph_fixtures["structural"]
        graph_row = graph_fixtures["graphs"][row["graph"]]
        dag = GraphUnderTest.CausalDAG(graph_row["nodes"], graph_row["directed_edges"], graph_row["observed_nodes"])
        coefficients = Dict((entry["source"], entry["target"]) => entry["value"] for entry in row["coefficients"])
        parent_names = Dict(node => sort!([source for (source, target) in dag.directed_edges if target == node]) for node in dag.nodes)
        mechanisms = Dict(node => ((parents, noise) -> parents * [coefficients[(parent, node)] for parent in parent_names[node]] + noise) for node in dag.nodes)
        exogenous = Dict(node => Vector{Float64}(row["noise"][node]) for node in dag.nodes)
        result = GraphUnderTest.evaluate_structural_causal_model(dag, mechanisms, exogenous)
        @test result isa GraphUnderTest.StructuralCausalModelResult
        @test result.node_order == Tuple(row["expected"]["node_order"])
        @test result.values ≈ graph_fixture_matrix(row["expected"]["values"]) rtol = 1e-10 atol = 1e-12
        @test result.values isa Matrix{Float64}
        exogenous[first(dag.nodes)] .= 1000
        @test result.values ≈ graph_fixture_matrix(row["expected"]["values"]) rtol = 1e-10 atol = 1e-12
    end
    dag = GraphUnderTest.CausalDAG(("A", "B"), (("A", "B"),), ("A", "B"))
    inputs = Dict("A" => [1.0, 2.0], "B" => [3.0, 4.0])
    clean = Dict("A" => ((parents, noise) -> noise), "B" => ((parents, noise) -> parents[:, 1] + noise))
    result = GraphUnderTest.evaluate_structural_causal_model(dag, clean, inputs)
    @test result.values == [1.0 4.0; 2.0 6.0]
    mutation = Dict("A" => ((parents, noise) -> (noise[1] = 90.0; noise)), "B" => clean["B"])
    GraphUnderTest.evaluate_structural_causal_model(dag, mutation, inputs)
    @test inputs == Dict("A" => [1.0, 2.0], "B" => [3.0, 4.0])
    mutation = Dict("A" => clean["A"], "B" => ((parents, noise) -> (parents[1, 1] = 90.0; noise)))
    GraphUnderTest.evaluate_structural_causal_model(dag, mutation, inputs)
    @test inputs == Dict("A" => [1.0, 2.0], "B" => [3.0, 4.0])
    @test_throws ArgumentError GraphUnderTest.evaluate_structural_causal_model(dag, Dict("A" => clean["A"]), inputs)
    @test_throws ArgumentError GraphUnderTest.evaluate_structural_causal_model(dag, Dict("A" => 3.0, "B" => clean["B"]), inputs)
    @test_throws ArgumentError GraphUnderTest.evaluate_structural_causal_model(dag, clean, Dict("A" => [1.0], "B" => [1.0, 2.0]))
    @test_throws ArgumentError GraphUnderTest.evaluate_structural_causal_model(dag, clean, Dict("A" => [NaN], "B" => [1.0]))
    @test_throws ArgumentError GraphUnderTest.evaluate_structural_causal_model(dag, clean, Dict("A" => [true], "B" => [false]))
    @test_throws ArgumentError GraphUnderTest.evaluate_structural_causal_model(dag, clean, Dict("A" => Float64[], "B" => Float64[]))
    for wrong in ([1.0], [NaN, 0.0], [true, false], 1.0)
        @test_throws ArgumentError GraphUnderTest.evaluate_structural_causal_model(dag, Dict("A" => ((parents, noise) -> wrong), "B" => clean["B"]), inputs)
    end
end
