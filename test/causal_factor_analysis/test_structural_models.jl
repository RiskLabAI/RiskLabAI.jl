function structural_chain()
    return CausalDAG(("X", "Y", "Z"), (("X", "Y"), ("Y", "Z")), ("X", "Y", "Z"))
end

@testset "deterministic structural causal model" begin
    graph = structural_chain()
    exogenous =
        Dict("X" => [1.0, 2.0, 3.0], "Y" => [0.5, -0.5, 1.0], "Z" => [2.0, 2.0, 2.0])
    mechanisms = Dict{String,Any}(
        "X" => ((parents, noise) -> noise),
        "Y" => ((parents, noise) -> 2.0 .* parents[:, 1] .+ noise),
        "Z" => ((parents, noise) -> parents[:, 1] .^ 2 .+ noise),
    )
    result = evaluate_structural_causal_model(graph, mechanisms, exogenous)
    x = exogenous["X"]
    y = 2.0 .* x .+ exogenous["Y"]
    z = y .^ 2 .+ exogenous["Z"]
    @test result.node_order == ("X", "Y", "Z")
    @test result.values ≈ hcat(x, y, z) atol = 1.0e-14

    named_result = evaluate_structural_causal_model(
        graph,
        (
            X = (parents, noise) -> noise,
            Y = (parents, noise) -> 2.0 .* parents[:, 1] .+ noise,
            Z = (parents, noise) -> parents[:, 1] .^ 2 .+ noise,
        ),
        (X = exogenous["X"], Y = exogenous["Y"], Z = exogenous["Z"]),
    )
    @test named_result.node_order == result.node_order
    @test named_result.values == result.values
end

@testset "lexical topological and parent order" begin
    graph = CausalDAG(("D", "C", "B", "A"), (("B", "D"), ("A", "D")), ("D", "C", "B", "A"))
    parent_inputs = Matrix{Float64}[]
    d_mechanism = function (parents, noise)
        push!(parent_inputs, copy(parents))
        return parents[:, 1] .+ 10.0 .* parents[:, 2] .+ noise
    end
    result = evaluate_structural_causal_model(
        graph,
        Dict{String,Any}(
            "D" => d_mechanism,
            "C" => ((parents, noise) -> noise),
            "B" => ((parents, noise) -> noise),
            "A" => ((parents, noise) -> noise),
        ),
        Dict(
            "D" => [0.0, 0.0],
            "C" => [30.0, 31.0],
            "B" => [20.0, 21.0],
            "A" => [1.0, 2.0],
        ),
    )
    @test result.node_order == ("A", "B", "C", "D")
    @test only(parent_inputs) == [1.0 20.0; 2.0 21.0]
    @test result.values[:, 4] == [201.0, 212.0]
end

@testset "independent linear-system oracle" begin
    nodes = ("A", "B", "C", "D")
    graph = CausalDAG(
        reverse(nodes),
        (("A", "C"), ("B", "C"), ("C", "D"), ("A", "D")),
        reverse(nodes),
    )
    coefficients = [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        1.5 -0.25 0.0 0.0
        0.5 0.0 2.0 0.0
    ]
    noises = [
        1.0 2.0 0.5 -1.0
        -2.0 1.0 1.5 0.25
        0.0 -1.0 2.0 3.0
    ]
    mechanisms = Dict{String,Any}(
        "A" => ((parents, noise) -> noise),
        "B" => ((parents, noise) -> noise),
        "C" => ((parents, noise) ->
                1.5 .* parents[:, 1] .- 0.25 .* parents[:, 2] .+ noise),
        "D" =>
            ((parents, noise) -> 0.5 .* parents[:, 1] .+ 2.0 .* parents[:, 2] .+ noise),
    )
    exogenous = Dict(node => noises[:, column] for (column, node) in enumerate(nodes))
    result = evaluate_structural_causal_model(graph, mechanisms, exogenous)
    oracle = transpose((I - coefficients) \ transpose(noises))
    @test result.node_order == nodes
    @test result.values ≈ oracle atol = 1.0e-14
end

@testset "SCM input and output snapshots" begin
    source = [1.0, 2.0]
    mechanism_output = copy(source)
    received = Tuple{Matrix{Float64},Vector{Float64}}[]
    root = function (parents, noise)
        push!(received, (parents, noise))
        noise .+= 5.0
        return mechanism_output
    end
    graph = CausalDAG(("X",), (), ("X",))
    result = evaluate_structural_causal_model(graph, Dict("X" => root), Dict("X" => source))
    source .= 10.0
    mechanism_output .= 20.0
    @test size(only(received)[1]) == (2, 0)
    @test only(received)[2] == [6.0, 7.0]
    @test result.values[:, 1] == [1.0, 2.0]

    result.values[1, 1] = -1.0
    @test source == [10.0, 10.0]
    @test mechanism_output == [20.0, 20.0]
end

@testset "SCM validation" begin
    graph = CausalDAG(("X",), (), ("X",))
    root = (parents, noise) -> noise
    @test_throws ArgumentError evaluate_structural_causal_model(nothing, Dict(), Dict())
    @test_throws ArgumentError evaluate_structural_causal_model(
        graph,
        (),
        Dict("X" => [1.0]),
    )
    @test_throws ArgumentError evaluate_structural_causal_model(
        graph,
        Dict("X" => root),
        (),
    )
    @test_throws ArgumentError evaluate_structural_causal_model(
        graph,
        Dict(),
        Dict("X" => [1.0]),
    )
    @test_throws ArgumentError evaluate_structural_causal_model(
        graph,
        Dict("X" => root),
        Dict(),
    )
    @test_throws ArgumentError evaluate_structural_causal_model(
        graph,
        Dict("X" => root, "Y" => root),
        Dict("X" => [1.0]),
    )
    @test_throws ArgumentError evaluate_structural_causal_model(
        graph,
        Dict{Any,Any}("X" => root, :X => root),
        Dict("X" => [1.0]),
    )
    @test_throws ArgumentError evaluate_structural_causal_model(
        graph,
        Dict("X" => 1),
        Dict("X" => [1.0]),
    )
    @test_throws ArgumentError evaluate_structural_causal_model(
        graph,
        Dict("X" => root),
        Dict("X" => Float64[]),
    )
    @test_throws ArgumentError evaluate_structural_causal_model(
        graph,
        Dict("X" => root),
        Dict("X" => reshape([1.0], 1, 1)),
    )
    @test_throws ArgumentError evaluate_structural_causal_model(
        graph,
        Dict("X" => root),
        Dict("X" => [true]),
    )
    @test_throws ArgumentError evaluate_structural_causal_model(
        graph,
        Dict("X" => root),
        Dict("X" => [Inf]),
    )

    for invalid_mechanism in (
        (parents, noise) -> 1.0,
        (parents, noise) -> reshape([1.0], 1, 1),
        (parents, noise) -> [1.0, 2.0],
        (parents, noise) -> [NaN],
        (parents, noise) -> ComplexF64[1.0+1.0im],
        (parents, noise) -> [true],
    )
        @test_throws ArgumentError evaluate_structural_causal_model(
            graph,
            Dict("X" => invalid_mechanism),
            Dict("X" => [0.0]),
        )
    end

    two_node_graph = CausalDAG(("X", "Y"), (("X", "Y"),), ("X", "Y"))
    called = String[]
    mechanisms = Dict{String,Any}(
        "X" => ((parents, noise) -> (push!(called, "X"); noise)),
        "Y" => ((parents, noise) -> (push!(called, "Y"); noise)),
    )
    @test_throws ArgumentError evaluate_structural_causal_model(
        two_node_graph,
        mechanisms,
        Dict("X" => [1.0], "Y" => [1.0, 2.0]),
    )
    @test isempty(called)
end
