"""Canonical node order and sample-by-node structural values."""
struct StructuralCausalModelResult
    node_order::Tuple{Vararg{String}}
    values::Matrix{Float64}

    function StructuralCausalModelResult(
        node_order::Tuple{Vararg{String}},
        values::Matrix{Float64},
        ::Val{:validated},
    )
        return new(node_order, copy(values))
    end
end

function StructuralCausalModelResult(node_order, values)
    nodes = try
        Tuple(_node(node, "node_order member") for node in node_order)
    catch error
        error isa ArgumentError && rethrow()
        throw(ArgumentError("node_order must be an iterable collection of node names"))
    end
    isempty(nodes) && throw(ArgumentError("node_order must be nonempty"))
    length(unique(nodes)) == length(nodes) ||
        throw(ArgumentError("node_order must not contain duplicate nodes"))
    matrix = _scm_real_matrix(values, "values")
    size(matrix, 1) > 0 || throw(ArgumentError("values must contain at least one sample"))
    size(matrix, 2) == length(nodes) ||
        throw(ArgumentError("values columns must match node_order"))
    return StructuralCausalModelResult(nodes, matrix, Val(:validated))
end

function _scm_real_vector(values, subject::AbstractString; expected_length = nothing)
    values isa AbstractVector || throw(ArgumentError("$subject must be a numeric vector"))
    all(value -> value isa Real && !(value isa Bool), values) ||
        throw(ArgumentError("$subject must be real-valued and non-boolean"))
    vector = try
        Vector{Float64}(values)
    catch
        throw(ArgumentError("$subject must be a numeric vector"))
    end
    expected_length !== nothing &&
        length(vector) != expected_length &&
        throw(ArgumentError("$subject must have length $expected_length"))
    all(isfinite, vector) ||
        throw(ArgumentError("$subject must contain only finite values"))
    return copy(vector)
end

function _scm_real_matrix(values, subject::AbstractString)
    values isa AbstractMatrix || throw(ArgumentError("$subject must be a numeric matrix"))
    all(value -> value isa Real && !(value isa Bool), values) ||
        throw(ArgumentError("$subject must be real-valued and non-boolean"))
    matrix = try
        Matrix{Float64}(values)
    catch
        throw(ArgumentError("$subject must be a numeric matrix"))
    end
    all(isfinite, matrix) ||
        throw(ArgumentError("$subject must contain only finite values"))
    return copy(matrix)
end

function _scm_key(key, subject::AbstractString)
    key isa Symbol && return String(key)
    key isa AbstractString && return String(key)
    throw(ArgumentError("$subject keys must be strings or symbols"))
end

function _scm_mapping(mapping, subject::AbstractString, graph_nodes::Set{String})
    entries = if mapping isa NamedTuple
        [(String(key), getproperty(mapping, key)) for key in keys(mapping)]
    elseif mapping isa AbstractDict
        [(_scm_key(key, subject), value) for (key, value) in pairs(mapping)]
    else
        throw(ArgumentError("$subject must be an AbstractDict or NamedTuple"))
    end

    normalized = Dict{String,Any}()
    for (key, value) in entries
        haskey(normalized, key) &&
            throw(ArgumentError("$subject must not contain duplicate normalized keys"))
        normalized[key] = value
    end
    Set(keys(normalized)) == graph_nodes ||
        throw(ArgumentError("$subject must contain exactly one entry per graph node"))
    return normalized
end

function _scm_parent_map(dag::CausalDAG)
    parents = Dict(node => String[] for node in dag.nodes)
    for (source, target) in dag.directed_edges
        push!(parents[target], source)
    end
    foreach(sort!, values(parents))
    return parents
end

function _scm_lexical_topological_order(dag::CausalDAG, parents)
    children = Dict(node => String[] for node in dag.nodes)
    indegree = Dict(node => length(parents[node]) for node in dag.nodes)
    for (source, target) in dag.directed_edges
        push!(children[source], target)
    end
    foreach(sort!, values(children))

    ready = sort([node for node in dag.nodes if indegree[node] == 0])
    ordered = String[]
    while !isempty(ready)
        node = popfirst!(ready)
        push!(ordered, node)
        for child in children[node]
            indegree[child] -= 1
            if indegree[child] == 0
                push!(ready, child)
                sort!(ready)
            end
        end
    end
    length(ordered) == length(dag.nodes) || throw(ArgumentError("dag must be acyclic"))
    return Tuple(ordered)
end

"""
    evaluate_structural_causal_model(dag, mechanisms, exogenous_inputs)

Evaluate one deterministic caller-supplied mechanism per graph node. Nodes
follow lexical tie-broken topological order; a mechanism's parent-matrix
columns follow lexical parent order. Each mechanism receives independent
`n_samples × n_parents` and `n_samples` snapshots and must return one finite
real vector of length `n_samples`.
"""
function evaluate_structural_causal_model(dag, mechanisms, exogenous_inputs)
    dag isa CausalDAG || throw(ArgumentError("dag must be a CausalDAG"))
    graph_nodes = Set(dag.nodes)
    mechanism_snapshot = _scm_mapping(mechanisms, "mechanisms", graph_nodes)
    exogenous_mapping = _scm_mapping(exogenous_inputs, "exogenous_inputs", graph_nodes)

    exogenous_snapshot = Dict{String,Vector{Float64}}()
    n_samples = nothing
    for node in dag.nodes
        vector = _scm_real_vector(
            exogenous_mapping[node],
            "exogenous input for node $(repr(node))",
        )
        if n_samples === nothing
            n_samples = length(vector)
            n_samples > 0 ||
                throw(ArgumentError("exogenous input vectors must contain a sample"))
        elseif length(vector) != n_samples
            throw(ArgumentError("all exogenous input vectors must have equal length"))
        end
        exogenous_snapshot[node] = vector
    end
    n_samples === nothing && throw(ArgumentError("dag must contain at least one node"))

    probe_parents = Matrix{Float64}(undef, n_samples, 0)
    for node in dag.nodes
        applicable(mechanism_snapshot[node], probe_parents, exogenous_snapshot[node]) ||
            throw(ArgumentError("mechanism for node $(repr(node)) must be callable"))
    end

    parents = _scm_parent_map(dag)
    node_order = _scm_lexical_topological_order(dag, parents)
    evaluated = Dict{String,Vector{Float64}}()
    for node in node_order
        parent_nodes = parents[node]
        parent_matrix = Matrix{Float64}(undef, n_samples, length(parent_nodes))
        for (column, parent) in enumerate(parent_nodes)
            parent_matrix[:, column] = evaluated[parent]
        end
        raw_result =
            mechanism_snapshot[node](copy(parent_matrix), copy(exogenous_snapshot[node]))
        evaluated[node] = _scm_real_vector(
            raw_result,
            "mechanism result for node $(repr(node))";
            expected_length = n_samples,
        )
    end

    values_matrix = Matrix{Float64}(undef, n_samples, length(node_order))
    for (column, node) in enumerate(node_order)
        values_matrix[:, column] = evaluated[node]
    end
    return StructuralCausalModelResult(node_order, values_matrix, Val(:validated))
end
