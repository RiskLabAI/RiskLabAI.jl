using Test
using TOML
using Random
using LinearAlgebra
using Zygote
using RiskLabAI
using RiskLabAI.Pde

function pde_fixture_array(values)
    dimensions = Int[]
    item = values
    while item isa AbstractVector
        push!(dimensions, length(item))
        item = first(item)
    end
    flatten(value::Number) = [Float64(value)]
    flatten(value::AbstractVector) = reduce(vcat, flatten.(value))
    array = reshape(flatten(values), reverse(dimensions)...)
    return permutedims(array, reverse(Tuple(1:length(dimensions))))
end

pde_row_major(array) = vec(permutedims(array, reverse(Tuple(1:ndims(array)))))

function pde_shared_values(shape, shift, scale, offset)
    indices = 0:prod(shape)-1
    integers = mod.(indices .* 37 .+ shift * 101, 9973) .- 4986
    flat = offset .+ scale .* Float64.(integers) ./ 4986
    return permutedims(reshape(flat, reverse(Tuple(shape))...), reverse(Tuple(1:length(shape))))
end

function pde_fixture_tree(template::NamedTuple, specifications, path = ())
    return NamedTuple{keys(template)}(Tuple(pde_fixture_tree(getproperty(template, name), specifications, (path..., string(name))) for name in keys(template)))
end

function pde_fixture_tree(template::Tuple, specifications, path = ())
    return Tuple(pde_fixture_tree(value, specifications, (path..., string(index))) for (index, value) in enumerate(template))
end

function pde_fixture_tree(template::AbstractArray, specifications, path = ())
    specification = specifications[join(path, "/")]
    @test Tuple(specification["shape"]) == size(template)
    return pde_shared_values(specification["shape"], specification["shift"], specification["scale"], specification["offset"])
end

function pde_get_path(value, path)
    for name in path
        isnothing(value) && return nothing
        value = value isa NamedTuple ? getproperty(value, Symbol(name)) : value[parse(Int, name)]
    end
    return value
end

function pde_fixture_model(kind, settings; parameters = nothing)
    kind == "Net1" && return Net1(settings["input_dim"], settings["output_dim"]; parameters)
    kind == "TimeNet" && return TimeNet(settings["output_dim"]; parameters)
    kind == "FBSNNNetwork" && return FBSNNNetwork(settings["layer_sizes"]; parameters)
    kind == "DeepBSDE" && return DeepBSDE(settings["layer_sizes"]; parameters)
    kind == "TimeNetForSet" && return TimeNetForSet(settings["input_dim"], settings["output_dim"]; parameters)
    kind == "DeepTimeSetTransformer" && return DeepTimeSetTransformer(settings["input_dim"]; parameters)
    kind == "TimeDependentNetwork" && return TimeDependentNetwork(settings["input_dim"], settings["hidden_sizes"], settings["output_dim"]; parameters)
    kind == "TimeDependentNetworkMonteCarlo" && return TimeDependentNetworkMonteCarlo(settings["input_dim"], settings["hidden_sizes"], settings["output_dim"], settings["sigma"]; parameters)
    kind == "MAB" && return MAB(settings["dim_q"], settings["dim_k"], settings["dim_v"], settings["num_heads"]; ln = settings["ln"], parameters)
    kind == "SAB" && return SAB(settings["dim_in"], settings["dim_out"], settings["num_heads"]; ln = settings["ln"], parameters)
    kind == "ISAB" && return ISAB(settings["dim_in"], settings["dim_out"], settings["num_heads"], settings["num_inds"]; ln = settings["ln"], parameters)
    kind == "PMA" && return PMA(settings["dim"], settings["num_heads"], settings["num_seeds"]; ln = settings["ln"], parameters)
    error("Unknown shared network fixture")
end

function pde_parameter_forward(model, parameters, inputs...)
    P = RiskLabAI.Pde
    model isa Net1 && return P._pde_dense(parameters.layer, only(inputs))
    model isa TimeNet && return P._pde_time_forward(parameters, only(inputs))
    model isa FBSNNNetwork && return P._pde_hidden_forward(parameters, only(inputs), sin)
    model isa DeepBSDE && return P._pde_hidden_forward(parameters, only(inputs), P._pde_relu)
    model isa TimeNetForSet && return P._pde_time_set_forward(parameters, inputs...)
    model isa DeepTimeSetTransformer && return P._pde_deep_set_forward(parameters, inputs...)
    model isa TimeDependentNetwork && return P._pde_dtnn_forward(model, parameters, inputs...; update_state = false)
    model isa TimeDependentNetworkMonteCarlo && return P._pde_monte_carlo_forward(model, parameters, inputs...)
    model isa MAB && return P._pde_attention(parameters, inputs..., model.num_heads, model.ln)
    model isa SAB && return P._pde_attention(parameters.mab, only(inputs), only(inputs), model.num_heads, model.ln)
    model isa ISAB && return P._pde_isab_forward(parameters, only(inputs), model.num_heads, model.ln)
    model isa PMA && return P._pde_pma_forward(parameters, only(inputs), model.num_heads, model.ln)
    error("Unknown native network type")
end

@testset "Shared PDE dense, attention and time-network values and gradients" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "pde_networks_python_320.toml"))
    for case in fixture["networks"]
        template = pde_fixture_model(case["kind"], case["settings"])
        specifications = Dict(join(value["native_path"], "/") => value for value in values(case["parameters"]))
        @test length(specifications) == length(case["parameters"])
        parameters = pde_fixture_tree(template.parameters, specifications)
        model = pde_fixture_model(case["kind"], case["settings"]; parameters)
        inputs = Tuple(pde_fixture_array(value) for value in case["inputs"])
        original = deepcopy((inputs, parameters))
        expected = pde_fixture_array(case["output"])
        output = model(inputs...)
        @test size(output) == size(expected)
        @test isapprox(output, expected; rtol = 1e-10, atol = 1e-12)
        weights = pde_fixture_array(case["objective_weights"])
        objective = supplied -> sum(pde_parameter_forward(model, supplied, inputs...) .* weights)
        gradients = only(Zygote.gradient(objective, model.parameters))
        for (name, encoded) in case["parameters"]
            gradient = pde_get_path(gradients, encoded["native_path"])
            gradient = isnothing(gradient) ? zeros(Tuple(encoded["shape"])) : gradient
            independent = case["parameter_gradients"][name]
            probe = pde_shared_values(encoded["shape"], encoded["shift"] + 71, 1.0, 0.0)
            probe ./= norm(probe)
            @test isapprox(sum(gradient .* probe), independent["projection"]; rtol = 1e-10, atol = 1e-12)
            @test isapprox(norm(gradient), independent["norm"]; rtol = 1e-10, atol = 1e-12)
            @test isapprox(pde_row_major(gradient)[Int.(independent["indices"])], independent["values"]; rtol = 1e-10, atol = 1e-12)
        end
        input_gradients = Zygote.gradient((values...) -> sum(pde_parameter_forward(model, model.parameters, values...) .* weights), inputs...)
        for (gradient, encoded) in zip(input_gradients, case["input_gradients"])
            reference = pde_fixture_array(encoded)
            actual = isnothing(gradient) ? zeros(size(reference)) : gradient
            @test isapprox(actual, reference; rtol = 1e-10, atol = 1e-12)
        end
        @test (inputs, parameters) == original
        if model isa TimeDependentNetwork
            for (actual, encoded) in zip(model.state, case["normalization_state"])
                @test isapprox(actual.mean, encoded["mean"]; rtol = 1e-10, atol = 1e-12)
                @test isapprox(actual.variance, encoded["variance"]; rtol = 1e-10, atol = 1e-12)
                @test actual.batches == encoded["batches"]
            end
            model.training = false
            before = deepcopy(model.state)
            evaluation = Tuple(pde_fixture_array(value) for value in case["evaluation_inputs"])
            @test isapprox(model(evaluation...), pde_fixture_array(case["evaluation_output"]); rtol = 1e-10, atol = 1e-12)
            @test model.state == before
        end
        if model isa Union{SAB, ISAB}
            permuted = inputs[1][:, end:-1:1, :]
            @test isapprox(model(permuted), output[:, end:-1:1, :]; rtol = 1e-10, atol = 1e-12)
        elseif model isa PMA
            @test isapprox(model(inputs[1][:, end:-1:1, :]), output; rtol = 1e-10, atol = 1e-12)
        elseif model isa MAB
            @test isapprox(model(inputs[1], inputs[2][:, end:-1:1, :]), output; rtol = 1e-10, atol = 1e-12)
            @test isapprox(model(inputs[1][:, end:-1:1, :], inputs[2]), output[:, end:-1:1, :]; rtol = 1e-10, atol = 1e-12)
        elseif model isa DeepTimeSetTransformer
            @test isapprox(model(inputs[1], inputs[2][:, end:-1:1, :]), output; rtol = 1e-10, atol = 1e-12)
        end
    end
end

@testset "PDE dense initialization, normalization modes and argument domains" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "pde_networks_python_320.toml"))
    for case in fixture["initializers"]
        model = Net1(case["input_dim"], case["output_dim"])
        draws = pde_fixture_array(case["normal_draws"])
        @test isnothing(initialize_weights(model.parameters.layer; normal_draws = draws))
        @test model.parameters.layer.weight == pde_fixture_array(case["weight"])
        @test model.parameters.layer.bias == case["bias"]
        @test draws == pde_fixture_array(case["normal_draws"])
    end
    model = Net1(2, 3)
    original = deepcopy(model.parameters)
    @test isnothing(initialize_weights(model.parameters))
    @test model.parameters == original
    Random.seed!(613)
    expected = rand(5)
    Random.seed!(613)
    initialize_weights(model.parameters.layer; seed = 25)
    @test rand(5) == expected
    second = Net1(2, 3; seed = 7)
    initialize_weights(second.parameters.layer; seed = 25)
    @test model.parameters == second.parameters
    @test_throws ArgumentError initialize_weights((weight = ones(3, 2),))
    @test_throws ArgumentError initialize_weights(model.parameters.layer; normal_draws = ones(2, 3))
    @test_throws ArgumentError Net1(0, 2)
    @test_throws ArgumentError FBSNNNetwork([2])
    @test_throws ArgumentError DeepBSDE([2, -1, 1])
    @test_throws ArgumentError TimeNet(true)
    @test_throws ArgumentError MAB(2, 2, 3, 2)
    @test_throws ArgumentError SAB(2, 4, 2; ln = 1)
    @test_throws ArgumentError ISAB(2, 4, 2, 0)
    @test_throws ArgumentError PMA(4, 2, 0)
    @test_throws ArgumentError TimeDependentNetwork(2, [], 2)
    @test_throws ArgumentError TimeDependentNetworkMonteCarlo(2, [3], 3, 0.4)
    @test_throws ArgumentError Net1(2, 2; parameters = (;))
    @test_throws ArgumentError model(ones(Float32, 3, 2))
    @test_throws ArgumentError model(fill(NaN, 3, 2))
    @test_throws ArgumentError MAB(2, 2, 4, 2)(ones(2, 3, 2), ones(3, 3, 2))
    @test_throws ArgumentError TimeNetForSet(2, 3)(ones(2, 1), ones(3, 2))
    dependent = TimeDependentNetwork(2, [3], 2; seed = 81)
    @test_throws ArgumentError dependent(ones(1, 1), ones(1, 2))
    @test all(state -> state.batches == 0, dependent.state)
    dependent(reshape([0.1, 0.2, 0.3], :, 1), [0.1 0.2; 0.4 0.3; 0.8 0.2])
    @test all(state -> state.batches == 1, dependent.state)
    dependent.training = false
    state = deepcopy(dependent.state)
    @test all(isfinite, dependent(ones(1, 1), ones(1, 2)))
    @test dependent.state == state
    Random.seed!(614)
    expected = rand(5)
    Random.seed!(614)
    SAB(2, 4, 2; seed = 12)
    @test rand(5) == expected
    @test SAB(2, 4, 2; seed = 12).parameters == SAB(2, 4, 2; seed = 12).parameters
end
