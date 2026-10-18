using Test
using TOML
using Random
using LinearAlgebra
using Zygote
using RiskLabAI
using RiskLabAI.Pde

function pdes_array(values)
    dimensions = Int[]
    current = values
    while current isa AbstractVector
        push!(dimensions, length(current))
        current = first(current)
    end
    flatten(value::Number) = [Float64(value)]
    flatten(value::AbstractVector) = reduce(vcat, flatten.(value))
    return permutedims(reshape(flatten(values), reverse(dimensions)...), reverse(Tuple(1:length(dimensions))))
end

function pdes_pattern(shape, shift, scale = 1.0, offset = 0.0)
    integers = mod.((0:prod(shape)-1) .* 37 .+ shift * 101, 9973) .- 4986
    values = offset .+ scale .* Float64.(integers) ./ 4986
    return permutedims(reshape(values, reverse(Tuple(shape))...), reverse(Tuple(1:length(shape))))
end

function pdes_parameters(template::NamedTuple, specifications, path = ())
    return NamedTuple{keys(template)}(Tuple(pdes_parameters(getproperty(template, key), specifications, (path..., string(key))) for key in keys(template)))
end

pdes_parameters(template::Tuple, specifications, path = ()) = Tuple(pdes_parameters(value, specifications, (path..., string(index))) for (index, value) in enumerate(template))

function pdes_parameters(template::AbstractArray, specifications, path = ())
    specification = specifications[join(path, "/")]
    @test size(template) == Tuple(specification["shape"])
    return pdes_pattern(specification["shape"], specification["shift"], specification["scale"], specification["offset"])
end

function pdes_path(value, path)
    for key in path
        isnothing(value) && return nothing
        value = value isa NamedTuple ? getproperty(value, Symbol(key)) : value[parse(Int, key)]
    end
    return value
end

function pdes_check_summary(value, specification, expected)
    value = isnothing(value) ? zeros(Tuple(specification["shape"])) : value
    probe = pdes_pattern(specification["shape"], specification["shift"] + 71)
    probe ./= norm(probe)
    flattened = vec(permutedims(value, reverse(Tuple(1:ndims(value)))))
    @test isapprox(norm(value), expected["norm"]; rtol = 1e-10, atol = 1e-12)
    @test isapprox(sum(value .* probe), expected["projection"]; rtol = 1e-10, atol = 1e-12)
    @test isapprox(flattened[Int.(expected["indices"])], expected["values"]; rtol = 1e-10, atol = 1e-12)
end

function pdes_model(case; parameters = nothing)
    config = case["configuration"]
    equation = getproperty(RiskLabAI.Pde, Symbol(case["kind"]))(config["dim"], config["total_time"], config["num_time_interval"])
    return case["method"] == "FBSNN" ? FBSNNSolver(equation, case["layer_sizes"], case["learning_rate"]; parameters) : FBSDESolver(equation, case["layer_sizes"], case["learning_rate"], case["method"]; parameters)
end

function pdes_batch(encoded)
    return (paths = pdes_array(encoded["paths"]), increments = pdes_array(encoded["increments"]), monte_carlo_gradients = haskey(encoded, "monte_carlo_gradients") ? pdes_array(encoded["monte_carlo_gradients"]) : nothing)
end

function pdes_loss(model, parameters, paths, increments, initial, gradient, monte_carlo)
    time = ones(size(paths, 1), 1)
    return model isa FBSNNSolver ? RiskLabAI.Pde._pde_fbsnn_loss(model, parameters, paths, increments, time) : first(RiskLabAI.Pde._pde_bsde_loss(model, parameters, paths, increments, time, initial, gradient, monte_carlo))
end

@testset "Shared PDE solver objectives, path derivatives and optimizer replay" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "pde_solvers_python_320.toml"))
    for case in fixture["solvers"]
        @testset "$(case["method"]) $(case["kind"])" begin
            template = pdes_model(case)
            specifications = Dict(join(encoded["native_path"], "/") => encoded for encoded in values(case["parameters"]))
            parameters = pdes_parameters(template.parameters, specifications)
            model = pdes_model(case; parameters)
            batch = pdes_batch(case["batch"])
            initial, gradient = [case["initial_value"]], pdes_array(case["initial_gradient"])
            preserved = deepcopy((parameters, batch, initial, gradient))
            time = ones(size(batch.paths, 1), 1)
            if model isa FBSNNSolver
                value = compute_pde_loss(model, batch.paths, batch.increments, time, initial)
            else
                value, components... = compute_pde_loss(model, batch.paths, batch.increments, time, initial, gradient; monte_carlo_gradients = batch.monte_carlo_gradients)
                for (actual, encoded) in zip(components, case["components"])
                    @test isapprox(actual, pdes_array(encoded); rtol = 1e-10, atol = 1e-12)
                end
            end
            @test isapprox(value, case["loss"]; rtol = 1e-10, atol = 1e-12)
            derivatives = only(Zygote.gradient(p -> pdes_loss(model, p, batch.paths, batch.increments, initial, gradient, batch.monte_carlo_gradients), model.parameters))
            for (name, encoded) in case["parameters"]
                pdes_check_summary(pdes_path(derivatives, encoded["native_path"]), encoded, case["parameter_gradients"][name])
            end
            input_derivatives = Zygote.gradient((x, dw, y, z) -> pdes_loss(model, model.parameters, x, dw, y, z, batch.monte_carlo_gradients), batch.paths, batch.increments, initial, gradient)
            for (actual, encoded) in zip(input_derivatives, case["argument_gradients"])
                expected = pdes_array(encoded)
                actual = isnothing(actual) ? zeros(size(expected)) : actual
                @test isapprox(actual, expected; rtol = 1e-10, atol = 1e-12)
            end
            @test (parameters, batch, initial, gradient) == preserved
            trained = pdes_model(case; parameters)
            batches = pdes_batch.(case["training_batches"])
            validation = pdes_batch(case["validation_batch"])
            original_batches = deepcopy((batches, validation))
            losses, estimates = solve_pde(trained; iterations = length(batches), batch_size = 4, init_y = case["initial_value"], training_batches = batches, validation_batch = validation)
            @test isapprox(losses, [row["validation_loss"] for row in case["history"]]; rtol = 1e-10, atol = 1e-12)
            @test isapprox(estimates, [row["estimate"] for row in case["history"]]; rtol = 1e-10, atol = 1e-12)
            for (name, encoded) in case["parameters"]
                pdes_check_summary(pdes_path(trained.parameters, encoded["native_path"]), encoded, case["trained_parameters"][name])
            end
            if case["method"] == "DTNN"
                for (actual, encoded) in zip(trained.solver.state, case["normalization_state"])
                    @test isapprox(actual.mean, encoded["mean"]; rtol = 1e-10, atol = 1e-12)
                    @test isapprox(actual.variance, encoded["variance"]; rtol = 1e-10, atol = 1e-12)
                    @test actual.batches == encoded["batches"]
                end
            end
            @test (batches, validation) == original_batches
            @test parameters == preserved[1]
        end
    end
end

@testset "PDE solver native derivatives, input boundaries and continuation" begin
    equation = HJBLQ(1, 0.2, 2)
    @test_throws ArgumentError FBSDESolver(equation, [1, 2, 1], 0.01, "unknown")
    @test_throws ArgumentError FBSDESolver(equation, [2, 2, 1], 0.01, "DeepBSDE")
    @test_throws ArgumentError FBSDESolver(equation, [2], 0., "DTNN")
    @test_throws ArgumentError FBSDESolver(HJBLQ(2, 0.2, 2), [2], 0.01, "Deep-Time-SetTransformer")
    @test_throws ArgumentError FBSNNSolver(equation, [1, 2, 1], 0.01)
    @test_throws ArgumentError FBSNNSolver(equation, [2, 2, 2], 0.01)
    @test_throws ArgumentError FBSNNSolver(equation, [2, 2, 1], -0.01)
    monte_carlo = FBSDESolver(equation, [2], 0.01, "Monte-Carlo")
    @test_throws ArgumentError solve_pde(monte_carlo)
    increments, paths = pde_sample(equation, 4; rng = MersenneTwister(812))
    batch = (paths = paths, increments = increments)
    @test_throws ArgumentError compute_pde_loss(monte_carlo, paths, increments, ones(4, 1), [0.], zeros(1, 1))
    sine = FBSNNSolver(equation, [2, 3, 2, 1], 0.001; seed = 832)
    inputs = [0.1 0.2; 0.2 -0.4; 0.3 0.1]
    output, derivative = RiskLabAI.Pde._pde_sine_value_gradient(sine.parameters.layers, inputs)
    @test output ≈ sine.solver(inputs)
    @test derivative ≈ only(Zygote.gradient(values -> sum(sine.solver(values)), inputs))
    original = deepcopy(sine.parameters)
    @test_throws ArgumentError solve_pde(sine; iterations = 2, batch_size = 4, training_batches = [batch], validation_batch = batch)
    @test_throws ArgumentError solve_pde(sine; iterations = 1, batch_size = 5, training_batches = [batch], validation_batch = batch)
    @test sine.parameters == original
    continuous = FBSNNSolver(equation, [2, 3, 2, 1], 0.001; seed = 832)
    whole = solve_pde(continuous; iterations = 2, batch_size = 4, training_batches = [batch, batch], validation_batch = batch)
    first_result = solve_pde(sine; iterations = 1, batch_size = 4, init_y = -35., training_batches = [batch], validation_batch = batch)
    second_result = solve_pde(sine; iterations = 1, batch_size = 4, init_y = 95., training_batches = [batch], validation_batch = batch)
    @test vcat(first_result[1], second_result[1]) == whole[1]
    @test vcat(first_result[2], second_result[2]) == whole[2]
    @test sine.parameters == continuous.parameters
    single = FBSDESolver(HJBLQ(1, 0.1, 1), [1, 2, 1], 0.001, "DeepBSDE")
    @test isempty(single.parameters)
    @test all(isfinite, first(solve_pde(single; iterations = 2, batch_size = 4, seed = 381)))
    Random.seed!(749)
    expected = rand(5)
    Random.seed!(749)
    native = FBSNNSolver(equation, [2, 3, 1], 0.001; seed = 64)
    result = solve_pde(native; iterations = 1, batch_size = 4, seed = 65)
    @test rand(5) == expected
    repeated = FBSNNSolver(equation, [2, 3, 1], 0.001; seed = 64)
    @test solve_pde(repeated; iterations = 1, batch_size = 4, seed = 65) == result
    wrapper_rng = MersenneTwister(614)
    direct = FBSDESolver(equation, [1, 3, 1], 0.003, "DeepBSDE"; seed = rand(wrapper_rng, UInt32))
    expected_wrapper = solve_pde(direct; iterations = 2, batch_size = 4, init_y = 0.2, rng = wrapper_rng)
    @test solve_deep_bsde(equation; hidden_sizes = [3], iterations = 2, batch_size = 4, init_y = 0.2, learning_rate = 0.003, rng = MersenneTwister(614)) == expected_wrapper
end
