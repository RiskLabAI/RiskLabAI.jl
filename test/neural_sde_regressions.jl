using Test
using TOML
using Random
using LinearAlgebra
using Zygote
using RiskLabAI
using RiskLabAI.Pde: normalized_call_grid_constraints, normalized_call_grid_witness,
    affine_price_state_interval, IntervalNeuralSDE, PolytopeNeuralSDE,
    negative_log_likelihood, fit_interval_neural_sde, fit_polytope_neural_sde,
    simulate_interval_neural_sde, simulate_polytope_neural_sde, hjm_drift_residual

nse_fixture_matrix(rows) = Float64.(permutedims(hcat(rows...)))
nse_fixture_array(windows) = cat((reshape(nse_fixture_matrix(window), 1, length(window), length(first(window))) for window in windows)...; dims = 1)

function nse_fixture_parameters(encoded)
    return (weight1 = nse_fixture_matrix(encoded["network.0.weight"]), bias1 = Float64.(encoded["network.0.bias"]),
        weight2 = nse_fixture_matrix(encoded["network.2.weight"]), bias2 = Float64.(encoded["network.2.bias"]))
end

function nse_test_parameters(actual, expected)
    @test keys(actual) == keys(expected)
    for name in keys(expected)
        @test isapprox(getproperty(actual, name), getproperty(expected, name); rtol = 1e-10, atol = 1e-12)
    end
end

@testset "Normalized call-grid inequalities and probability witnesses" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "neural_sde_python_320.toml"))
    for case in fixture["grids"]
        strikes, maturities, prices = Float64.(case["strikes"]), Float64.(case["maturities"]), nse_fixture_matrix(case["prices"])
        matrix, bounds = normalized_call_grid_constraints(strikes, maturities)
        @test isapprox(matrix, nse_fixture_matrix(case["matrix"]); rtol = 1e-10, atol = 1e-12)
        @test isapprox(bounds, case["bounds"]; rtol = 1e-10, atol = 1e-12)
        @test minimum(matrix * vec(permutedims(prices)) - bounds) >= -1e-12
        witness = normalized_call_grid_witness(strikes, maturities, prices)
        @test isapprox(witness.support, case["support"]; rtol = 1e-10, atol = 1e-12)
        @test isapprox(witness.probabilities, nse_fixture_matrix(case["probabilities"]); rtol = 1e-10, atol = 1e-12)
        @test witness.maximum_mean_error <= 1e-10
        @test witness.maximum_price_error <= 1e-10
        @test all(>=(0), witness.probabilities)
        @test isapprox(vec(sum(witness.probabilities; dims = 2)), ones(length(maturities)); rtol = 1e-10, atol = 1e-12)
        @test isapprox(witness.probabilities * witness.support, ones(length(maturities)); rtol = 1e-10, atol = 1e-12)
        @test isapprox(witness.probabilities * max.(witness.support .- strikes', 0), prices; rtol = 1e-10, atol = 1e-12)
        fine_strikes = collect(range(0, witness.support[end]; length = 301))
        calls = witness.probabilities * max.(witness.support .- fine_strikes', 0)
        @test all(diff(calls; dims = 1) .>= -1e-12)
    end
    matrix, bounds = normalized_call_grid_constraints([0.5, 1.5], [1.0])
    @test minimum(matrix * [1.0, 1.0] - bounds) >= -1e-12
    @test_throws ArgumentError normalized_call_grid_witness([0.5, 1.5], [1.0], [1.0 1.0])
    zero_tail = normalized_call_grid_witness([1.0, 2.0], [1.0], [0.0 0.0])
    @test zero_tail.probabilities == [0.0 1.0 0.0 0.0]
    for (strikes, maturities) in (([1.0], [1.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.5], [1.0]), ([1.0, 1.0], [1.0]), ([0.5, 1.0], [0.0]), ([0.5, 1.0], [1.0, 1.0]), ([0.5, NaN], [1.0]), (Float64[], [1.0]))
        @test_throws ArgumentError normalized_call_grid_constraints(strikes, maturities)
    end
    @test_throws ArgumentError normalized_call_grid_witness([0.5, 1.5], [1.0], [0.2 0.4])
    @test_throws ArgumentError normalized_call_grid_witness([0.5, 1.5], [1.0], ones(2, 2))
end

@testset "Supplied affine state bounds and HJM residuals" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "neural_sde_python_320.toml"))
    for case in fixture["affine"]
        offset, loading = Float64.(case["offset"]), Float64.(case["loading"])
        matrix, bounds = nse_fixture_matrix(case["matrix"]), Float64.(case["bounds"])
        interval = affine_price_state_interval(offset, loading, matrix, bounds)
        @test isapprox(collect(interval), [case["lower"], case["upper"]]; rtol = 1e-10, atol = 1e-12)
        @test all(matrix * (offset + interval[1] * loading) .>= bounds .- 1e-12)
        @test all(matrix * (offset + interval[2] * loading) .>= bounds .- 1e-12)
        @test any(matrix * (offset + (interval[1] - 1e-5) * loading) .< bounds)
        @test any(matrix * (offset + (interval[2] + 1e-5) * loading) .< bounds)
    end
    for case in fixture["hjm"]
        basis, diffusion = nse_fixture_matrix(case["price_basis"]), nse_fixture_matrix(case["diffusion"])
        drift, required = Float64.(case["drift"]), Float64.(case["required_price_drift"])
        actual = hjm_drift_residual(basis, drift, diffusion, required)
        for name in (:market_price_of_risk, :residual, :squared_residual, :singular_values)
            @test isapprox(getproperty(actual, name), case["expected"][string(name)]; rtol = 1e-10, atol = 1e-12)
        end
        @test actual.rank == case["expected"]["rank"]
        @test norm((basis * diffusion)' * actual.residual) <= 1e-10
        @test isapprox(actual.residual, basis * drift - required - basis * diffusion * actual.market_price_of_risk; rtol = 1e-10, atol = 1e-12)
    end
    @test_throws ArgumentError affine_price_state_interval([0.0], [0.0], ones(1, 1), [1.0])
    @test_throws ArgumentError affine_price_state_interval([0.0], [1.0], ones(1, 1), [0.0])
    @test_throws ArgumentError affine_price_state_interval([0.0], [1.0], reshape([1.0, -1.0], 2, 1), [0.0, 0.0])
    @test_throws ArgumentError affine_price_state_interval([0.0], [1.0], reshape([1.0, -1.0], 2, 1), [1.0, 0.0])
    @test_throws ArgumentError affine_price_state_interval([0.0], [1.0, 2.0], ones(1, 1), [0.0])
    @test_throws ArgumentError hjm_drift_residual(ones(2, 2), ones(3), ones(2, 2), ones(2))
    @test_throws ArgumentError hjm_drift_residual(ones(2, 2), ones(2), fill(NaN, 2, 2), ones(2))
end

@testset "Scalar neural-SDE boundaries, likelihood and Adam parity" begin
    P = RiskLabAI.Pde
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "neural_sde_python_320.toml"))
    for case in fixture["intervals"]
        parameters = nse_fixture_parameters(case["parameters"])
        model = IntervalNeuralSDE(case["lower"], case["upper"]; drift_bound = case["drift_bound"], diffusion_scale = case["diffusion_scale"], hidden_size = case["hidden_size"], parameters)
        states, following, boundary = Float64.(case["states"]), Float64.(case["next_states"]), Float64.(case["boundary_states"])
        drift, diffusion = model(boundary)
        @test isapprox(drift, case["drift"]; rtol = 1e-10, atol = 1e-12)
        @test isapprox(diffusion, case["diffusion"]; rtol = 1e-10, atol = 1e-12)
        @test diffusion[1] == diffusion[end] == 0
        @test drift[1] >= 0 && drift[end] <= 0
        @test all(>(0), diffusion[2:end-1])
        @test isapprox(negative_log_likelihood(model, states, following, case["dt"]), case["nll"]; rtol = 1e-10, atol = 1e-12)
        gradient = only(Zygote.gradient(value -> P._nse_likelihood(model, value, states, following, case["dt"]), model.parameters))
        nse_test_parameters(gradient, nse_fixture_parameters(case["gradients"]))
        state_gradient = only(Zygote.gradient(states) do values
            drift, diffusion = model(values)
            sum(drift) + 0.3sum(diffusion)
        end)
        @test isapprox(state_gradient, case["state_gradients"]; rtol = 1e-10, atol = 1e-12)
        original = deepcopy((states, following, parameters))
        history = fit_interval_neural_sde(model, states, following; dt = case["dt"], iterations = case["iterations"], learning_rate = case["learning_rate"])
        @test isapprox(history, case["history"]; rtol = 1e-10, atol = 1e-12)
        nse_test_parameters(model.parameters, nse_fixture_parameters(case["trained_parameters"]))
        @test (states, following, parameters) == original
        simulation = simulate_interval_neural_sde(model, Float64.(case["initial"]), nse_fixture_matrix(case["increments"]); dt = case["dt"])
        @test isapprox(simulation, nse_fixture_matrix(case["simulation"]); rtol = 1e-10, atol = 1e-12)
        @test all(value -> model.lower <= value <= model.upper, simulation)
        fitted = deepcopy(model.parameters)
        drift, diffusion = model(Float64.(case["evaluation_states"]))
        @test isapprox(drift, case["evaluation_drift"]; rtol = 1e-10, atol = 1e-12)
        @test isapprox(diffusion, case["evaluation_diffusion"]; rtol = 1e-10, atol = 1e-12)
        @test model.parameters == fitted
        @test_throws ArgumentError negative_log_likelihood(model, [model.lower], [states[1]], case["dt"])
        failure = try
            simulate_interval_neural_sde(model, states[1:1], fill(1e6, 1, 2); dt = case["dt"])
        catch exception
            exception
        end
        @test failure isa ErrorException
        @test occursin("Euler step 1 exits", sprint(showerror, failure))
    end
end

@testset "Fixed-coordinate polytope diffusion, likelihood and Adam parity" begin
    P = RiskLabAI.Pde
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "neural_sde_python_320.toml"))
    for case in fixture["polytopes"]
        normals, bounds, interior = nse_fixture_matrix(case["normals"]), Float64.(case["bounds"]), Float64.(case["interior"])
        parameters = nse_fixture_parameters(case["parameters"])
        settings = (drift_bound = case["drift_bound"], diffusion_scale = case["diffusion_scale"], hidden_size = case["hidden_size"])
        model = PolytopeNeuralSDE(normals, bounds, interior; settings..., parameters)
        @test isapprox(model.normals, nse_fixture_matrix(case["normalized_normals"]); rtol = 1e-10, atol = 1e-12)
        @test isapprox(model.bounds, case["normalized_bounds"]; rtol = 1e-10, atol = 1e-12)
        @test isapprox(model.rho, case["rho"]; rtol = 1e-10, atol = 1e-12)
        states, following, boundary = nse_fixture_matrix(case["states"]), nse_fixture_matrix(case["next_states"]), nse_fixture_matrix(case["boundary_states"])
        drift, diffusion = model(boundary)
        @test isapprox(drift, nse_fixture_matrix(case["drift"]); rtol = 1e-10, atol = 1e-12)
        @test isapprox(diffusion, nse_fixture_array(case["diffusion"]); rtol = 1e-10, atol = 1e-12)
        distances = boundary * model.normals' .- model.bounds'
        for index in findall(abs.(distances) .< 1e-14)
            row, face = Tuple(index)
            @test dot(model.normals[face, :], drift[row, :]) >= -1e-12
            @test norm(model.normals[face, :]' * diffusion[row, :, :]) <= 1e-12
        end
        @test isapprox(negative_log_likelihood(model, states, following, case["dt"]), case["nll"]; rtol = 1e-10, atol = 1e-12)
        gradient = only(Zygote.gradient(value -> P._nse_likelihood(model, value, states, following, case["dt"]), model.parameters))
        nse_test_parameters(gradient, nse_fixture_parameters(case["gradients"]))
        state_gradient = only(Zygote.gradient(states) do values
            drift, diffusion = model(values)
            sum(drift) + 0.3sum(diffusion)
        end)
        @test isapprox(state_gradient, nse_fixture_matrix(case["state_gradients"]); rtol = 1e-10, atol = 1e-12)
        rescaling = Float64.(1:length(bounds))
        reversed = length(bounds):-1:1
        equivalent = PolytopeNeuralSDE((normals .* rescaling)[reversed, :], (bounds .* rescaling)[reversed], interior; settings..., parameters)
        other_drift, other_diffusion = equivalent(states)
        baseline_drift, baseline_diffusion = model(states)
        @test isapprox(other_drift, baseline_drift; rtol = 1e-10, atol = 1e-12)
        @test isapprox(other_diffusion, baseline_diffusion; rtol = 1e-10, atol = 1e-12)
        original = deepcopy((states, following, parameters, normals, bounds, interior))
        history = fit_polytope_neural_sde(model, states, following; dt = case["dt"], iterations = case["iterations"], learning_rate = case["learning_rate"])
        @test isapprox(history, case["history"]; rtol = 1e-10, atol = 1e-12)
        nse_test_parameters(model.parameters, nse_fixture_parameters(case["trained_parameters"]))
        @test (states, following, parameters, normals, bounds, interior) == original
        initial = nse_fixture_matrix(case["initial"])
        simulation = simulate_polytope_neural_sde(model, initial, nse_fixture_array(case["increments"]); dt = case["dt"])
        @test isapprox(simulation, nse_fixture_array(case["simulation"]); rtol = 1e-10, atol = 1e-12)
        fitted = deepcopy(model.parameters)
        drift, diffusion = model(nse_fixture_matrix(case["evaluation_states"]))
        @test isapprox(drift, nse_fixture_matrix(case["evaluation_drift"]); rtol = 1e-10, atol = 1e-12)
        @test isapprox(diffusion, nse_fixture_array(case["evaluation_diffusion"]); rtol = 1e-10, atol = 1e-12)
        @test model.parameters == fitted
        @test_throws ArgumentError negative_log_likelihood(model, boundary[1:1, :], states[1:1, :], case["dt"])
        failure = try
            simulate_polytope_neural_sde(model, initial[1:1, :], fill(1e6, 1, 2, model.dimension); dt = case["dt"])
        catch exception
            exception
        end
        @test failure isa ErrorException
        @test occursin("Euler step 1 exits", sprint(showerror, failure))
    end
end

@testset "Neural-SDE domain checks, roundoff and local RNG" begin
    P = RiskLabAI.Pde
    settings = (drift_bound = 0.3, diffusion_scale = 0.2, hidden_size = 2)
    Random.seed!(193)
    expected = rand(5)
    Random.seed!(193)
    model = IntervalNeuralSDE(-1, 1; settings..., seed = 14)
    @test rand(5) == expected
    @test model.parameters == IntervalNeuralSDE(-1, 1; settings..., seed = 14).parameters
    @test model.parameters != IntervalNeuralSDE(-1, 1; settings..., seed = 15).parameters
    polytope = PolytopeNeuralSDE(reshape([1.0, -1.0], 2, 1), [-1.0, -1.0], [0.0]; settings...)
    @test P._nse_distances(polytope, reshape([-1.0 - 5e-13], 1, 1)) == zeros(1, 2) .+ [0.0 2.0+5e-13]
    @test polytope(reshape([-1.0 - 5e-13], 1, 1))[2][1, 1, 1] == 0
    @test polytope(reshape([-1.0 + 1e-13], 1, 1))[2][1, 1, 1] > 0
    @test_throws ArgumentError polytope(reshape([-1.0 - 2e-12], 1, 1))
    @test P._nse_softplus(30.0) == 30.0
    @test only(Zygote.gradient(P._nse_softplus, 0.0)) == 0.5
    zero_tangent = only(Zygote.gradient(P._nse_relu, 0.0))
    @test isnothing(zero_tangent) || iszero(zero_tangent)
    for (lower, upper) in ((1, 1), (2, 1), (-Inf, 1), (-1e308, 1e308), (true, 2))
        @test_throws ArgumentError IntervalNeuralSDE(lower, upper; settings...)
    end
    for invalid in ((drift_bound = 0.0,), (diffusion_scale = -1.0,), (hidden_size = true,), (seed = 1.5,), (parameters = (;),))
        @test_throws ArgumentError IntervalNeuralSDE(-1, 1; merge(settings, invalid)...)
    end
    @test_throws ArgumentError PolytopeNeuralSDE([1.0 0.0; 0.0 1.0], [0.0, 0.0], [1.0, 1.0]; settings...)
    @test_throws ArgumentError PolytopeNeuralSDE([1.0 0.0; -1.0 0.0], [-1.0, -1.0], [0.0, 0.0]; settings...)
    @test_throws ArgumentError PolytopeNeuralSDE(reshape([1.0, -1.0], 2, 1), [-1.0, -1.0], [-1.0]; settings...)
    for values in (Float64[], [1.1], [NaN], [true], ones(Float32, 2), ones(2, 1))
        @test_throws ArgumentError model(values)
    end
    for values in (zeros(0, 1), ones(2, 2), fill(NaN, 1, 1), ones(Float32, 2, 1), [0.0])
        @test_throws ArgumentError polytope(values)
    end
    @test_throws ArgumentError negative_log_likelihood(model, [0.0], [0.0, 0.1], 0.01)
    @test_throws ArgumentError negative_log_likelihood(model, [0.0], [2.0], 0.01)
    @test_throws ArgumentError negative_log_likelihood(model, [0.0], [0.1], 0.0)
    @test_throws ArgumentError simulate_interval_neural_sde(model, [0.0], zeros(1, 0); dt = 0.01)
    @test_throws ArgumentError simulate_interval_neural_sde(model, [0.0], ones(Float32, 1, 2); dt = 0.01)
    @test_throws ArgumentError simulate_polytope_neural_sde(polytope, zeros(1, 1), zeros(1, 1, 2); dt = 0.01)
    initial = deepcopy(model.parameters)
    @test_throws ArgumentError fit_interval_neural_sde(model, [0.0], [0.1]; dt = 0.01, iterations = 0)
    @test_throws ArgumentError fit_interval_neural_sde(model, [0.0], [0.1]; dt = 0.01, iterations = 1, learning_rate = 0)
    @test_throws ArgumentError fit_interval_neural_sde(model, [-1.0], [0.1]; dt = 0.01, iterations = 1)
    @test model.parameters == initial
end
