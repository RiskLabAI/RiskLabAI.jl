function _pde_training_batch(model, batch, expected_count = nothing)
    batch isa NamedTuple && haskey(batch, :paths) && haskey(batch, :increments) || throw(ArgumentError("A supplied batch must contain paths and increments"))
    time = ones(size(batch.paths, 1), 1)
    _pde_solver_batch(model.pde, batch.paths, batch.increments, time)
    isnothing(expected_count) || size(batch.paths, 1) == expected_count || throw(ArgumentError("Training batches must match batch_size"))
    gradients = get(batch, :monte_carlo_gradients, nothing)
    model isa FBSDESolver && _pde_control_gradients(model, batch.paths, gradients)
    model isa FBSNNSolver && !isnothing(gradients) && throw(ArgumentError("FBSNN obtains controls from the solution network derivative"))
    model isa FBSDESolver && model.method == "DTNN" && size(batch.paths, 1) < 2 && throw(ArgumentError("DTNN requires at least two observations for public training-mode batch normalization"))
    return (paths = copy(batch.paths), increments = copy(batch.increments), time = time,
        monte_carlo_gradients = isnothing(gradients) ? nothing : copy(gradients))
end

function _pde_draw_training_batch(model, count, rng)
    increments, paths = pde_sample(model.pde, count; rng)
    return _pde_training_batch(model, (paths = paths, increments = increments), count)
end

function _pde_training_inputs(model, iterations, batch_size, init_y, seed, rng, training_batches, validation_batch)
    count, batch = _nse_count(iterations, "iterations"), _nse_count(batch_size, "batch_size")
    initial = _nse_real(init_y, "init_y")
    random = isnothing(rng) ? _nse_rng(seed) : rng
    random isa AbstractRNG || throw(ArgumentError("rng must be a native random-number generator"))
    if model isa FBSDESolver && model.method == "Monte-Carlo"
        !isnothing(training_batches) && !isnothing(validation_batch) || throw(ArgumentError("Monte-Carlo training requires explicit training and validation batches with supplied gradient estimates"))
    end
    if isnothing(training_batches)
        prepared = nothing
        model isa FBSDESolver && model.method == "DTNN" && batch < 2 && throw(ArgumentError("DTNN batch_size must exceed one"))
    else
        training_batches isa Union{Tuple, AbstractVector} && length(training_batches) == count || throw(ArgumentError("Supply exactly one training batch per iteration"))
        prepared = Tuple(_pde_training_batch(model, item, batch) for item in training_batches)
    end
    validation = isnothing(validation_batch) ? _pde_draw_training_batch(model, 128, random) : _pde_training_batch(model, validation_batch)
    return count, batch, initial, random, prepared, validation
end

_pde_optimizer_step(state, parameters, gradient) = isnothing(gradient) ? (state, parameters) : Optimisers.update(state, parameters, gradient)

"""
    solve_pde(model; iterations=20, batch_size=64, init_y=0.0, seed=0,
              rng=nothing, training_batches=nothing, validation_batch=nothing)

Train for a fixed number of iterations and return validation losses and
initial-value estimates. Supplied batches contain `paths` and `increments`;
Monte Carlo batches also contain `monte_carlo_gradients`. Supplied batches
are checked and copied before training. Otherwise one fixed validation batch
of 128 and fresh training batches are sampled using the local RNG.

DeepBSDE initial value and gradient use separate public Adam learning rates
0.1 and 0.001, both with betas `(0.9,0.99)`. Other backward solvers retain
the supplied initial value during training and report the public coefficient
estimate after each update. DTNN preserves public training-mode moments for
validation. Network optimizer state persists across calls; DeepBSDE initial
parameters and their optimizers restart on each call, as in Python.
"""
function solve_pde(model::FBSDESolver; iterations = 20, batch_size = 64, init_y = 0.0,
    seed = 0, rng = nothing, training_batches = nothing, validation_batch = nothing)
    count, batch_size, initial, random, batches, validation = _pde_training_inputs(model, iterations, batch_size, init_y, seed, rng, training_batches, validation_batch)
    model.method == "DTNN" && (model.solver.training = true)
    y0, z0 = [initial], zeros(1, model.pde.dim)
    value_state = Optimisers.setup(Optimisers.Adam(0.1, (0.9, 0.99)), y0)
    gradient_state = Optimisers.setup(Optimisers.Adam(0.001, (0.9, 0.99)), z0)
    losses, estimates = Float64[], Float64[]
    for index in 1:count
        batch = isnothing(batches) ? _pde_draw_training_batch(model, batch_size, random) : batches[index]
        if model.method == "DeepBSDE"
            parameters = (networks = model.parameters, value = y0, gradient = z0)
            value, derivative = Zygote.withgradient(parameters) do supplied
                first(_pde_bsde_loss(model, supplied.networks, batch.paths, batch.increments, batch.time, supplied.value, supplied.gradient, nothing))
            end
            isfinite(value) || throw(DomainError(value, "Nonfinite PDE training loss"))
            derivatives = only(derivative)
            model.optimizer_state, model.parameters = _pde_optimizer_step(model.optimizer_state, model.parameters, derivatives.networks)
            value_state, y0 = _pde_optimizer_step(value_state, y0, derivatives.value)
            gradient_state, z0 = _pde_optimizer_step(gradient_state, z0, derivatives.gradient)
        else
            value, derivative = Zygote.withgradient(model.parameters) do supplied
                first(_pde_bsde_loss(model, supplied, batch.paths, batch.increments, batch.time, y0, z0, batch.monte_carlo_gradients))
            end
            isfinite(value) || throw(DomainError(value, "Nonfinite PDE training loss"))
            model.optimizer_state, model.parameters = _pde_optimizer_step(model.optimizer_state, model.parameters, only(derivative))
        end
        _pde_sync_parameters!(model)
        loss, coefficient, stochastic, payoff = _pde_bsde_loss(model, model.parameters, validation.paths, validation.increments, validation.time, y0, z0, validation.monte_carlo_gradients)
        estimate = model.method == "DeepBSDE" ? only(y0) : Statistics.mean((payoff .- stochastic) ./ coefficient)
        isfinite(loss) && isfinite(estimate) || throw(DomainError((loss, estimate), "Nonfinite PDE validation result"))
        push!(losses, loss)
        push!(estimates, estimate)
    end
    return losses, estimates
end

"""Train the public FBSNN objective with persistent Adam state and derivative-enabled held-out evaluation; return validation losses and network initial values."""
function solve_pde(model::FBSNNSolver; iterations = 20, batch_size = 64, init_y = 0.0,
    seed = 0, rng = nothing, training_batches = nothing, validation_batch = nothing)
    count, batch_size, _, random, batches, validation = _pde_training_inputs(model, iterations, batch_size, init_y, seed, rng, training_batches, validation_batch)
    losses, estimates = Float64[], Float64[]
    for index in 1:count
        batch = isnothing(batches) ? _pde_draw_training_batch(model, batch_size, random) : batches[index]
        value, derivative = Zygote.withgradient(model.parameters) do supplied
            _pde_fbsnn_loss(model, supplied, batch.paths, batch.increments, batch.time)
        end
        isfinite(value) || throw(DomainError(value, "Nonfinite FBSNN training loss"))
        model.optimizer_state, model.parameters = _pde_optimizer_step(model.optimizer_state, model.parameters, only(derivative))
        _pde_sync_parameters!(model)
        loss = _pde_fbsnn_loss(model, model.parameters, validation.paths, validation.increments, validation.time)
        initial_inputs = hcat(zeros(size(validation.paths, 1), 1), validation.paths[:, :, 1])
        estimate = Statistics.mean(_pde_hidden_forward(model.parameters, initial_inputs, sin))
        isfinite(loss) && isfinite(estimate) || throw(DomainError((loss, estimate), "Nonfinite FBSNN validation result"))
        push!(losses, loss)
        push!(estimates, estimate)
    end
    return losses, estimates
end
