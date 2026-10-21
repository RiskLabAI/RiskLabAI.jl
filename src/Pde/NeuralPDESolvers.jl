"""
    FBSDESolver(equation, layer_sizes, learning_rate, method; seed=0, parameters=nothing)

Native public backward-equation solver. `DeepBSDE` takes full input/hidden/
output widths; `DTNN` and `Monte-Carlo` take hidden widths. Network Adam uses
betas `(0.9,0.99)`. The unused first Python DeepBSDE network is omitted.
The set-transformer route retains the stated one-dimensional equation scope
and inserts its required singleton set axis. Monte Carlo controls require
supplied gradient arrays; no estimator is inferred. Parameters use Float64.
"""
mutable struct FBSDESolver
    pde::Equation
    layer_sizes::Tuple
    learning_rate::Float64
    method::String
    solver::Union{AbstractPDENetwork, Tuple}
    parameters::Union{NamedTuple, Tuple}
    optimizer_state::Any
    function FBSDESolver(equation::Equation, layer_sizes, learning_rate, method; seed = 0, parameters = nothing)
        _pde_equation_configuration(equation.dim, equation.total_time, equation.num_time_interval)
        rate = _nse_real(learning_rate, "learning_rate")
        rate > 0 || throw(ArgumentError("The learning rate must be positive"))
        method isa AbstractString && method in ("DeepBSDE", "DTNN", "Monte-Carlo", "Deep-Time-SetTransformer") || throw(ArgumentError("Unknown public PDE solving method"))
        rng = _nse_rng(seed)
        if method == "DeepBSDE"
            sizes = _pde_layer_sizes(layer_sizes)
            first(sizes) == last(sizes) == equation.dim || throw(ArgumentError("DeepBSDE input and output widths must match the equation dimension"))
            models = Tuple(DeepBSDE(sizes; seed = rand(rng, UInt32)) for _ in 2:equation.num_time_interval)
            template = Tuple(model.parameters for model in models)
        else
            sizes = _pde_hidden_sizes(layer_sizes)
            if method == "Deep-Time-SetTransformer"
                equation.dim == 1 || throw(ArgumentError("The public set-transformer solver declares a one-dimensional equation"))
                models = DeepTimeSetTransformer(1; seed)
            elseif method == "DTNN"
                models = TimeDependentNetwork(equation.dim, sizes, equation.dim; seed)
            else
                models = TimeDependentNetworkMonteCarlo(equation.dim, sizes, equation.dim, equation.sigma; seed)
            end
            template = models.parameters
        end
        supplied = _pde_parameter_copy(parameters, template)
        state = method == "DeepBSDE" && isempty(supplied) ? () : Optimisers.setup(Optimisers.Adam(rate, (0.9, 0.99)), supplied)
        model = new(equation, sizes, rate, String(method), models, supplied, state)
        _pde_sync_parameters!(model)
        return model
    end
end

"""
    FBSNNSolver(equation, layer_sizes, learning_rate; seed=0, parameters=nothing)

Public sine-network FBSNN objective with full widths starting at `dim+1`
and ending at one. Analytic sine-network state derivatives remain
differentiable in the parameters, including during validation. Adam retains
the public `(0.99,0.999)` betas. The unused public initial-value argument is
accepted by `solve_pde` without entering the objective.
"""
mutable struct FBSNNSolver
    pde::Equation
    layer_sizes::Tuple
    learning_rate::Float64
    solver::FBSNNNetwork
    parameters::NamedTuple
    optimizer_state::Any
    function FBSNNSolver(equation::Equation, layer_sizes, learning_rate; seed = 0, parameters = nothing)
        _pde_equation_configuration(equation.dim, equation.total_time, equation.num_time_interval)
        sizes = _pde_layer_sizes(layer_sizes)
        first(sizes) == equation.dim + 1 && last(sizes) == 1 || throw(ArgumentError("FBSNN widths must start with time plus state and end with one solution value"))
        rate = _nse_real(learning_rate, "learning_rate")
        rate > 0 || throw(ArgumentError("The learning rate must be positive"))
        model = FBSNNNetwork(sizes; seed, parameters)
        state = Optimisers.setup(Optimisers.Adam(rate, (0.99, 0.999)), model.parameters)
        new(equation, sizes, rate, model, model.parameters, state)
    end
end

function _pde_sync_parameters!(model)
    if model.solver isa Tuple
        for (network, parameters) in zip(model.solver, model.parameters)
            network.parameters = parameters
        end
    else
        model.solver.parameters = model.parameters
    end
    return nothing
end

function _pde_solver_batch(equation, paths, increments, time)
    paths isa AbstractArray{Float64, 3} && !isempty(paths) && size(paths, 2) == equation.dim && size(paths, 3) == equation.num_time_interval + 1 && all(isfinite, paths) || throw(ArgumentError("Paths must be finite Float64 batch-by-dimension-by-time arrays"))
    increments isa AbstractArray{Float64, 3} && size(increments) == (size(paths, 1), equation.dim, equation.num_time_interval) && all(isfinite, increments) || throw(ArgumentError("Brownian increments must match the equation and path batch"))
    time isa AbstractMatrix{Float64} && size(time) == (size(paths, 1), 1) && all(isfinite, time) || throw(ArgumentError("The time multiplier must be a finite Float64 batch-by-one matrix"))
    return nothing
end

function _pde_control_gradients(model, paths, gradients)
    if model.method == "Monte-Carlo"
        gradients isa AbstractArray{Float64, 3} && size(gradients) == (size(paths, 1), model.pde.dim, model.pde.num_time_interval) && all(isfinite, gradients) || throw(ArgumentError("Monte Carlo controls require supplied finite Float64 batch-by-dimension-by-step gradient estimates"))
    else
        isnothing(gradients) || throw(ArgumentError("Monte Carlo gradient inputs apply only to the Monte-Carlo method"))
    end
    return nothing
end

function _pde_control(model, parameters, time, states, index, initial_gradient, monte_carlo_gradients)
    if model.method == "DeepBSDE"
        return index == 1 ? ones(size(states, 1), 1) * initial_gradient : _pde_hidden_forward(parameters[index-1], states, _pde_relu)
    elseif model.method == "DTNN"
        return _pde_dtnn_forward(model.solver, parameters, time, states)
    elseif model.method == "Monte-Carlo"
        return _pde_monte_carlo_forward(model.solver, parameters, time, states, monte_carlo_gradients[:, :, index])
    end
    return _pde_deep_set_forward(parameters, time, reshape(states, size(states, 1), 1, 1))
end

function _pde_bsde_loss(model, parameters, paths, increments, time, initial_value, initial_gradient, monte_carlo_gradients)
    equation = model.pde
    y = ones(size(paths, 1), 1) .* only(initial_value)
    coefficient, stochastic = ones(size(y)), zeros(size(y))
    for index in 1:equation.num_time_interval
        states = paths[:, :, index]
        current_time = time .* (equation.delta_t * (index - 1))
        control = _pde_control(model, parameters, current_time, states, index, initial_gradient, monte_carlo_gradients)
        rate = pde_driver(equation, current_time, states, y, control)
        hamiltonian = pde_hamiltonian(equation, current_time, states, y, control)
        increment = sum(control .* increments[:, :, index]; dims = 2) .+ hamiltonian .* equation.delta_t
        factor = 1 .+ rate .* equation.delta_t
        y = y .* factor .+ increment
        stochastic = stochastic .* factor .+ increment
        coefficient = coefficient .* factor
    end
    payoff = pde_terminal(equation, time .* equation.total_time, paths[:, :, end])
    return Statistics.mean((payoff .- y).^2), coefficient, stochastic, payoff
end

"""Evaluate the public backward-equation terminal MSE and its coefficient, stochastic and payoff arrays on supplied paths."""
function compute_pde_loss(model::FBSDESolver, paths, increments, time, initial_value, initial_gradient; monte_carlo_gradients = nothing)
    Zygote.ignore() do
        _pde_solver_batch(model.pde, paths, increments, time)
        initial_value isa AbstractVector{Float64} && length(initial_value) == 1 && all(isfinite, initial_value) || throw(ArgumentError("The initial value must be one finite Float64 parameter"))
        initial_gradient isa AbstractMatrix{Float64} && size(initial_gradient) == (1, model.pde.dim) && all(isfinite, initial_gradient) || throw(ArgumentError("The initial gradient must be a finite Float64 one-by-dimension matrix"))
        _pde_control_gradients(model, paths, monte_carlo_gradients)
    end
    return _pde_bsde_loss(model, model.parameters, paths, increments, time, initial_value, initial_gradient, monte_carlo_gradients)
end

function _pde_sine_value_gradient(layers, inputs)
    projected = _pde_dense(first(layers), inputs)
    if length(layers) == 1
        return projected, ones(size(projected)) * first(layers).weight
    end
    output, gradient = _pde_sine_value_gradient(Base.tail(layers), sin.(projected))
    return output, (gradient .* cos.(projected)) * first(layers).weight
end

function _pde_fbsnn_loss(model, parameters, paths, increments, time)
    equation, loss = model.pde, 0.0
    for index in 1:equation.num_time_interval
        states, next_states = paths[:, :, index], paths[:, :, index+1]
        current_time, next_time = time .* (equation.delta_t * (index - 1)), time .* (equation.delta_t * index)
        value, derivative = _pde_sine_value_gradient(parameters.layers, hcat(current_time, states))
        next_value = _pde_hidden_forward(parameters, hcat(next_time, next_states), sin)
        control = derivative[:, 2:end] .* pde_sigma(equation, states)
        rate = pde_driver(equation, current_time, states, value, control)
        hamiltonian = pde_hamiltonian(equation, current_time, states, value, control)
        predicted = value .* (1 .+ rate .* equation.delta_t) .+ hamiltonian .* equation.delta_t .+ sum(control .* increments[:, :, index]; dims = 2)
        loss += Statistics.mean((predicted .- next_value).^2)
    end
    terminal_time, terminal_states = time .* equation.total_time, paths[:, :, end]
    payoff = pde_terminal(equation, terminal_time, terminal_states)
    terminal_value = _pde_hidden_forward(parameters, hcat(terminal_time, terminal_states), sin)
    return loss + Statistics.mean((payoff .- terminal_value).^2)
end

"""Evaluate the public FBSNN sum of one-step residual MSEs and terminal MSE, retaining state derivatives during evaluation."""
function compute_pde_loss(model::FBSNNSolver, paths, increments, time, initial_value = nothing)
    Zygote.ignore() do
        _pde_solver_batch(model.pde, paths, increments, time)
    end
    return _pde_fbsnn_loss(model, model.parameters, paths, increments, time)
end
