function _fit_neural_sde(model, states, next_states; dt, iterations, learning_rate)
    count = _nse_count(iterations, "iterations")
    rate = _nse_real(learning_rate, "learning_rate"; positive = true)
    timestep = _nse_real(dt, "dt"; positive = true)
    negative_log_likelihood(model, states, next_states, timestep)
    optimizer = Optimisers.setup(Optimisers.Adam(rate, (0.9, 0.999), 1e-8), model.parameters)
    history = Float64[]
    for _ in 1:count
        objective = parameters -> _nse_likelihood(model, parameters, states, next_states, timestep)
        loss, pullback = Zygote.pullback(objective, model.parameters)
        isfinite(loss) || throw(ArgumentError("Transition likelihood is nonfinite"))
        gradients = only(pullback(one(loss)))
        all(value -> !isnothing(value) && all(isfinite, value), values(gradients)) || error("Nonfinite neural-SDE gradients")
        optimizer, parameters = Optimisers.update(optimizer, model.parameters, gradients)
        all(value -> all(isfinite, value), values(parameters)) || error("Nonfinite neural-SDE parameters")
        model.parameters = parameters
        push!(history, loss)
    end
    return history
end

"""Fit the scalar Euler likelihood on supplied training transitions using fixed iterations and native Adam; return pre-update losses."""
function fit_interval_neural_sde(model::IntervalNeuralSDE, states, next_states; dt, iterations, learning_rate = 0.001)
    return _fit_neural_sde(model, states, next_states; dt, iterations, learning_rate)
end

"""
Fit the supplied polytope interior transitions by approximate Euler likelihood.
Only training transitions enter fixed-step Adam. Finite optimization does not
certify parameter recovery, HJM consistency or global optimality.
"""
function fit_polytope_neural_sde(model::PolytopeNeuralSDE, states, next_states; dt, iterations, learning_rate = 0.001)
    return _fit_neural_sde(model, states, next_states; dt, iterations, learning_rate)
end

"""
Apply ordinary scalar Euler steps to supplied Brownian increments, with shape
`(paths,steps)`. Return detached `(paths,steps+1)` states. An exit invalidates
the entire run and reports its one-based step; no clipping or reflection is
performed. Use coupled finer increments to assess discretization sensitivity.
Simulation remains under the fitted measure.
"""
function simulate_interval_neural_sde(model::IntervalNeuralSDE, initial, brownian_increments; dt)
    return Zygote.ignore() do
        timestep = _nse_real(dt, "dt"; positive = true)
        _nse_states(model, initial)
        increments = brownian_increments
        increments isa AbstractMatrix{Float64} && size(increments, 1) == length(initial) && size(increments, 2) > 0 && all(isfinite, increments) || throw(ArgumentError("Brownian increments must match initial states and Float64 model precision"))
        states = reshape(copy(initial), :, 1)
        for time in axes(increments, 2)
            drift, diffusion = model(states[:, end])
            proposal = states[:, end] + timestep .* drift + diffusion .* increments[:, time]
            all(isfinite, proposal) && all(value -> model.lower <= value <= model.upper, proposal) || error("Euler step $time exits the admissible price state; reduce the step and reassess")
            states = hcat(states, proposal)
        end
        return states
    end
end

"""
Ordinary polytope Euler simulation with explicit whole-run exit failure.
Initial states are `(paths,factors)` and Brownian increments are
`(paths,steps,factors)` with variance dt supplied by the caller. Return
detached `(paths,steps+1,factors)` states. No projection, clipping, reflection
or discarded path is introduced.
"""
function simulate_polytope_neural_sde(model::PolytopeNeuralSDE, initial, brownian_increments; dt)
    return Zygote.ignore() do
        timestep = _nse_real(dt, "dt"; positive = true)
        _nse_states(model, initial)
        increments = brownian_increments
        increments isa AbstractArray{Float64, 3} && size(increments, 1) == size(initial, 1) && size(increments, 2) > 0 && size(increments, 3) == model.dimension && all(isfinite, increments) || throw(ArgumentError("Brownian increments must match state dimensions and Float64 model precision"))
        current = copy(initial)
        states = reshape(copy(initial), size(initial, 1), 1, model.dimension)
        for time in axes(increments, 2)
            drift, diffusion = model(current)
            noise = permutedims(hcat((diffusion[index, :, :] * increments[index, time, :] for index in axes(current, 1))...))
            proposal = current + timestep .* drift + noise
            try
                _nse_states(model, proposal)
            catch exception
                exception isa ArgumentError || rethrow()
                error("Euler step $time exits the polytope; reduce the step and reassess")
            end
            states = cat(states, reshape(proposal, size(initial, 1), 1, model.dimension); dims = 2)
            current = proposal
        end
        return states
    end
end
