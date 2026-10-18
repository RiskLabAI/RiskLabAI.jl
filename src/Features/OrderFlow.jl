using Random: AbstractRNG, MersenneTwister, default_rng, randexp
using Optim

function _hawkes_parameters(mu, alpha, beta)
    mu = _feature_real_scalar(mu, "mu"; minimum = 0, strict = true)
    alpha = _feature_real_scalar(alpha, "alpha"; minimum = 0)
    beta = _feature_real_scalar(beta, "beta"; minimum = 0, strict = true)
    alpha < beta || throw(ArgumentError("stationarity requires alpha < beta"))
    return mu, alpha, beta
end

function _hawkes_events(events, horizon)
    values = _feature_real_array(events, "events", 1; empty = true)
    all(0 .<= values .<= horizon) && all(>(0), diff(values)) || throw(ArgumentError("events must be strictly increasing in [0, horizon]"))
    return values
end

"""Return empty-prehistory predictable exponential Hawkes intensity using only events strictly before each query time."""
function hawkes_intensity(times, events, mu, alpha, beta)
    mu, alpha, beta = _hawkes_parameters(mu, alpha, beta)
    times = _feature_real_array(times, "times", 1; empty = true)
    events = _hawkes_events(events, Inf)
    all(>=(0), times) || throw(ArgumentError("times must be nonnegative"))
    return [mu + alpha * sum(exp(-beta * (time - event)) for event in events if event < time; init = 0.0) for time in times]
end

"""Integrate the empty-history exponential Hawkes intensity over a positive horizon."""
function hawkes_integrated_intensity(events, horizon, mu, alpha, beta)
    mu, alpha, beta = _hawkes_parameters(mu, alpha, beta)
    horizon = _feature_real_scalar(horizon, "horizon"; minimum = 0, strict = true)
    events = _hawkes_events(events, horizon)
    result = mu * horizon + alpha / beta * sum(-expm1.(-beta .* (horizon .- events)))
    isfinite(result) || throw(ArgumentError("integrated intensity exceeds floating-point range"))
    return result
end

"""Evaluate the exact Hawkes event log likelihood with an O(n) exponential recurrence and empty prehistory."""
function hawkes_log_likelihood(events, horizon, mu, alpha, beta)
    mu, alpha, beta = _hawkes_parameters(mu, alpha, beta)
    horizon = _feature_real_scalar(horizon, "horizon"; minimum = 0, strict = true)
    events = _hawkes_events(events, horizon)
    excitation, total, previous = 0.0, 0.0, 0.0
    for (i, event) in enumerate(events)
        excitation = exp(-beta * (event - previous)) * (excitation + (i > 1))
        total += log(mu + alpha * excitation)
        previous = event
    end
    return total - hawkes_integrated_intensity(events, horizon, mu, alpha, beta)
end

"""
    simulate_hawkes(horizon, mu, alpha, beta; random_state=nothing, max_events=100000)

Simulate the stationary-parameter exponential Hawkes model with empty
prehistory using adaptive thinning. Accept a local seed, an `AbstractRNG` or
the default RNG. Reaching `max_events` raises an error instead of returning
a truncated history. Equal seeds do not imply NumPy/Julia stream equality.
"""
function simulate_hawkes(horizon, mu, alpha, beta; random_state = nothing, max_events = 100000)
    mu, alpha, beta = _hawkes_parameters(mu, alpha, beta)
    horizon = _feature_real_scalar(horizon, "horizon"; minimum = 0, strict = true)
    limit = _feature_positive_integer(max_events, "max_events")
    rng = if isnothing(random_state)
        default_rng()
    elseif random_state isa AbstractRNG
        random_state
    elseif random_state isa Integer && !(random_state isa Bool) && random_state >= 0
        MersenneTwister(random_state)
    else
        throw(ArgumentError("random_state must be a nonnegative seed or an AbstractRNG"))
    end
    events = Float64[]
    time, excitation = 0.0, 0.0
    while true
        bound = mu + excitation
        wait = randexp(rng) / bound
        next_time = time + wait
        next_time > horizon && break
        next_time > time || error("time increments are below floating-point resolution")
        excitation *= exp(-beta * wait)
        time = next_time
        if rand(rng) * bound <= mu + excitation
            length(events) < limit || error("max_events reached before simulation completed")
            push!(events, time)
            excitation += alpha
        end
    end
    return events
end

function _hawkes_scaled_objective_gradient!(gradient, parameters, events)
    baseline, branching, decay = parameters
    remaining = 1 .- events
    value = baseline + branching * sum(-expm1.(-decay .* remaining))
    gradient .= (1.0, sum(-expm1.(-decay .* remaining)), branching * sum(remaining .* exp.(-decay .* remaining)))
    excitation, derivative, previous = 0.0, 0.0, 0.0
    for (i, event) in enumerate(events)
        gap = event - previous
        multiplier = exp(-decay * gap)
        derivative = multiplier * derivative - gap * multiplier * (excitation + (i > 1))
        excitation = multiplier * (excitation + (i > 1))
        intensity = baseline + branching * decay * excitation
        value -= log(intensity)
        gradient[1] -= 1 / intensity
        gradient[2] -= decay * excitation / intensity
        gradient[3] -= branching * (excitation + decay * derivative) / intensity
        previous = event
    end
    return value
end

"""
    fit_hawkes(events, horizon; initial=nothing)

Fit the empty-history exponential Hawkes likelihood under positive baseline
and decay and branching ratio in `[0,1)`. Use the same three fixed starts as
the public Python reference, or one caller-supplied start. Time is rescaled
to the unit interval; Optim's bounded solver uses the analytic score.
Return parameters, original-unit likelihood and convergence diagnostics.
A numerical fit does not establish parameter identification or global
likelihood optimality; at least two events are required.
"""
function fit_hawkes(events, horizon; initial = nothing)
    horizon = _feature_real_scalar(horizon, "horizon"; minimum = 0, strict = true)
    events = _hawkes_events(events, horizon)
    n = length(events)
    n >= 2 || throw(ArgumentError("at least two events are required to fit three parameters"))
    scaled = events / horizon
    starts = [[n * (1 - ratio), ratio, Float64(n)] for ratio in (0.1, 0.5, 0.8)]
    if !isnothing(initial)
        supplied = _feature_real_array(initial, "initial", 1)
        length(supplied) == 3 || throw(ArgumentError("initial must contain mu, alpha and beta"))
        mu, alpha, beta = _hawkes_parameters(supplied...)
        starts = [[mu * horizon, alpha / beta, beta * horizon]]
    end
    lower, upper = [1e-9, 0.0, 1e-9], [Inf, 1 - 1e-8, Inf]
    objective(x) = _hawkes_scaled_objective_gradient!(zeros(3), x, scaled)
    gradient!(g, x) = (_hawkes_scaled_objective_gradient!(g, x, scaled); g)
    results = Any[]
    for start in starts
        start = max.(start, lower .+ 1e-8)
        start[2] = min(start[2], upper[2] - 1e-8)
        result = Optim.optimize(objective, gradient!, lower, upper, start, Fminbox(LBFGS()), Optim.Options(g_tol = 1e-8, iterations = 1000, outer_iterations = 20))
        Optim.converged(result) && isfinite(Optim.minimum(result)) && push!(results, result)
    end
    isempty(results) && error("Hawkes likelihood optimization failed")
    best = results[argmin(Optim.minimum.(results))]
    baseline, branching, decay = Optim.minimizer(best)
    mu, alpha, beta = baseline / horizon, branching * decay / horizon, decay / horizon
    return (; mu, alpha, beta, log_likelihood = hawkes_log_likelihood(events, horizon, mu, alpha, beta),
        branching_ratio = branching, converged_starts = length(results), message = "Optim bounded likelihood convergence")
end

"""
    volume_synchronized_pin(buy_volume, sell_volume; bucket_volume, window)

Compute rolling VPIN for completed, equally sized, preclassified volume
buckets. Return one value aligned to the last bucket of each full window.
Partial buckets and implicit trade classification are not accepted.
"""
function volume_synchronized_pin(buy_volume, sell_volume; bucket_volume, window)
    buy = _feature_real_array(buy_volume, "buy_volume", 1)
    sell = _feature_real_array(sell_volume, "sell_volume", 1)
    volume = _feature_real_scalar(bucket_volume, "bucket_volume"; minimum = 0, strict = true)
    window = _feature_positive_integer(window, "window")
    length(buy) == length(sell) && all(>=(0), buy) && all(>=(0), sell) && all(abs.((buy + sell) / volume .- 1) .<= 2e-12) || throw(ArgumentError("volumes must be nonnegative completed equal-volume buckets"))
    window <= length(buy) || throw(ArgumentError("window exceeds the completed bucket count"))
    imbalance = abs.(buy - sell) / volume
    return [sum(@view imbalance[i:i+window-1]) / window for i in 1:length(buy)-window+1]
end
