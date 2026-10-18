import Zygote
import Optimisers

function _hedging_array(value, name, dimension)
    value isa AbstractArray && ndims(value) == dimension && eltype(value) <: AbstractFloat && !isempty(value) && all(isfinite, value) || throw(ArgumentError("$name must be a nonempty finite floating array with $dimension dimensions"))
    return value
end

function _hedging_count(value, name)
    value isa Integer && !(value isa Bool) && 0 < value <= typemax(Int) || throw(ArgumentError("$name must be a positive supported integer"))
    return Int(value)
end

function _hedging_rng(seed)
    seed isa Integer && !(seed isa Bool) && typemin(Int64) <= seed <= typemax(UInt64) || throw(ArgumentError("seed must be an integer in the supported 64-bit range"))
    return MersenneTwister(UInt64(mod(BigInt(seed), BigInt(1) << 64)))
end

function _hedging_positive(value, name; allow_zero = false)
    result = _tail_real_scalar(value, name)
    (allow_zero ? result >= 0 : result > 0) || throw(ArgumentError("$name is below its allowed bound"))
    return result
end

"""
    gbm_hedging_paths(; spot, maturity, volatility, drift, steps, paths, seed,
        normal_draws=nothing)

Return Float64 GBM observations with shape `(paths,steps+1)`. The local Julia
RNG preserves global state. Supplied standard-normal primitive draws with
shape `(paths,steps)` permit exact cross-language comparisons; equal seeds
do not identify equal random streams. Drift specifies the simulation measure.
Zero interest and dividend rates remain the benchmark's accounting convention.
"""
function gbm_hedging_paths(; spot, maturity, volatility, drift, steps, paths, seed, normal_draws = nothing)
    initial = _hedging_positive(spot, "spot")
    horizon = _hedging_positive(maturity, "maturity")
    sigma = _hedging_positive(volatility, "volatility"; allow_zero = true)
    mu = _tail_real_scalar(drift, "drift")
    count, samples = _hedging_count(steps, "steps"), _hedging_count(paths, "paths")
    rng = _hedging_rng(seed)
    draws = if isnothing(normal_draws)
        randn(rng, samples, count)
    else
        _hedging_array(normal_draws, "normal_draws", 2)
        size(normal_draws) == (samples, count) || throw(ArgumentError("normal_draws must match paths and steps"))
        Float64.(normal_draws)
    end
    dt = horizon / count
    increments = (mu - sigma^2 / 2) * dt .+ sigma * sqrt(dt) .* draws
    result = initial .* exp.(hcat(zeros(samples, 1), cumsum(increments; dims = 2)))
    all(value -> isfinite(value) && value > 0, result) || throw(ArgumentError("GBM paths exceeded the supported numerical range"))
    return result
end

"""
    hedging_pnl(prices, positions; strike, cost_rate, premium=0.0)

Return terminal cash flow of a short European call and self-financing hedge.
Prices have shape `(paths,steps+1)` and positions `(paths,steps)`, with the same
floating element type. Proportional costs include initial buying and terminal
liquidation. Positions may be unconstrained; their adaptation to available
information is the caller's responsibility. Native Zygote gradients are supported.
"""
function hedging_pnl(prices, positions; strike, cost_rate, premium = 0.0)
    strike_value, cost, initial_cash = Zygote.ignore() do
        _hedging_array(prices, "prices", 2)
        _hedging_array(positions, "positions", 2)
        size(prices) == (size(positions, 1), size(positions, 2) + 1) && all(>(0), prices) && eltype(prices) == eltype(positions) || throw(ArgumentError("Positive prices and matching positions must share a floating element type"))
        type = eltype(prices)
        return type(_hedging_positive(strike, "strike")), type(_hedging_positive(cost_rate, "cost_rate"; allow_zero = true)), type(_tail_real_scalar(premium, "premium"))
    end
    trades = hcat(positions[:, 1:1], positions[:, 2:end] .- positions[:, 1:end-1], -positions[:, end:end])
    gains = vec(sum(positions .* (prices[:, 2:end] .- prices[:, 1:end-1]); dims = 2))
    costs = cost .* vec(sum(prices .* abs.(trades); dims = 2))
    excess = prices[:, end] .- strike_value
    payoff = ifelse.(excess .> 0, excess, zero(eltype(prices)))
    result = initial_cash .+ gains .- costs .- payoff
    Zygote.ignore() do
        all(isfinite, result) || throw(ArgumentError("Hedging cash flow exceeded the supported numerical range"))
    end
    return result
end

"""Stable entropic risk of supplied PnL with positive risk aversion and native gradients."""
function entropic_risk(pnl, risk_aversion)
    aversion, anchor = Zygote.ignore() do
        _hedging_array(pnl, "pnl", 1)
        return eltype(pnl)(_hedging_positive(risk_aversion, "risk_aversion")), minimum(pnl)
    end
    result = -anchor + (log(sum(exp.(-aversion .* (pnl .- anchor)))) - eltype(pnl)(log(length(pnl)))) / aversion
    Zygote.ignore() do
        isfinite(result) || throw(ArgumentError("Entropic risk exceeded the supported numerical range"))
    end
    return result
end

"""Zero-rate European call deltas at the preterminal equally spaced observation times."""
function black_scholes_call_delta(prices; strike, maturity, volatility)
    _hedging_array(prices, "prices", 2)
    size(prices, 2) >= 2 && all(>(0), prices) || throw(ArgumentError("At least two positive price observations are required"))
    type = eltype(prices)
    strike_value = type(_hedging_positive(strike, "strike"))
    horizon = type(_hedging_positive(maturity, "maturity"))
    sigma = type(_hedging_positive(volatility, "volatility"))
    count = size(prices, 2) - 1
    remaining = horizon .* type.(count:-1:1) ./ type(count)
    d1 = (log.(prices[:, 1:end-1] ./ strike_value) .+ sigma^2 .* remaining' ./ type(2)) ./ (sigma .* sqrt.(remaining'))
    return type.(cdf.(Normal(), d1))
end

function _hedging_parameters(parameters, hidden_size)
    parameters isa NamedTuple && keys(parameters) == (:weight1, :bias1, :weight2, :bias2, :weight3, :bias3) || throw(ArgumentError("parameters must name the three dense weights and biases"))
    shapes = ((hidden_size, 3), (hidden_size,), (hidden_size, hidden_size), (hidden_size,), (1, hidden_size), (1,))
    for (value, shape) in zip(values(parameters), shapes)
        value isa AbstractArray{Float64} && size(value) == shape && all(isfinite, value) || throw(ArgumentError("hedge parameters must be finite Float64 arrays with the specified layer shapes"))
    end
    return parameters
end

"""
    DeepHedge(hidden_size=16; seed=0, parameters=nothing)

Shared three-layer dense hedge with two tanh hidden activations. Inputs are
remaining time, current log-price relative to strike and the previous holding;
output positions are unconstrained. The local seeded initializer uses the
public fan-in uniform bounds in Float32 before conversion to Float64.
Explicit Float64 parameter arrays allow shared-weight validation. Evaluate
with `model(prices; strike,maturity)` and fit with `train_deep_hedge`.
"""
mutable struct DeepHedge
    hidden_size::Int
    parameters::NamedTuple
    function DeepHedge(hidden_size = 16; seed = 0, parameters = nothing)
        width = _hedging_count(hidden_size, "hidden_size")
        rng = _hedging_rng(seed)
        if isnothing(parameters)
            initialize(shape, inputs) = Float64.((rand(rng, Float32, shape) .* 2f0 .- 1f0) .* Float32(1 / sqrt(inputs)))
            parameters = (weight1 = initialize((width, 3), 3), bias1 = initialize((width,), 3),
                weight2 = initialize((width, width), width), bias2 = initialize((width,), width),
                weight3 = initialize((1, width), width), bias3 = initialize((1,), width))
        end
        checked = _hedging_parameters(parameters, width)
        new(width, map(copy, checked))
    end
end

function _deep_hedge_positions(parameters, prices, strike, maturity)
    count = size(prices, 2) - 1
    initial = (previous = zeros(eltype(prices), size(prices, 1)), positions = ())
    result = foldl(1:count; init = initial) do state, index
        features = vcat(fill(maturity * (count - index + 1) / count, 1, size(prices, 1)),
            permutedims(log.(prices[:, index] ./ strike)), permutedims(state.previous))
        hidden1 = tanh.(parameters.weight1 * features .+ parameters.bias1)
        hidden2 = tanh.(parameters.weight2 * hidden1 .+ parameters.bias2)
        holding = vec(parameters.weight3 * hidden2 .+ parameters.bias3)
        return (previous = holding, positions = (state.positions..., holding))
    end
    return hcat(result.positions...)
end

function (model::DeepHedge)(prices; strike, maturity)
    strike_value, horizon = Zygote.ignore() do
        _hedging_array(prices, "prices", 2)
        eltype(prices) == Float64 && size(prices, 2) >= 2 && all(>(0), prices) || throw(ArgumentError("Hedge prices must be positive Float64 observations with at least two times"))
        _hedging_parameters(model.parameters, model.hidden_size)
        return _hedging_positive(strike, "strike"), _hedging_positive(maturity, "maturity")
    end
    return _deep_hedge_positions(model.parameters, prices, strike_value, horizon)
end

"""
    train_deep_hedge(model, training_prices; strike, maturity, cost_rate,
        risk_aversion, iterations, learning_rate=0.001)

Mutate the hedge parameters with full-batch Adam and return each pre-update
entropic loss. Only supplied training paths enter optimization. Adam retains
the public moment decays `(0.9,0.999)` and epsilon `1e-8`. Evaluate independent
paths afterwards; no evaluation outcomes choose weights or stopping here.
Finite gradients/parameters are required; no optimality or superiority is asserted.
"""
function train_deep_hedge(model::DeepHedge, training_prices; strike, maturity, cost_rate,
    risk_aversion, iterations, learning_rate = 0.001)
    count = _hedging_count(iterations, "iterations")
    rate = _hedging_positive(learning_rate, "learning_rate")
    model(training_prices; strike, maturity)
    strike_value = _hedging_positive(strike, "strike")
    horizon = _hedging_positive(maturity, "maturity")
    objective(parameters) = entropic_risk(hedging_pnl(training_prices,
        _deep_hedge_positions(parameters, training_prices, strike_value, horizon); strike, cost_rate), risk_aversion)
    state = Optimisers.setup(Optimisers.Adam(rate, (0.9, 0.999), 1e-8), model.parameters)
    history = Float64[]
    for _ in 1:count
        loss, pullback = Zygote.pullback(objective, model.parameters)
        gradient = only(pullback(one(loss)))
        all(value -> value isa AbstractArray && all(isfinite, value), values(gradient)) || error("Nonfinite hedging gradients")
        state, parameters = Optimisers.update(state, model.parameters, gradient)
        all(value -> all(isfinite, value), values(parameters)) || error("Nonfinite fitted hedge parameters")
        model.parameters = parameters
        push!(history, loss)
    end
    return history
end
