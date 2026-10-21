import Random
import Zygote
import Optimisers
import Optim
import LinearAlgebra

function _timegan_real(value, name; minimum = nothing, strict = false)
    value isa Real && !(value isa Bool) && isfinite(value) || throw(ArgumentError("$name must be a finite real scalar"))
    result = Float64(value)
    isfinite(result) || throw(ArgumentError("$name is outside the Float64 range"))
    if !isnothing(minimum) && (strict ? result <= minimum : result < minimum)
        throw(ArgumentError("$name is below its allowed bound"))
    end
    return result
end

function _timegan_count(value, name)
    value isa Integer && !(value isa Bool) && 0 < value <= typemax(Int) || throw(ArgumentError("$name must be a positive supported integer"))
    return Int(value)
end

function _timegan_array(value, name, dimension)
    value isa AbstractArray && ndims(value) == dimension && !isempty(value) || throw(ArgumentError("$name must be a nonempty real array with $dimension dimensions"))
    return map(item -> _timegan_real(item, name), value)
end

function _timegan_rng(seed)
    seed isa Integer && !(seed isa Bool) && typemin(Int64) <= seed <= typemax(UInt64) || throw(ArgumentError("seed must be an integer in the supported 64-bit range"))
    return Random.MersenneTwister(UInt64(mod(BigInt(seed), BigInt(1) << 64)))
end

function _timegan_windows(segment, length)
    return cat((reshape(segment[index:index+length-1, :], 1, length, size(segment, 2)) for index in 1:(size(segment, 1)-length+1))...; dims = 1)
end

"""
    chronological_timegan_data(series; training_end, window_length)

Split before windowing and scale only with training-segment feature minima
and ranges. Constant training features have scale one; validation is never
clipped. Returned arrays have `(windows,time,features)` dimensions, and no
window crosses the split. `training_end` is the training row count.
"""
function chronological_timegan_data(series; training_end, window_length)
    values = _timegan_array(series, "series", 2)
    split = _timegan_count(training_end, "training_end")
    length = _timegan_count(window_length, "window_length")
    length >= 2 && split >= length && size(values, 1) - split >= length || throw(ArgumentError("Both chronological segments need a complete window of length >= 2"))
    training = values[1:split, :]
    minimum_values = vec(minimum(training; dims = 1))
    scale = vec(maximum(training; dims = 1)) .- minimum_values
    scale[scale .== 0] .= 1.0
    scaled = (values .- minimum_values') ./ scale'
    return (training = _timegan_windows(scaled[1:split, :], length), validation = _timegan_windows(scaled[split+1:end, :], length),
        minimum = minimum_values, scale = scale, training_end = split)
end

function _timegan_shapes(features, hidden)
    recurrent(inputs) = (weight_ih = (3hidden, inputs), weight_hh = (3hidden, hidden), bias_ih = (3hidden,), bias_hh = (3hidden,))
    output(inputs, outputs) = (weight = (outputs, inputs), bias = (outputs,))
    return (embedding = (forward = recurrent(features), output = output(hidden, hidden)),
        recovery = (forward = recurrent(hidden), output = output(hidden, features)),
        generator_cell = recurrent(2hidden), generator_output = output(hidden, hidden),
        discriminator = (forward = recurrent(hidden), backward = recurrent(hidden), output = output(2hidden, 1)))
end

function _timegan_validate_parameters(parameters, shapes)
    if shapes isa NamedTuple
        parameters isa NamedTuple && keys(parameters) == keys(shapes) || throw(ArgumentError("TimeGAN parameters must match the named public architecture"))
        for name in keys(shapes)
            _timegan_validate_parameters(getproperty(parameters, name), getproperty(shapes, name))
        end
    else
        parameters isa AbstractArray{Float64} && size(parameters) == shapes && all(isfinite, parameters) || throw(ArgumentError("TimeGAN parameter arrays must have the required Float64 shapes and finite values"))
    end
    return parameters
end

function _timegan_initial_parameters(rng, features, hidden)
    uniform(shape, inputs) = Float64.((rand(rng, Float32, shape) .* 2f0 .- 1f0) .* Float32(1 / sqrt(inputs)))
    recurrent(inputs) = (weight_ih = uniform((3hidden, inputs), hidden), weight_hh = uniform((3hidden, hidden), hidden),
        bias_ih = uniform((3hidden,), hidden), bias_hh = uniform((3hidden,), hidden))
    output(inputs, outputs) = (weight = uniform((outputs, inputs), inputs), bias = uniform((outputs,), inputs))
    return (embedding = (forward = recurrent(features), output = output(hidden, hidden)),
        recovery = (forward = recurrent(hidden), output = output(hidden, features)),
        generator_cell = recurrent(2hidden), generator_output = output(hidden, hidden),
        discriminator = (forward = recurrent(hidden), backward = recurrent(hidden), output = output(2hidden, 1)))
end

"""
    TimeGAN(features, hidden_size=12; seed=0, parameters=nothing)

Temporal-only GRU TimeGAN with causal embedding/recovery, an autoregressive
generator and a bidirectional discriminator. The public reset/update/new GRU
gate order and reset-after hidden projection are retained. Parameters and
model inputs use Float64. Local initialization preserves global RNG state;
supplied parameter arrays are copied for shared-weight comparisons.
Use `latent_sequence`, `timegan_losses`, `sample_timegan` and `fit_timegan`.
Synthetic fixed-length windows do not define an unlimited market path.
"""
mutable struct TimeGAN
    features::Int
    hidden_size::Int
    parameters::NamedTuple
    function TimeGAN(features, hidden_size = 12; seed = 0, parameters = nothing)
        dimensions = _timegan_count(features, "features")
        hidden = _timegan_count(hidden_size, "hidden_size")
        rng = _timegan_rng(seed)
        supplied = isnothing(parameters) ? _timegan_initial_parameters(rng, dimensions, hidden) : parameters
        _timegan_validate_parameters(supplied, _timegan_shapes(dimensions, hidden))
        new(dimensions, hidden, deepcopy(supplied))
    end
end

_timegan_sigmoid(value) = value >= 0 ? 1 / (1 + exp(-value)) : exp(value) / (1 + exp(value))

function _timegan_check(model, value, name, features)
    value isa AbstractArray{Float64, 3} && size(value, 1) > 0 && size(value, 2) >= 2 && size(value, 3) == features && all(isfinite, value) || throw(ArgumentError("$name must be finite Float64 (windows,time>=2,$features) data matching the model"))
    _timegan_validate_parameters(model.parameters, _timegan_shapes(model.features, model.hidden_size))
    return nothing
end

function _timegan_cell(parameters, input, state)
    hidden = size(state, 1)
    observed = parameters.weight_ih * input .+ parameters.bias_ih
    previous = parameters.weight_hh * state .+ parameters.bias_hh
    reset = _timegan_sigmoid.(observed[1:hidden, :] .+ previous[1:hidden, :])
    update = _timegan_sigmoid.(observed[hidden+1:2hidden, :] .+ previous[hidden+1:2hidden, :])
    candidate = tanh.(observed[2hidden+1:3hidden, :] .+ reset .* previous[2hidden+1:3hidden, :])
    return candidate .+ (state .- candidate) .* update
end

function _timegan_recurrent(parameters, inputs; backward = false)
    hidden = size(parameters.weight_hh, 2)
    times = backward ? (size(inputs, 2):-1:1) : (1:size(inputs, 2))
    initial = (hidden = zeros(hidden, size(inputs, 1)), sequence = zeros(hidden * size(inputs, 1), 0))
    result = foldl(times; init = initial) do state, time
        next_hidden = _timegan_cell(parameters, permutedims(inputs[:, time, :]), state.hidden)
        return (hidden = next_hidden, sequence = hcat(state.sequence, vec(next_hidden)))
    end
    sequence = backward ? reverse(result.sequence; dims = 2) : result.sequence
    return permutedims(reshape(sequence, hidden, size(inputs, 1), size(inputs, 2)), (2, 3, 1))
end

function _timegan_sequence_map(parameters, inputs; sigmoid = true)
    forward = _timegan_recurrent(parameters.forward, inputs)
    hidden = haskey(parameters, :backward) ? cat(forward, _timegan_recurrent(parameters.backward, inputs; backward = true); dims = 3) : forward
    flattened = reshape(permutedims(hidden, (3, 1, 2)), size(hidden, 3), :)
    projected = parameters.output.weight * flattened .+ parameters.output.bias
    output = permutedims(reshape(projected, size(projected, 1), size(inputs, 1), size(inputs, 2)), (2, 3, 1))
    return sigmoid ? _timegan_sigmoid.(output) : output
end

function _timegan_latent(parameters, noise, teacher)
    hidden, windows = size(noise, 3), size(noise, 1)
    initial = (state = zeros(hidden, windows), previous = zeros(hidden, windows), sequence = zeros(hidden * windows, 0))
    result = foldl(1:size(noise, 2); init = initial) do prior, time
        previous = !isnothing(teacher) && time > 1 ? permutedims(teacher[:, time-1, :]) : prior.previous
        state = _timegan_cell(parameters.generator_cell, vcat(previous, permutedims(noise[:, time, :])), prior.state)
        output = _timegan_sigmoid.(parameters.generator_output.weight * state .+ parameters.generator_output.bias)
        return (state = state, previous = output, sequence = hcat(prior.sequence, vec(output)))
    end
    return permutedims(reshape(result.sequence, hidden, windows, size(noise, 2)), (2, 3, 1))
end

"""Generate latent windows freely or condition time t on the supplied teacher latent at t-1."""
function latent_sequence(model::TimeGAN, noise; teacher = nothing)
    Zygote.ignore() do
        _timegan_check(model, noise, "noise", model.hidden_size)
        if !isnothing(teacher)
            _timegan_check(model, teacher, "teacher", model.hidden_size)
            size(teacher) == size(noise) || throw(ArgumentError("Teacher and noise shapes must match"))
        end
    end
    return _timegan_latent(model.parameters, noise, teacher)
end

_timegan_logsigmoid(value) = value >= 0 ? -log1p(exp(-value)) : value - log1p(exp(value))

function _timegan_losses(parameters, observations, noise)
    latent = _timegan_sequence_map(parameters.embedding, observations)
    reconstructed = _timegan_sequence_map(parameters.recovery, latent)
    synthetic = _timegan_latent(parameters, noise, nothing)
    teacher = _timegan_latent(parameters, noise, latent)
    real_logits = _timegan_sequence_map(parameters.discriminator, latent; sigmoid = false)
    fake_logits = _timegan_sequence_map(parameters.discriminator, synthetic; sigmoid = false)
    count = size(observations, 1)
    reconstruction = sum(abs2, observations .- reconstructed) / count
    supervised = sum(abs2, latent .- teacher) / count
    real_log = sum(_timegan_logsigmoid.(real_logits)) / count
    fake_log = sum(_timegan_logsigmoid.(-fake_logits)) / count
    return (reconstruction = reconstruction, supervised = supervised, discriminator = -real_log - fake_log, generator_adversarial = fake_log)
end

"""
Return the public reconstruction, supervision, discriminator and minimax
generator losses, averaged over windows and summed over time/features.
Supervision includes the initial latent. No separate supervisor network or
moment-matching extension is introduced.
"""
function timegan_losses(model::TimeGAN, observations, noise)
    Zygote.ignore() do
        _timegan_check(model, observations, "observations", model.features)
        _timegan_check(model, noise, "noise", model.hidden_size)
        all(value -> 0 <= value <= 1, observations) || throw(ArgumentError("Training observations must be scaled to [0,1]"))
        size(observations)[1:2] == size(noise)[1:2] || throw(ArgumentError("Observations and noise need matching windows/times"))
    end
    return _timegan_losses(model.parameters, observations, noise)
end

"""Return detached scaled windows from fresh local Gaussian noise or supplied primitive draws."""
function sample_timegan(model::TimeGAN, windows, length; seed, normal_draws = nothing)
    return Zygote.ignore() do
        count = _timegan_count(windows, "windows")
        times = _timegan_count(length, "length")
        rng = _timegan_rng(seed)
        noise = isnothing(normal_draws) ? randn(rng, count, times, model.hidden_size) : normal_draws
        size(noise) == (count, times, model.hidden_size) || throw(ArgumentError("Supplied normal draws must match the requested window shape"))
        _timegan_check(model, noise, "noise", model.hidden_size)
        _timegan_sequence_map(model.parameters.recovery, _timegan_latent(model.parameters, noise, nothing))
    end
end
