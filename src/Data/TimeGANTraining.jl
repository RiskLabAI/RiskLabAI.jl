_timegan_allfinite(value::NamedTuple) = all(_timegan_allfinite, values(value))
_timegan_allfinite(value) = value isa AbstractArray && all(isfinite, value)

function _timegan_groups(parameters)
    return ((embedding = parameters.embedding, recovery = parameters.recovery),
        (generator_cell = parameters.generator_cell, generator_output = parameters.generator_output),
        (discriminator = parameters.discriminator,))
end

function _timegan_training_draw(draw, data, batch_size, hidden)
    draw isa NamedTuple && keys(draw) == (:indices, :noise) || throw(ArgumentError("Each supplied training draw must name indices and noise"))
    indices, noise = draw.indices, draw.noise
    indices isa AbstractVector && length(indices) == batch_size && all(index -> index isa Integer && !(index isa Bool) && 1 <= index <= size(data, 1), indices) || throw(ArgumentError("Supplied minibatch indices must be valid one-based observation positions"))
    noise isa AbstractArray{Float64, 3} && size(noise) == (batch_size, size(data, 2), hidden) && all(isfinite, noise) || throw(ArgumentError("Supplied Gaussian training draws must match batch/time/hidden dimensions"))
    return Int.(indices), noise
end

"""
    fit_timegan(model, training_windows; embedding_steps, supervised_steps,
        joint_steps, batch_size, learning_rate=0.001, embedding_weight=1.0,
        generator_weight=10.0, seed=0, training_draws=nothing)

Mutate the temporal-only model through reconstruction, teacher-forced generator
training and joint training. Positive step counts are fixed in advance.
Joint updates occur in E/R, G, D order, recomputing losses after each update.
The generator uses the public minimax fake-data term plus supervision. Adam
states persist across phases. Histories record losses before the last group
update of each step, matching the public contract.

Only scaled training windows enter optimization. Optional per-step named
tuples `(indices,noise)` supply one-based minibatch positions and Gaussian
primitive draws for cross-language validation; otherwise a local RNG is used.
Training losses do not establish generation quality or choose stopping time.
"""
function fit_timegan(model::TimeGAN, training_windows; embedding_steps, supervised_steps,
    joint_steps, batch_size, learning_rate = 0.001, embedding_weight = 1.0,
    generator_weight = 10.0, seed = 0, training_draws = nothing)
    counts = (_timegan_count(embedding_steps, "embedding_steps"), _timegan_count(supervised_steps, "supervised_steps"), _timegan_count(joint_steps, "joint_steps"))
    batch = _timegan_count(batch_size, "batch_size")
    rate = _timegan_real(learning_rate, "learning_rate"; minimum = 0, strict = true)
    embedding_scale = _timegan_real(embedding_weight, "embedding_weight"; minimum = 0)
    generator_scale = _timegan_real(generator_weight, "generator_weight"; minimum = 0)
    data = _timegan_array(training_windows, "training_windows", 3)
    _timegan_check(model, data, "training_windows", model.features)
    all(value -> 0 <= value <= 1, data) || throw(ArgumentError("Training observations must be scaled to [0,1]"))
    rng = _timegan_rng(seed)
    if !isnothing(training_draws)
        training_draws isa Union{Tuple, AbstractVector} && length(training_draws) == sum(counts) || throw(ArgumentError("Supply exactly one training draw for each declared step"))
        foreach(draw -> _timegan_training_draw(draw, data, batch, model.hidden_size), training_draws)
    end
    groups = _timegan_groups(model.parameters)
    states = Any[Optimisers.setup(Optimisers.Adam(rate, (0.9, 0.999), 1e-8), group) for group in groups]
    history = NamedTuple{(:phase, :reconstruction, :supervised, :discriminator, :generator_adversarial), Tuple{String, Float64, Float64, Float64, Float64}}[]
    draw_index = 0
    for (phase, count) in enumerate(counts)
        for _ in 1:count
            draw_index += 1
            indices, noise = isnothing(training_draws) ? (rand(rng, 1:size(data, 1), batch), randn(rng, batch, size(data, 2), model.hidden_size)) : _timegan_training_draw(training_draws[draw_index], data, batch, model.hidden_size)
            observed = data[indices, :, :]
            updates = phase == 1 ? (1,) : phase == 2 ? (2,) : (1, 2, 3)
            final_losses = nothing
            for group in updates
                parameters = model.parameters
                selected = _timegan_groups(parameters)[group]
                function objective(candidate)
                    losses = _timegan_losses(merge(parameters, candidate), observed, noise)
                    if group == 1
                        return losses.reconstruction + (phase == 3 ? embedding_scale * losses.supervised : 0.0)
                    elseif group == 2
                        return generator_scale * losses.supervised + (phase == 3 ? losses.generator_adversarial : 0.0)
                    end
                    return losses.discriminator
                end
                final_losses = _timegan_losses(parameters, observed, noise)
                loss, pullback = Zygote.pullback(objective, selected)
                gradient = only(pullback(one(loss)))
                isfinite(loss) && _timegan_allfinite(gradient) || error("TimeGAN objective or gradients became nonfinite")
                states[group], fitted = Optimisers.update(states[group], selected, gradient)
                _timegan_allfinite(fitted) || error("TimeGAN parameters became nonfinite")
                model.parameters = merge(parameters, fitted)
            end
            push!(history, merge((phase = ("embedding", "supervised", "joint")[phase],), final_losses))
        end
    end
    return history
end
