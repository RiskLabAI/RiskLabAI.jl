function _pde_time_dependent_parameters(rng, inputs, hidden, outputs; monte_carlo = false)
    widths = (inputs, hidden...)
    return (layers = Tuple(_pde_dense_parameters(rng, widths[index], widths[index+1]) for index in 1:length(hidden)),
        time_layers = Tuple(_pde_feedforward_parameters(rng, (4, 100, 150, 200, 300, 200, width)) for width in (monte_carlo ? (widths[1:end-1]..., outputs) : widths[1:end-1])),
        linear = _pde_dense_parameters(rng, last(hidden), outputs),
        normalization = monte_carlo ? () : Tuple((scale = ones(width), offset = zeros(width)) for width in widths[1:end-1]))
end

function _pde_hidden_sizes(values)
    values isa Union{Tuple, AbstractVector} && !isempty(values) || throw(ArgumentError("Supply at least one hidden layer"))
    return Tuple(_nse_count(value, "hidden layer size") for value in values)
end

function _pde_batchnorm(parameters, running, inputs, training)
    if training
        Zygote.ignore() do
            size(inputs, 1) > 1 || throw(ArgumentError("Training batch normalization requires more than one observation"))
        end
        center = Statistics.mean(inputs; dims = 1)
        variance = Statistics.mean((inputs .- center).^2; dims = 1)
        next_state = Zygote.ignore() do
            n = size(inputs, 1)
            (mean = 0.99 .* running.mean .+ 0.01 .* vec(center),
                variance = 0.99 .* running.variance .+ 0.01 .* vec(variance) .* (n / (n - 1)), batches = running.batches + 1)
        end
    else
        center, variance = running.mean', running.variance'
        next_state = running
    end
    normalized = (inputs .- center) ./ sqrt.(variance .+ 1e-6)
    return normalized .* parameters.scale' .+ parameters.offset', next_state
end

"""
Time-dependent sine network with the public active batch normalization,
polynomial/exponential time features and multiplicative time weights.
`training=true` uses batch moments and updates running mean/unbiased variance
with momentum 0.01 and epsilon 1e-6. Set `model.training=false` for fixed
running-statistic evaluation. Unused normalization and output-time modules
from the Python constructor are omitted from the native parameter tree.
"""
mutable struct TimeDependentNetwork <: AbstractPDENetwork
    input_dim::Int
    hidden_sizes::Tuple
    output_dim::Int
    training::Bool
    parameters::NamedTuple
    state::Tuple
    function TimeDependentNetwork(input_dim, hidden_sizes, output_dim; seed = 0, parameters = nothing)
        inputs, outputs = _nse_count(input_dim, "input_dim"), _nse_count(output_dim, "output_dim")
        hidden = _pde_hidden_sizes(hidden_sizes)
        template = _pde_time_dependent_parameters(_nse_rng(seed), inputs, hidden, outputs)
        state = Tuple((mean = zeros(width), variance = ones(width), batches = 0) for width in (inputs, hidden[1:end-1]...))
        new(inputs, hidden, outputs, true, _pde_parameter_copy(parameters, template), state)
    end
end

function _pde_dtnn_forward(model, parameters, time, inputs; update_state = true)
    time_features = hcat(time, time.^2, time.^3, exp.(time))
    features, next_states = inputs, ()
    for index in eachindex(parameters.layers)
        normalized, next_state = _pde_batchnorm(parameters.normalization[index], model.state[index], features, model.training)
        time_weights = _pde_time_forward(parameters.time_layers[index], time_features)
        projected = _pde_dense(parameters.layers[index], normalized .* (1 .+ time_weights))
        features = sin.(projected)
        next_states = (next_states..., next_state)
    end
    if update_state
        Zygote.ignore() do
            model.state = next_states
        end
    end
    return _pde_dense(parameters.linear, features)
end

function (model::TimeDependentNetwork)(time, inputs)
    Zygote.ignore() do
        _pde_check_array(time, 2, 1)
        _pde_check_array(inputs, 2, model.input_dim)
        size(time, 1) == size(inputs, 1) || throw(ArgumentError("Time and feature batches must agree"))
    end
    return _pde_dtnn_forward(model, model.parameters, time, inputs)
end

"""
Public time-dependent Monte Carlo control network with ReLU hidden layers.
The supplied `y_mc` gradient estimate enters exactly as
`sigma*x*network_output + (1-time_weight)*y_mc`. No gradient estimator or
pricing measure is inferred. Inactive Python batch-normalization modules are
omitted. Input/output dimensions must support the public elementwise product.
"""
mutable struct TimeDependentNetworkMonteCarlo <: AbstractPDENetwork
    input_dim::Int
    hidden_sizes::Tuple
    output_dim::Int
    sigma::Float64
    parameters::NamedTuple
    function TimeDependentNetworkMonteCarlo(input_dim, hidden_sizes, output_dim, sigma; seed = 0, parameters = nothing)
        inputs, outputs = _nse_count(input_dim, "input_dim"), _nse_count(output_dim, "output_dim")
        inputs == outputs || min(inputs, outputs) == 1 || throw(ArgumentError("Input and output dimensions do not support the public elementwise control formula"))
        hidden, volatility = _pde_hidden_sizes(hidden_sizes), _nse_real(sigma, "sigma")
        template = _pde_time_dependent_parameters(_nse_rng(seed), inputs, hidden, outputs; monte_carlo = true)
        new(inputs, hidden, outputs, volatility, _pde_parameter_copy(parameters, template))
    end
end

function _pde_monte_carlo_forward(model, parameters, time, inputs, y_mc)
    time_features = hcat(time, time.^2, time.^3, exp.(time))
    features = copy(inputs)
    for index in eachindex(parameters.layers)
        time_weights = _pde_time_forward(parameters.time_layers[index], time_features)
        features = _pde_relu.(_pde_dense(parameters.layers[index], features .* (1 .+ time_weights)))
    end
    output_weight = _pde_time_forward(parameters.time_layers[end], time_features)
    return model.sigma .* inputs .* _pde_dense(parameters.linear, features) .+ (1 .- output_weight) .* y_mc
end

function (model::TimeDependentNetworkMonteCarlo)(time, inputs, y_mc)
    Zygote.ignore() do
        _pde_check_array(time, 2, 1)
        _pde_check_array(inputs, 2, model.input_dim)
        _pde_check_array(y_mc, 2, max(model.input_dim, model.output_dim))
        size(time, 1) == size(inputs, 1) == size(y_mc, 1) || throw(ArgumentError("Time, features and Monte Carlo gradients need matching batch sizes"))
    end
    return _pde_monte_carlo_forward(model, model.parameters, time, inputs, y_mc)
end

"""
    initialize_weights(layer_parameters; seed=0, normal_draws=nothing)

Mutate one native dense parameter leaf `(weight,bias)` to public normal
weights with standard deviation 0.1 and zero bias. Supplied primitive normal
draws support exact comparisons. Other parameter containers are unchanged,
matching the Python callback's restriction to a Linear module. A bias-free
leaf is rejected, preserving the public callback's unsupported case.
"""
function initialize_weights(parameters; seed = 0, normal_draws = nothing)
    parameters isa NamedTuple && haskey(parameters, :weight) || return nothing
    keys(parameters) == (:weight, :bias) || throw(ArgumentError("The public initializer requires a dense layer with bias"))
    parameters.weight isa AbstractMatrix{Float64} && parameters.bias isa AbstractVector{Float64} && size(parameters.weight, 1) == length(parameters.bias) || throw(ArgumentError("Dense weight and bias shapes must agree and use Float64"))
    rng = _nse_rng(seed)
    draws = isnothing(normal_draws) ? randn(rng, size(parameters.weight)) : normal_draws
    draws isa AbstractMatrix{Float64} && size(draws) == size(parameters.weight) && all(isfinite, draws) || throw(ArgumentError("Primitive normal draws must match the dense weight shape"))
    parameters.weight .= 0.1 .* draws
    parameters.bias .= 0
    return nothing
end
