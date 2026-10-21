abstract type AbstractPDENetwork end

function _pde_dense_parameters(rng, inputs, outputs; bias = true)
    bound = Float32(1 / sqrt(inputs))
    weight = Float64.((2f0 .* rand(rng, Float32, outputs, inputs) .- 1f0) .* bound)
    return bias ? (weight = weight, bias = Float64.((2f0 .* rand(rng, Float32, outputs) .- 1f0) .* bound)) : (weight = weight,)
end

function _pde_parameter_copy(supplied, template)
    isnothing(supplied) && return template
    if template isa NamedTuple
        supplied isa NamedTuple && keys(supplied) == keys(template) || throw(ArgumentError("Network parameters must match the named architecture"))
        return NamedTuple{keys(template)}(Tuple(_pde_parameter_copy(getproperty(supplied, name), getproperty(template, name)) for name in keys(template)))
    elseif template isa Tuple
        supplied isa Tuple && length(supplied) == length(template) || throw(ArgumentError("Network layer counts must match the architecture"))
        return map(_pde_parameter_copy, supplied, template)
    end
    supplied isa AbstractArray{Float64} && size(supplied) == size(template) && all(isfinite, supplied) || throw(ArgumentError("Network parameters must be finite Float64 arrays with the expected shapes"))
    return copy(supplied)
end

function _pde_check_array(value, dimensions, features)
    value isa AbstractArray{Float64} && ndims(value) == dimensions && !isempty(value) && size(value, dimensions) == features && all(isfinite, value) || throw(ArgumentError("Inputs must be finite nonempty Float64 arrays with $dimensions dimensions and $features features"))
    return nothing
end

function _pde_dense(parameters, inputs::AbstractMatrix)
    result = inputs * parameters.weight'
    return haskey(parameters, :bias) ? result .+ parameters.bias' : result
end

function _pde_dense(parameters, inputs::AbstractArray{<:Real, 3})
    flattened = reshape(permutedims(inputs, (3, 1, 2)), size(inputs, 3), :)
    projected = parameters.weight * flattened
    projected = haskey(parameters, :bias) ? projected .+ parameters.bias : projected
    return permutedims(reshape(projected, size(projected, 1), size(inputs, 1), size(inputs, 2)), (2, 3, 1))
end

_pde_relu(value) = value > 0 ? value : zero(value)

function _pde_layer_sizes(sizes)
    sizes isa Union{Tuple, AbstractVector} && length(sizes) >= 2 || throw(ArgumentError("Supply at least input and output layer sizes"))
    return Tuple(_nse_count(value, "layer size") for value in sizes)
end

function _pde_feedforward_parameters(rng, sizes; bias = true)
    return (layers = Tuple(_pde_dense_parameters(rng, sizes[index], sizes[index+1]; bias) for index in 1:length(sizes)-1),)
end

"""Public linear-only PDE component; the unused Python batch-normalization module does not enter the native computation."""
mutable struct Net1 <: AbstractPDENetwork
    input_dim::Int
    output_dim::Int
    parameters::NamedTuple
    function Net1(input_dim, output_dim; seed = 0, parameters = nothing)
        inputs, outputs = _nse_count(input_dim, "input_dim"), _nse_count(output_dim, "output_dim")
        template = (layer = _pde_dense_parameters(_nse_rng(seed), inputs, outputs),)
        new(inputs, outputs, _pde_parameter_copy(parameters, template))
    end
end

function (model::Net1)(inputs)
    Zygote.ignore() do
        _pde_check_array(inputs, 2, model.input_dim)
    end
    return _pde_dense(model.parameters.layer, inputs)
end

"""Six-layer public time network with widths 4,100,150,200,300,200,output_dim and tanh after every layer."""
mutable struct TimeNet <: AbstractPDENetwork
    output_dim::Int
    parameters::NamedTuple
    function TimeNet(output_dim; seed = 0, parameters = nothing)
        outputs = _nse_count(output_dim, "output_dim")
        template = _pde_feedforward_parameters(_nse_rng(seed), (4, 100, 150, 200, 300, 200, outputs))
        new(outputs, _pde_parameter_copy(parameters, template))
    end
end

function (model::TimeNet)(inputs)
    Zygote.ignore() do
        _pde_check_array(inputs, 2, 4)
    end
    return _pde_time_forward(model.parameters, inputs)
end

function _pde_time_forward(parameters, inputs)
    output = inputs
    for layer in parameters.layers
        output = tanh.(_pde_dense(layer, output))
    end
    return output
end

"""Feed-forward FBSNN with sine after every hidden dense layer and a linear final layer."""
mutable struct FBSNNNetwork <: AbstractPDENetwork
    layer_sizes::Tuple
    parameters::NamedTuple
    function FBSNNNetwork(layer_sizes; seed = 0, parameters = nothing)
        sizes = _pde_layer_sizes(layer_sizes)
        new(sizes, _pde_parameter_copy(parameters, _pde_feedforward_parameters(_nse_rng(seed), sizes)))
    end
end

"""Public bias-free Deep BSDE network with ReLU hidden layers and linear output; inactive batch normalization is omitted."""
mutable struct DeepBSDE <: AbstractPDENetwork
    layer_sizes::Tuple
    parameters::NamedTuple
    function DeepBSDE(layer_sizes; seed = 0, parameters = nothing)
        sizes = _pde_layer_sizes(layer_sizes)
        new(sizes, _pde_parameter_copy(parameters, _pde_feedforward_parameters(_nse_rng(seed), sizes; bias = false)))
    end
end

function _pde_hidden_forward(parameters, inputs, activation)
    count = length(parameters.layers)
    output = inputs
    for (index, layer) in enumerate(parameters.layers)
        output = _pde_dense(layer, output)
        output = index < count ? activation.(output) : output
    end
    return output
end

function (model::Union{FBSNNNetwork, DeepBSDE})(inputs)
    Zygote.ignore() do
        _pde_check_array(inputs, 2, first(model.layer_sizes))
    end
    return _pde_hidden_forward(model.parameters, inputs, model isa FBSNNNetwork ? sin : _pde_relu)
end

function _pde_time_set_parameters(rng, inputs, outputs)
    return (feature = _pde_dense_parameters(rng, inputs, outputs), time = _pde_feedforward_parameters(rng, (1, 10, 10, 10, outputs)))
end

function _pde_time_set_forward(parameters, time, inputs)
    time_features = _pde_hidden_forward(parameters.time, time, _pde_relu)
    return exp.(time_features) .* _pde_dense(parameters.feature, inputs)
end

"""Public time-conditioned feature map with a 1,10,10,10,output_dim ReLU time network and exponential multiplicative modulation."""
mutable struct TimeNetForSet <: AbstractPDENetwork
    input_dim::Int
    output_dim::Int
    parameters::NamedTuple
    function TimeNetForSet(input_dim = 1, output_dim = 64; seed = 0, parameters = nothing)
        inputs, outputs = _nse_count(input_dim, "input_dim"), _nse_count(output_dim, "output_dim")
        new(inputs, outputs, _pde_parameter_copy(parameters, _pde_time_set_parameters(_nse_rng(seed), inputs, outputs)))
    end
end

function (model::TimeNetForSet)(time, inputs)
    Zygote.ignore() do
        _pde_check_array(time, 2, 1)
        _pde_check_array(inputs, 2, model.input_dim)
        size(time, 1) == size(inputs, 1) || throw(ArgumentError("Time and feature batch sizes must agree"))
    end
    return _pde_time_set_forward(model.parameters, time, inputs)
end
