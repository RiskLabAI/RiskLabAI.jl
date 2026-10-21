function _pde_attention_dimensions(dim_q, dim_k, dim_v, num_heads, ln)
    dimensions = Tuple(_nse_count(value, "attention dimension") for value in (dim_q, dim_k, dim_v, num_heads))
    dimensions[3] % dimensions[4] == 0 && ln isa Bool || throw(ArgumentError("Value dimensions must divide into whole heads, and ln must be Boolean"))
    return dimensions
end

function _pde_mab_parameters(rng, query, key, value, ln)
    norm = () -> (scale = ones(value), offset = zeros(value))
    return (fc_q = _pde_dense_parameters(rng, query, value), fc_k = _pde_dense_parameters(rng, key, value),
        fc_v = _pde_dense_parameters(rng, key, value), fc_o = _pde_dense_parameters(rng, value, value),
        norm0 = ln ? norm() : (;), norm1 = ln ? norm() : (;))
end

function _pde_layernorm(parameters, inputs)
    centered = inputs .- Statistics.mean(inputs; dims = 3)
    variance = Statistics.mean(centered.^2; dims = 3)
    return centered ./ sqrt.(variance .+ 1e-5) .* reshape(parameters.scale, 1, 1, :) .+ reshape(parameters.offset, 1, 1, :)
end

function _pde_attention(parameters, queries, keys, heads, ln)
    q, k, v = _pde_dense(parameters.fc_q, queries), _pde_dense(parameters.fc_k, keys), _pde_dense(parameters.fc_v, keys)
    dimension, query_count = size(q, 3), size(q, 2)
    width = dimension ÷ heads
    batches = map(1:size(q, 1)) do batch
        foldl(1:heads; init = zeros(query_count, 0)) do prior, head
            indices = (head-1)*width+1:head*width
            observed = q[batch, :, indices]
            scores = observed * k[batch, :, indices]' ./ sqrt(dimension)
            anchor = Zygote.ignore() do
                maximum(scores; dims = 2)
            end
            weights = exp.(scores .- anchor)
            attention = weights ./ sum(weights; dims = 2)
            hcat(prior, observed + attention * v[batch, :, indices])
        end
    end
    flattened = foldl((prior, value) -> hcat(prior, vec(value)), batches; init = zeros(query_count * dimension, 0))
    output = permutedims(reshape(flattened, query_count, dimension, size(q, 1)), (3, 1, 2))
    output = ln ? _pde_layernorm(parameters.norm0, output) : output
    output = output .+ _pde_relu.(_pde_dense(parameters.fc_o, output))
    return ln ? _pde_layernorm(parameters.norm1, output) : output
end

"""
Multi-head attention with the public `sqrt(dim_v)` score scaling, projected
query residual, ReLU output residual and optional two layer-normalization
operations. Inputs have `(batch,set,features)` dimensions. Native dimensions
must divide into complete heads. Shared Float64 parameter arrays are copied.
"""
mutable struct MAB <: AbstractPDENetwork
    dim_q::Int
    dim_k::Int
    dim_v::Int
    num_heads::Int
    ln::Bool
    parameters::NamedTuple
    function MAB(dim_q, dim_k, dim_v, num_heads; ln = false, seed = 0, parameters = nothing)
        query, key, value, heads = _pde_attention_dimensions(dim_q, dim_k, dim_v, num_heads, ln)
        new(query, key, value, heads, ln, _pde_parameter_copy(parameters, _pde_mab_parameters(_nse_rng(seed), query, key, value, ln)))
    end
end

function (model::MAB)(queries, keys)
    Zygote.ignore() do
        _pde_check_array(queries, 3, model.dim_q)
        _pde_check_array(keys, 3, model.dim_k)
        size(queries, 1) == size(keys, 1) || throw(ArgumentError("Attention query and key batch sizes must agree"))
    end
    return _pde_attention(model.parameters, queries, keys, model.num_heads, model.ln)
end

"""Public self-attention block, using the same set as attention queries and keys."""
mutable struct SAB <: AbstractPDENetwork
    dim_in::Int
    dim_out::Int
    num_heads::Int
    ln::Bool
    parameters::NamedTuple
    function SAB(dim_in, dim_out, num_heads; ln = false, seed = 0, parameters = nothing)
        inputs, _, outputs, heads = _pde_attention_dimensions(dim_in, dim_in, dim_out, num_heads, ln)
        new(inputs, outputs, heads, ln, _pde_parameter_copy(parameters, (mab = _pde_mab_parameters(_nse_rng(seed), inputs, inputs, outputs, ln),)))
    end
end

function (model::SAB)(inputs)
    Zygote.ignore() do
        _pde_check_array(inputs, 3, model.dim_in)
    end
    return _pde_attention(model.parameters.mab, inputs, inputs, model.num_heads, model.ln)
end

function _pde_inducing(rng, count, dimension)
    bound = Float32(sqrt(6 / ((count + 1) * dimension)))
    return Float64.((2f0 .* rand(rng, Float32, 1, count, dimension) .- 1f0) .* bound)
end

"""Public induced self-attention with trainable Xavier-uniform inducing vectors and two attention blocks."""
mutable struct ISAB <: AbstractPDENetwork
    dim_in::Int
    dim_out::Int
    num_heads::Int
    num_inds::Int
    ln::Bool
    parameters::NamedTuple
    function ISAB(dim_in, dim_out, num_heads, num_inds; ln = false, seed = 0, parameters = nothing)
        inputs, _, outputs, heads = _pde_attention_dimensions(dim_in, dim_in, dim_out, num_heads, ln)
        count = _nse_count(num_inds, "num_inds")
        rng = _nse_rng(seed)
        template = (inducing = _pde_inducing(rng, count, outputs), mab0 = _pde_mab_parameters(rng, outputs, inputs, outputs, ln), mab1 = _pde_mab_parameters(rng, inputs, outputs, outputs, ln))
        new(inputs, outputs, heads, count, ln, _pde_parameter_copy(parameters, template))
    end
end

function _pde_isab_forward(parameters, inputs, heads, ln)
    inducing = repeat(parameters.inducing; outer = (size(inputs, 1), 1, 1))
    hidden = _pde_attention(parameters.mab0, inducing, inputs, heads, ln)
    return _pde_attention(parameters.mab1, inputs, hidden, heads, ln)
end

function (model::ISAB)(inputs)
    Zygote.ignore() do
        _pde_check_array(inputs, 3, model.dim_in)
    end
    return _pde_isab_forward(model.parameters, inputs, model.num_heads, model.ln)
end

function _pde_pma_parameters(rng, dimension, seeds, ln)
    return (seeds = _pde_inducing(rng, seeds, dimension), mab = _pde_mab_parameters(rng, dimension, dimension, dimension, ln))
end

function _pde_pma_forward(parameters, inputs, heads, ln)
    seeds = repeat(parameters.seeds; outer = (size(inputs, 1), 1, 1))
    return _pde_attention(parameters.mab, seeds, inputs, heads, ln)
end

"""Public pooling attention with trainable Xavier-uniform seed vectors; output shape is `(batch,num_seeds,dim)`."""
mutable struct PMA <: AbstractPDENetwork
    dim::Int
    num_heads::Int
    num_seeds::Int
    ln::Bool
    parameters::NamedTuple
    function PMA(dim, num_heads, num_seeds; ln = false, seed = 0, parameters = nothing)
        inputs, _, _, heads = _pde_attention_dimensions(dim, dim, dim, num_heads, ln)
        count = _nse_count(num_seeds, "num_seeds")
        new(inputs, heads, count, ln, _pde_parameter_copy(parameters, _pde_pma_parameters(_nse_rng(seed), inputs, count, ln)))
    end
end

function (model::PMA)(inputs)
    Zygote.ignore() do
        _pde_check_array(inputs, 3, model.dim)
    end
    return _pde_pma_forward(model.parameters, inputs, model.num_heads, model.ln)
end

"""Public five-layer feature-centered set network, one-seed attention pool and four learned time-conditioned output layers."""
mutable struct DeepTimeSetTransformer <: AbstractPDENetwork
    input_dim::Int
    parameters::NamedTuple
    function DeepTimeSetTransformer(input_dim; seed = 0, parameters = nothing)
        inputs = _nse_count(input_dim, "input_dim")
        rng = _nse_rng(seed)
        template = (layers = _pde_feedforward_parameters(rng, (inputs, 32, 32, 32, 32, 32)), regressor = _pde_pma_parameters(rng, 32, 1, false),
            time_layers = Tuple(_pde_time_set_parameters(rng, 32, outputs) for outputs in (32, 32, 32, 1)))
        new(inputs, _pde_parameter_copy(parameters, template))
    end
end

function _pde_deep_set_forward(parameters, time, inputs)
    features = inputs
    for (index, layer) in enumerate(parameters.layers.layers)
        projected = _pde_relu.(_pde_dense(layer, features))
        features = index < 5 ? projected .- Statistics.mean(projected; dims = 3) : projected
    end
    output = dropdims(_pde_pma_forward(parameters.regressor, features, 4, false); dims = 2)
    for (index, layer) in enumerate(parameters.time_layers)
        projected = _pde_time_set_forward(layer, time, output)
        output = index < 4 ? _pde_relu.(projected) : projected
    end
    return output
end

function (model::DeepTimeSetTransformer)(time, inputs)
    Zygote.ignore() do
        _pde_check_array(time, 2, 1)
        _pde_check_array(inputs, 3, model.input_dim)
        size(time, 1) == size(inputs, 1) || throw(ArgumentError("Time and set batch sizes must agree"))
    end
    return _pde_deep_set_forward(model.parameters, time, inputs)
end
