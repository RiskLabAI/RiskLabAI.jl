abstract type AbstractConstrainedNeuralSDE end

function _nse_parameters(parameters, rng, inputs, hidden, outputs)
    shapes = (weight1 = (hidden, inputs), bias1 = (hidden,), weight2 = (outputs, hidden), bias2 = (outputs,))
    uniform(shape, fan_in) = Float64.((2f0 .* rand(rng, Float32, shape) .- 1f0) .* Float32(1 / sqrt(fan_in)))
    supplied = isnothing(parameters) ? (weight1 = uniform(shapes.weight1, inputs), bias1 = uniform(shapes.bias1, inputs),
        weight2 = uniform(shapes.weight2, hidden), bias2 = uniform(shapes.bias2, hidden)) : parameters
    supplied isa NamedTuple && keys(supplied) == keys(shapes) || throw(ArgumentError("Parameters must name weight1,bias1,weight2,bias2"))
    for name in keys(shapes)
        value = getproperty(supplied, name)
        value isa AbstractArray{Float64} && size(value) == getproperty(shapes, name) && all(isfinite, value) || throw(ArgumentError("Network parameters must have finite Float64 values and matching shapes"))
    end
    return deepcopy(supplied)
end

_nse_network(parameters, inputs) = tanh.(inputs * parameters.weight1' .+ parameters.bias1') * parameters.weight2' .+ parameters.bias2'
_nse_relu(value) = value > 0 ? value : zero(value)
_nse_nonnegative(value) = value < 0 ? zero(value) : value
_nse_softplus(value) = value > 20 ? value : value >= 0 ? value + log1p(exp(-value)) : log1p(exp(value))

"""
    IntervalNeuralSDE(lower, upper; drift_bound, diffusion_scale,
        hidden_size=12, seed=0, parameters=nothing)

Scalar-factor neural SDE on a supplied interval, with the public inward drift
correction and linear boundary-vanishing diffusion. The Float64 dense/tanh
network uses a local initializer; explicit shared weights are copied.
Diffusion is zero and drift points inward at the endpoints. Continuous-time
boundary conditions do not certify invariant Euler steps or a pricing measure.
"""
mutable struct IntervalNeuralSDE <: AbstractConstrainedNeuralSDE
    lower::Float64
    upper::Float64
    drift_bound::Float64
    diffusion_scale::Float64
    parameters::NamedTuple
    function IntervalNeuralSDE(lower, upper; drift_bound, diffusion_scale, hidden_size = 12, seed = 0, parameters = nothing)
        left, right = _nse_real(lower, "lower"), _nse_real(upper, "upper")
        left < right && isfinite(right - left) || throw(ArgumentError("Interval width must be finite and positive"))
        bound = _nse_real(drift_bound, "drift_bound"; positive = true)
        scale = _nse_real(diffusion_scale, "diffusion_scale"; positive = true)
        hidden = _nse_count(hidden_size, "hidden_size")
        new(left, right, bound, scale, _nse_parameters(parameters, _nse_rng(seed), 1, hidden, 2))
    end
end

function _nse_states(model::IntervalNeuralSDE, states)
    states isa AbstractVector{Float64} && !isempty(states) && all(isfinite, states) && all(value -> model.lower <= value <= model.upper, states) || throw(ArgumentError("States must be a finite admissible Float64 vector"))
    return nothing
end

function _nse_forward(model::IntervalNeuralSDE, parameters, states)
    width = model.upper - model.lower
    center = model.lower + width / 2
    raw = _nse_network(parameters, reshape((states .- center) ./ width, :, 1))
    raw_drift = model.drift_bound .* tanh.(raw[:, 1])
    raw_diffusion = _nse_softplus.(raw[:, 2])
    left, right = states .- model.lower, model.upper .- states
    rho = width / 4
    drift = raw_drift .+ _nse_relu.(-raw_drift .- model.drift_bound .* left ./ rho) .- _nse_relu.(raw_drift .- model.drift_bound .* right ./ rho)
    distance = min.(left, right)
    diffusion = raw_diffusion .* distance ./ (model.diffusion_scale .+ distance)
    return drift, diffusion
end

"""Return native drift and diffusion for a finite vector of states within the supplied interval."""
function (model::IntervalNeuralSDE)(states)
    Zygote.ignore() do
        _nse_states(model, states)
    end
    return _nse_forward(model, model.parameters, states)
end

"""
    PolytopeNeuralSDE(normals, bounds, interior; drift_bound,
        diffusion_scale, hidden_size=16, seed=0, parameters=nothing)

Hard-constrained neural dynamics on the supplied bounded polytope
`normals*state >= bounds`. Rows are normalized without changing feasibility.
A strictly interior point fixes the drift-correction geometry. Diffusion uses
the corrected fixed-coordinate transformation `Q*diag(g(distance))*Q'*L`,
with linear `g(d)=d/(diffusion_scale+d)` and nearest independent normals.
This differs from the paper's literal one-sided square-root transformation.
The constructor checks boundedness using the approved JuMP/Clarabel supplier.
It does not identify factors, calibrate prices or establish dynamic no-arbitrage.
"""
mutable struct PolytopeNeuralSDE <: AbstractConstrainedNeuralSDE
    dimension::Int
    normals::Matrix{Float64}
    bounds::Vector{Float64}
    interior::Vector{Float64}
    drift_bound::Float64
    diffusion_scale::Float64
    rho::Float64
    parameters::NamedTuple
    function PolytopeNeuralSDE(normals, bounds, interior; drift_bound, diffusion_scale, hidden_size = 16, seed = 0, parameters = nothing)
        normals, bounds, interior = _nse_array(normals, "normals", 2), _nse_array(bounds, "bounds", 1), _nse_array(interior, "interior", 1)
        dimension = length(interior)
        size(normals) == (length(bounds), dimension) || throw(ArgumentError("Normals, bounds and interior dimensions must agree"))
        lengths = [LinearAlgebra.norm(row) for row in eachrow(normals)]
        all(value -> isfinite(value) && value > 0, lengths) && LinearAlgebra.rank(normals) == dimension || throw(ArgumentError("Normals must be nonzero and span the state dimension"))
        normals, bounds = normals ./ lengths, bounds ./ lengths
        slack = normals * interior - bounds
        all(>(0), slack) || throw(ArgumentError("The interior must satisfy every inequality strictly"))
        _nse_bounded_polytope(normals, bounds)
        bound = _nse_real(drift_bound, "drift_bound"; positive = true)
        scale = _nse_real(diffusion_scale, "diffusion_scale"; positive = true)
        hidden = _nse_count(hidden_size, "hidden_size")
        outputs = dimension + dimension * (dimension + 1) ÷ 2
        new(dimension, normals, bounds, interior, bound, scale, minimum(slack) / 2,
            _nse_parameters(parameters, _nse_rng(seed), dimension, hidden, outputs))
    end
end

function _nse_states(model::PolytopeNeuralSDE, states)
    states isa AbstractMatrix{Float64} && size(states, 1) > 0 && size(states, 2) == model.dimension && all(isfinite, states) || throw(ArgumentError("States must be finite Float64 rows matching the polytope dimension"))
    distances = states * model.normals' .- model.bounds'
    tolerance = 1e-12 * max(1.0, maximum(abs, model.bounds), maximum(abs, states))
    minimum(distances) >= -tolerance || throw(ArgumentError("State lies outside the supplied polytope"))
    return nothing
end

function _nse_distances(model::PolytopeNeuralSDE, states)
    return _nse_nonnegative.(states * model.normals' .- model.bounds')
end

function _nse_frame(model, distances)
    return Zygote.ignore() do
        directions, faces = Vector{Float64}[], Int[]
        for face in sortperm(distances; alg = Base.Sort.MergeSort)
            residual = copy(model.normals[face, :])
            for _ in 1:2
                for direction in directions
                    residual -= LinearAlgebra.dot(residual, direction) * direction
                end
            end
            magnitude = LinearAlgebra.norm(residual)
            if magnitude > 1e-10
                push!(directions, residual / magnitude)
                push!(faces, face)
            end
            length(directions) == model.dimension && break
        end
        length(directions) == model.dimension || error("Boundary normals are numerically rank deficient")
        return hcat(directions...), faces
    end
end

function _nse_forward(model::PolytopeNeuralSDE, parameters, states)
    distances = _nse_distances(model, states)
    raw = _nse_network(parameters, (states .- model.interior') ./ model.rho)
    dimension = model.dimension
    drift = model.drift_bound .* tanh.(raw[:, 1:dimension])
    transformed = map(1:size(states, 1)) do index
        base_diffusion = [column > row ? 0.0 : row == column ? _nse_softplus(raw[index, dimension+row*(row-1)÷2+column]) : raw[index, dimension+row*(row-1)÷2+column] for row in 1:dimension, column in 1:dimension]
        basis, faces = _nse_frame(model, distances[index, :])
        selected = distances[index, faces]
        scales = selected ./ (model.diffusion_scale .+ selected)
        return (basis .* scales') * basis' * base_diffusion
    end
    flattened = foldl((accumulated, value) -> hcat(accumulated, vec(value)), transformed; init = zeros(dimension^2, 0))
    diffusion = permutedims(reshape(flattened, dimension, dimension, size(states, 1)), (3, 1, 2))
    corrections = model.interior' .- states
    normal_drift = drift * model.normals'
    inward_product = corrections * model.normals'
    active = distances .< model.rho
    denominator = ifelse.(active, inward_product, 1.0)
    tolerance = model.drift_bound * sqrt(dimension) .* distances ./ model.rho
    weights = ifelse.(active, _nse_relu.(-normal_drift .- tolerance) ./ denominator, 0.0)
    return drift .+ sum(weights; dims = 2) .* corrections, diffusion
end

"""
Return drift `(rows,factors)` and diffusion `(rows,factors,factors)`. Negative
distances within the public relative 1e-12 tolerance are evaluated as zero;
positive distances are never snapped to a face. Nearest-normal ties retain
input order, while the two-sided coordinate transformation preserves equal
shrinking scales under frame permutations.
"""
function (model::PolytopeNeuralSDE)(states)
    Zygote.ignore() do
        _nse_states(model, states)
    end
    return _nse_forward(model, model.parameters, states)
end

function _nse_likelihood(model::IntervalNeuralSDE, parameters, states, next_states, dt)
    drift, diffusion = _nse_forward(model, parameters, states)
    Zygote.ignore() do
        all(>(0), diffusion) || throw(ArgumentError("Likelihood requires strictly interior starts with nonsingular diffusion"))
    end
    variance = diffusion.^2 .* dt
    return Statistics.mean(log(2pi) .+ log.(variance) .+ (next_states .- states .- drift .* dt).^2 ./ variance) / 2
end

function _nse_likelihood(model::PolytopeNeuralSDE, parameters, states, next_states, dt)
    Zygote.ignore() do
        minimum(_nse_distances(model, states)) > 0 || throw(ArgumentError("Likelihood starts must be strictly interior"))
    end
    drift, diffusion = _nse_forward(model, parameters, states)
    losses = map(1:size(states, 1)) do index
        covariance = dt .* (diffusion[index, :, :] * diffusion[index, :, :]')
        factor = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(covariance, :L))
        residual = next_states[index, :] - states[index, :] - dt .* drift[index, :]
        standardized = factor.L \ residual
        log_determinant = 2sum(log, LinearAlgebra.diag(factor.L))
        return (model.dimension * log(2pi) + log_determinant + sum(abs2, standardized)) / 2
    end
    return Statistics.mean(losses)
end

"""
    negative_log_likelihood(model, states, next_states, dt)

Mean approximate Gaussian Euler transition NLL for supplied admissible
transitions with strictly interior starts. No variance jitter is added;
singular boundary starts and numerically nonpositive covariance fail. This
likelihood is not the exact constrained transition density.
"""
function negative_log_likelihood(model::AbstractConstrainedNeuralSDE, states, next_states, dt)
    timestep = Zygote.ignore() do
        _nse_states(model, states)
        _nse_states(model, next_states)
        size(states) == size(next_states) || throw(ArgumentError("Transition shapes must match"))
        _nse_real(dt, "dt"; positive = true)
    end
    loss = _nse_likelihood(model, model.parameters, states, next_states, timestep)
    Zygote.ignore() do
        isfinite(loss) || throw(ArgumentError("Transition likelihood is nonfinite"))
    end
    return loss
end
