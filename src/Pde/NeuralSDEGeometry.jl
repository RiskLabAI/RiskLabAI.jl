import LinearAlgebra
import Statistics
import Random
import Zygote
import Optimisers
import JuMP
import Clarabel

function _nse_real(value, name; positive = false)
    value isa Real && !(value isa Bool) && isfinite(value) || throw(ArgumentError("$name must be finite and real"))
    result = Float64(value)
    isfinite(result) && (!positive || result > 0) || throw(ArgumentError("$name is outside its allowed range"))
    return result
end

function _nse_array(value, name, dimensions)
    value isa AbstractArray && ndims(value) == dimensions && !isempty(value) || throw(ArgumentError("$name must be a nonempty real array of dimension $dimensions"))
    return map(item -> _nse_real(item, name), value)
end

function _nse_count(value, name)
    value isa Integer && !(value isa Bool) && 0 < value <= typemax(Int) || throw(ArgumentError("$name must be a positive supported integer"))
    return Int(value)
end

function _nse_rng(seed)
    seed isa Integer && !(seed isa Bool) && typemin(Int64) <= seed <= typemax(UInt64) || throw(ArgumentError("seed must be in the supported integer range"))
    return Random.MersenneTwister(UInt64(mod(BigInt(seed), BigInt(1) << 64)))
end

"""
    normalized_call_grid_constraints(strikes, maturities)

Return `(matrix,bounds)` for weak normalized-call inequalities `matrix*c >=
bounds`. Flatten prices in maturity-major order. Conditions retain intrinsic
and upper bounds, vertical slopes in [-1,0], convexity including the known
strike-zero call value one, and calendar monotonicity. These finite-grid
inequalities alone do not certify attainable marginals or dynamic arbitrage
freedom, particularly at degenerate positive flat tails.
"""
function normalized_call_grid_constraints(strikes, maturities)
    strikes = _nse_array(strikes, "strikes", 1)
    maturities = _nse_array(maturities, "maturities", 1)
    length(strikes) >= 2 && all(>(0), strikes) && all(>(0), maturities) && all(>(0), diff(strikes)) && all(>(0), diff(maturities)) || throw(ArgumentError("Require increasing positive strikes and maturities, with at least two strikes"))
    n, m = length(strikes), length(maturities)
    identity = Matrix{Float64}(LinearAlgebra.I, n * m, n * m)
    rows, bounds = Vector{Float64}[], Float64[]
    for time in 1:m
        for strike in 1:n
            row = identity[(time-1)*n+strike, :]
            append!(rows, (row, -row))
            append!(bounds, (max(1 - strikes[strike], 0), -1))
        end
        slopes = [identity[(time-1)*n+1, :] / strikes[1]]
        constants = [-1 / strikes[1]]
        for strike in 1:n-1
            push!(slopes, (identity[(time-1)*n+strike+1, :] - identity[(time-1)*n+strike, :]) / (strikes[strike+1] - strikes[strike]))
            push!(constants, 0)
        end
        for (slope, constant) in zip(slopes, constants)
            append!(rows, (slope, -slope))
            append!(bounds, (-1 - constant, constant))
        end
        for strike in 1:n-1
            push!(rows, slopes[strike+1] - slopes[strike])
            push!(bounds, constants[strike] - constants[strike+1])
        end
        if time > 1
            for strike in 1:n
                push!(rows, identity[(time-1)*n+strike, :] - identity[(time-2)*n+strike, :])
                push!(bounds, 0)
            end
        end
    end
    return permutedims(hcat(rows...)), bounds
end

"""
    normalized_call_grid_witness(strikes, maturities, prices)

Construct finite mean-one marginals reproducing the supplied call grid.
Piecewise-linear curves join (0,1), the quotes and a common zero-price tail.
Reject positive flat tails. Round only tiny negative masses and check mass,
mean and repricing residuals at the public tolerances. This static witness
does not identify a unique interpolation or a dynamic pricing measure.
"""
function normalized_call_grid_witness(strikes, maturities, prices)
    matrix, bounds = normalized_call_grid_constraints(strikes, maturities)
    strikes = _nse_array(strikes, "strikes", 1)
    maturities = _nse_array(maturities, "maturities", 1)
    prices = _nse_array(prices, "prices", 2)
    size(prices) == (length(maturities), length(strikes)) || throw(ArgumentError("Prices must have shape (maturities,strikes)"))
    minimum(matrix * vec(permutedims(prices)) - bounds) >= -1e-12 || throw(ArgumentError("Prices violate normalized call-grid inequalities"))
    final_slope = (prices[:, end] - prices[:, end-1]) / (strikes[end] - strikes[end-1])
    positive_tail = prices[:, end] .> 0
    any(positive_tail .& (final_slope .>= 0)) && throw(ArgumentError("A positive flat tail has no attainable call-price witness"))
    tail_length = zeros(length(maturities))
    tail_length[positive_tail] .= prices[positive_tail, end] ./ -final_slope[positive_tail]
    tail = strikes[end] + max(1.0, 2maximum(tail_length))
    isfinite(tail) && tail > strikes[end] || throw(ArgumentError("Tail support is not numerically representable"))
    support = [0.0; strikes; tail]
    values = hcat(ones(length(maturities)), prices, zeros(length(maturities)))
    slopes = diff(values; dims = 2) ./ permutedims(diff(support))
    probabilities = hcat(1 .+ slopes[:, 1], diff(slopes; dims = 2), -slopes[:, end])
    minimum(probabilities) >= -1e-12 || throw(ArgumentError("Call slopes do not yield nonnegative probability masses"))
    probabilities = max.(probabilities, 0)
    probabilities ./= sum(probabilities; dims = 2)
    reconstructed = probabilities * max.(support .- strikes', 0)
    mean_error = maximum(abs, probabilities * support .- 1)
    price_error = maximum(abs, reconstructed - prices)
    isfinite(mean_error) && isfinite(price_error) && max(mean_error, price_error) <= 1e-10 || throw(ArgumentError("Probability witness fails mean or repricing validation"))
    return (support = support, probabilities = probabilities, maximum_mean_error = mean_error, maximum_price_error = price_error)
end

"""Solve the supplied affine price inequalities exactly for a bounded scalar state interval with nonempty interior."""
function affine_price_state_interval(offset, loading, matrix, bounds)
    offset, loading = _nse_array(offset, "offset", 1), _nse_array(loading, "loading", 1)
    matrix, bounds = _nse_array(matrix, "matrix", 2), _nse_array(bounds, "bounds", 1)
    length(offset) == length(loading) && size(matrix) == (length(bounds), length(offset)) || throw(ArgumentError("Price basis and inequality shapes must agree"))
    slopes, right = matrix * loading, bounds - matrix * offset
    any((slopes .== 0) .& (right .> 0)) && throw(ArgumentError("Price constraints are infeasible for this basis"))
    lower = maximum(right[slopes .> 0] ./ slopes[slopes .> 0]; init = -Inf)
    upper = minimum(right[slopes .< 0] ./ slopes[slopes .< 0]; init = Inf)
    isfinite(lower) && isfinite(upper) && lower < upper || throw(ArgumentError("The affine price state must have a bounded nonempty interior"))
    return lower, upper
end

function _nse_bounded_polytope(normals, bounds)
    dimension = size(normals, 2)
    for coordinate in 1:dimension, direction in (-1.0, 1.0)
        model = JuMP.Model(Clarabel.Optimizer)
        JuMP.set_silent(model)
        for name in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
            JuMP.set_optimizer_attribute(model, name, 1e-10)
        end
        JuMP.@variable(model, state[1:dimension])
        JuMP.@constraint(model, normals * state .>= bounds)
        JuMP.@objective(model, Min, direction * state[coordinate])
        JuMP.optimize!(model)
        JuMP.termination_status(model) == JuMP.MOI.OPTIMAL && JuMP.has_values(model) || throw(ArgumentError("The supplied polytope must be bounded and numerically feasible"))
        optimum = JuMP.value.(state)
        tolerance = 1e-8 * max(1, maximum(abs, bounds), maximum(abs, optimum))
        all(isfinite, optimum) && minimum(normals * optimum - bounds) >= -tolerance || throw(ArgumentError("The polytope bounding solution is not verified"))
    end
    return nothing
end

"""
    hjm_drift_residual(price_basis, drift, diffusion, required_price_drift)

Solve the supplied finite-grid HJM linear equation by minimum-norm least
squares. Return risk prices, residuals, squared residual, numerical rank and
singular values. Required price drift is independently supplied by the caller;
even zero residual does not verify derivatives or an equivalent martingale
measure. SVD rank uses the public dimension-scaled machine-epsilon threshold.
"""
function hjm_drift_residual(price_basis, drift, diffusion, required_price_drift)
    basis, diffusion = _nse_array(price_basis, "price_basis", 2), _nse_array(diffusion, "diffusion", 2)
    drift, required = _nse_array(drift, "drift", 1), _nse_array(required_price_drift, "required_price_drift", 1)
    size(basis) == (length(required), length(drift)) && size(diffusion) == (length(drift), length(drift)) || throw(ArgumentError("Price and state drift/diffusion dimensions must agree"))
    operator, rhs = basis * diffusion, basis * drift - required
    decomposition = LinearAlgebra.svd(operator)
    threshold = maximum(size(operator)) * eps(Float64) * maximum(decomposition.S)
    active = decomposition.S .> threshold
    coefficients = decomposition.U' * rhs
    coefficients = [active[index] ? coefficients[index] / decomposition.S[index] : 0.0 for index in eachindex(coefficients)]
    risk_price = decomposition.V * coefficients
    residual = rhs - operator * risk_price
    return (market_price_of_risk = risk_price, residual = residual, squared_residual = sum(abs2, residual), rank = count(active), singular_values = decomposition.S)
end
