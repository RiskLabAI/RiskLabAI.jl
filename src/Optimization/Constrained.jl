using JuMP
using Clarabel
using LinearAlgebra

function _portfolio_array(values, name, dimensions)
    values isa AbstractArray && ndims(values) == dimensions &&
        eltype(values) <: Real && !(eltype(values) <: Bool) && !isempty(values) ||
        throw(ArgumentError("$name must be a nonempty real array of dimension $dimensions"))
    converted = Float64.(values)
    all(isfinite, converted) || throw(ArgumentError("$name must contain finite values"))
    return converted
end

function _portfolio_scalar(value, name; minimum = -Inf)
    value isa Real && !(value isa Bool) || throw(ArgumentError("$name must be a real scalar"))
    converted = Float64(value)
    isfinite(converted) && converted >= minimum ||
        throw(ArgumentError("$name must be finite and at least $minimum"))
    return converted
end

function _portfolio_inputs(covariance, expected_returns, caps, return_floor)
    covariance = _portfolio_array(covariance, "covariance", 2)
    n = size(covariance, 1)
    size(covariance, 2) == n || throw(ArgumentError("covariance must be square"))
    magnitude = max(maximum(abs, covariance), floatmin(Float64))
    normalized = covariance / magnitude
    all(abs.(normalized - normalized') .<= 1e-12 .+ 1e-12 .* abs.(normalized')) ||
        throw(ArgumentError("covariance must be symmetric"))
    eigmin(Symmetric(normalized, :L)) >= -1e-12 * opnorm(normalized, 2) ||
        throw(ArgumentError("covariance must be positive semidefinite"))
    expected = isnothing(expected_returns) ? zeros(n) : _portfolio_array(expected_returns, "expected_returns", 1)
    cap = isnothing(caps) ? ones(n) : _portfolio_array(caps, "caps", 1)
    length(expected) == length(cap) == n && all(>=(0), cap) && sum(cap) >= 1 ||
        throw(ArgumentError("means and caps must match covariance and allow full investment"))
    floor = isnothing(return_floor) ? nothing : _portfolio_scalar(return_floor, "return_floor")
    !isnothing(floor) && isnothing(expected_returns) &&
        throw(ArgumentError("a return floor requires expected_returns"))
    return covariance, expected, cap, floor
end

function _solve_portfolio(covariance, expected, cap, floor; loading = nothing, radius = 0.0, risk_aversion = 1.0, minimum_variance = false)
    n = length(expected)
    scale = minimum_variance ? max(maximum(abs, covariance), floatmin(Float64)) :
        max(maximum(abs, expected), risk_aversion * maximum(abs, covariance), radius * norm(loading), floatmin(Float64))
    isfinite(scale) || throw(ArgumentError("objective scaling exceeds floating-point range"))
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    for name in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
        set_optimizer_attribute(model, name, 1e-10)
    end
    @variable(model, weights[1:n])
    @constraint(model, sum(weights) == 1)
    @constraint(model, weights .>= 0)
    @constraint(model, weights .<= cap)
    if !isnothing(floor)
        @constraint(model, sum(expected[i] * weights[i] for i in 1:n) >= floor)
    end
    variance = @expression(model, sum(covariance[i, j] * weights[i] * weights[j] for i in 1:n, j in 1:n))
    if minimum_variance
        @objective(model, Min, variance / scale)
    else
        penalty = 0.0
        if radius > 0
            @variable(model, uncertainty_norm)
            @constraint(model, [uncertainty_norm; loading' * weights] in SecondOrderCone())
            penalty = radius * uncertainty_norm
        end
        @objective(model, Min, (risk_aversion * variance / 2 - sum(expected[i] * weights[i] for i in 1:n) + penalty) / scale)
    end
    try
        optimize!(model)
    catch exception
        exception isa InterruptException && rethrow()
        error("portfolio solver failed: $(sprint(showerror, exception))")
    end
    status = termination_status(model)
    status in (JuMP.MOI.INFEASIBLE, JuMP.MOI.ALMOST_INFEASIBLE) &&
        throw(ArgumentError("portfolio constraints are infeasible"))
    status == JuMP.MOI.OPTIMAL && has_values(model) ||
        error("portfolio optimization is not verified: $status")
    result = value.(weights)
    nominal = dot(expected, result)
    if !all(isfinite, result) || abs(sum(result) - 1) > 1e-7 ||
       any(result .< -1e-7) || any(result .> cap .+ 1e-7) ||
       (!isnothing(floor) && nominal < floor - 1e-7 * max(1, abs(floor)))
        error("portfolio solver returned an infeasible result")
    end
    risk = dot(result, covariance * result)
    worst = isnothing(loading) ? nominal : nominal - radius * norm(loading' * result)
    objective = minimum_variance ? risk : worst - risk_aversion * risk / 2
    all(isfinite, (risk, nominal, worst, objective)) || error("portfolio diagnostics exceed floating-point range")
    return (weights = result, variance = risk, expected_return = nominal,
        worst_case_expected_return = worst, objective = objective, solver_status = "optimal")
end

"""
    constrained_minimum_variance(covariance; expected_returns=nothing,
                                return_floor=nothing, caps=nothing)

Minimize variance over fully invested nonnegative capped weights. An optional
return floor uses supplied expected returns. Covariance must be positive
semidefinite within the numerical eigensolver tolerance; it is not repaired.
JuMP and Clarabel solve the scaled quadratic objective. Solver status and
primal constraints are checked before returning weights and diagnostics.
"""
function constrained_minimum_variance(covariance; expected_returns = nothing, return_floor = nothing, caps = nothing)
    covariance, expected, cap, floor = _portfolio_inputs(covariance, expected_returns, caps, return_floor)
    return _solve_portfolio(covariance, expected, cap, floor; minimum_variance = true)
end

"""
    robust_mean_variance(expected_returns, covariance, uncertainty_loading;
                         radius, risk_aversion, caps=nothing, return_floor=nothing)

Maximize the worst mean minus half `risk_aversion` times variance for means in
`expected_returns + uncertainty_loading * u`, with `norm(u) <= radius`.
Covariance is fixed and every uncertainty input is supplied by the caller.
The optional return floor constrains nominal return. Radius and risk aversion
may be zero. Returns weights, nominal and worst-case means, variance, the
maximized objective and verified solver status.
"""
function robust_mean_variance(expected_returns, covariance, uncertainty_loading; radius, risk_aversion, caps = nothing, return_floor = nothing)
    covariance, expected, cap, floor = _portfolio_inputs(covariance, expected_returns, caps, return_floor)
    loading = _portfolio_array(uncertainty_loading, "uncertainty_loading", 2)
    size(loading, 1) == length(expected) ||
        throw(ArgumentError("uncertainty_loading must have one row per asset"))
    radius = _portfolio_scalar(radius, "radius"; minimum = 0.0)
    risk_aversion = _portfolio_scalar(risk_aversion, "risk_aversion"; minimum = 0.0)
    return _solve_portfolio(covariance, expected, cap, floor; loading, radius, risk_aversion)
end
