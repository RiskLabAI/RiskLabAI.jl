"""
    eisenberg_noe_clearing(external_assets, nominal_liabilities,
                          relative_liabilities; atol=1e-8)

Return the greatest proportional clearing payment vector with an outside
creditor. Row `i` of `relative_liabilities` gives debtor `i`'s fractions owed
to network creditors; missing row mass is owed outside. Assets, liabilities
and fractions are finite and nonnegative. There are no bankruptcy costs or
fire sales. A linear program maximizes total feasible payments, including
networks with multiple fixed points. Bounds and the fixed-point residual
are checked after scaling by the greatest nominal liability.
"""
function eisenberg_noe_clearing(external_assets, nominal_liabilities, relative_liabilities; atol = 1e-8)
    assets = _portfolio_array(external_assets, "external_assets", 1)
    liabilities = _portfolio_array(nominal_liabilities, "nominal_liabilities", 1)
    relative = _portfolio_array(relative_liabilities, "relative_liabilities", 2)
    tolerance = _portfolio_scalar(atol, "atol")
    0 < tolerance < 1 || throw(ArgumentError("atol must lie in (0, 1)"))
    n = length(assets)
    length(liabilities) == n && size(relative) == (n, n) || throw(ArgumentError("liability dimensions must match assets"))
    all(>=(0), assets) && all(>=(0), liabilities) && all(>=(0), relative) || throw(ArgumentError("inputs must be nonnegative"))
    all(sum(relative; dims = 2) .<= 1) || throw(ArgumentError("relative-liability rows must sum to at most one"))
    scale = maximum(liabilities)
    scale == 0 && return zeros(n)
    upper = liabilities / scale
    resources = min.(assets, liabilities) / scale
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    for name in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
        set_optimizer_attribute(model, name, min(1e-14, tolerance / 10))
    end
    @variable(model, 0 <= payment[i = 1:n] <= upper[i])
    @constraint(model, [i = 1:n], payment[i] - sum(relative[j, i] * payment[j] for j in 1:n) <= resources[i])
    @objective(model, Max, sum(payment))
    optimize!(model)
    termination_status(model) == JuMP.MOI.OPTIMAL && has_values(model) || error("Clearing optimization is not verified: $(termination_status(model))")
    result = value.(payment)
    all(isfinite, result) && all(-tolerance .<= result .<= upper .+ tolerance) || error("Clearing returned invalid payment bounds")
    result = clamp.(result, 0, upper)
    maximum(abs.(result - min.(upper, resources + relative' * result))) <= tolerance || error("Clearing failed the fixed-point check")
    return clamp.(result * scale, 0, liabilities)
end
