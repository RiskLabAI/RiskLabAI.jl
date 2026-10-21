"""
    almgren_chriss_execution(inventory, horizon, intervals; volatility,
                            temporary_impact, permanent_impact=0.0,
                            risk_aversion=0.0)

Compute the discrete, single-asset, zero-drift Almgren--Chriss sell schedule
with constant linear impact and no spread. All quantities are finite and
nonnegative, with positive horizon, temporary impact and integer intervals.
The effective temporary impact `temporary_impact - permanent_impact*tau/2`
must be positive. Cost variance uses post-trade inventory, so liquidation
in one interval has zero modeled variance. Return times, inventories, trades,
expected cost and cost variance. Reject nonfinite derived costs.
"""
function almgren_chriss_execution(inventory, horizon, intervals; volatility, temporary_impact, permanent_impact = 0.0, risk_aversion = 0.0)
    quantity = _portfolio_scalar(inventory, "inventory"; minimum = 0)
    duration = _portfolio_scalar(horizon, "horizon"; minimum = 0)
    sigma = _portfolio_scalar(volatility, "volatility"; minimum = 0)
    eta = _portfolio_scalar(temporary_impact, "temporary_impact"; minimum = 0)
    gamma = _portfolio_scalar(permanent_impact, "permanent_impact"; minimum = 0)
    risk = _portfolio_scalar(risk_aversion, "risk_aversion"; minimum = 0)
    intervals isa Integer && !(intervals isa Bool) && intervals > 0 || throw(ArgumentError("intervals must be a positive integer"))
    duration > 0 && eta > 0 || throw(ArgumentError("horizon and temporary_impact must be positive"))
    tau = duration / intervals
    effective_eta = eta - gamma * tau / 2
    tau > 0 && isfinite(effective_eta) && effective_eta > 0 || throw(ArgumentError("time step and effective temporary impact must be positive"))
    index = Float64.(0:intervals)
    remaining = if quantity == 0 || risk == 0 || sigma == 0
        quantity .* (1 .- index / intervals)
    else
        log_z = log(tau) + log(sigma) + (log(risk) - log(effective_eta)) / 2 - log(2.0)
        step = log_z < 350 ? 2 * asinh(exp(log_z)) : 2 * (log_z + log(2.0))
        step == 0 ? quantity .* (1 .- index / intervals) :
            quantity .* exp.(-step .* index) .* (-expm1.(-2 * step .* (intervals .- index))) ./ (-expm1(-2 * step * intervals))
    end
    remaining[1], remaining[end] = quantity, 0.0
    trades = -diff(remaining)
    cost = gamma * quantity^2 / 2 + effective_eta / tau * dot(trades, trades)
    variance = sigma^2 * tau * sum(abs2, remaining[2:end])
    isfinite(cost) && isfinite(variance) || throw(ArgumentError("execution costs exceed floating-point range"))
    return (times = collect(range(0, duration; length = intervals + 1)), inventory = remaining,
        trades, expected_cost = cost, cost_variance = variance)
end
