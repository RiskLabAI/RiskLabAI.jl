"""
Closed-form optimal Ornstein–Uhlenbeck trading rules (Lipton–López de Prado 2020).
de Prado finds optimal profit-take / stop-loss thresholds for a mean-reverting (OU)
process by a Monte-Carlo grid search (AFML ch.13), which carries simulation noise
and is limited by the grid resolution. The closed form gives the exact optimum for
the OU model from first-passage theory, with no simulation noise.

For an OU deviation entered at `y0 = -entry_gap` with upper barrier `b = y0 + pt`
and lower barrier `a = y0 - sl`, the hit-upper probability is
`u = [S(y0) - S(a)] / [S(b) - S(a)]` with scale function
`S(x) = ∫₀ˣ exp(θ s²/σ²) ds` (proportional to the imaginary error function), and the
mean first-exit time follows the Karlin–Taylor Green's-function form. The objective
is the expected net return per unit time `E[gain] / E[τ]`.

Clean-room Julia port; the closed-form metrics are deterministic and parity-matched
in `test/runtests.jl`. Deliberate divergence: the optimizer replaces SciPy's
L-BFGS-B with a coarse-to-fine grid maximization of the same closed-form objective
(no Optim dependency). Admitted in Appraisal 23
(`library_extension/appraisals/23_verdict.md`).

References: Lipton, A. & López de Prado, M. (2020), A closed-form solution for
optimal mean-reverting trading strategies; López de Prado (2018), AFML ch.13.
"""

using SpecialFunctions: erfi

# Trapezoidal integral ∫ y dx (matches numpy.trapz, including descending x).
function _ou_trapz(y, x)
    s = 0.0
    @inbounds for i = 1:(length(x)-1)
        s += (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2.0
    end
    return s
end

"""
    theta_from_half_life(half_life) -> Float64

Continuous mean-reversion speed `θ = ln 2 / half_life`. Mirrors Python.
"""
theta_from_half_life(half_life::Real) = log(2.0) / float(half_life)

"""
    stationary_std(theta, sigma) -> Float64

Stationary standard deviation of the OU deviation `σ / √(2θ)`. Mirrors Python.
"""
stationary_std(theta::Real, sigma::Real) = float(sigma) / sqrt(2.0 * float(theta))

# exp(-cK)(S(x) - S(a)) with K = max(a², b²) (overflow-safe), S the OU scale fn.
function _scale_relative(x, a, b, c)
    big = max(a * a, b * b)
    rc = sqrt(c)
    val = (sqrt(pi) / (2.0 * rc)) * (erfi(rc * x) - erfi(rc * a)) * exp(-c * big)
    isfinite(val) && return val
    s = range(a, x; length = 2000)
    return _ou_trapz(exp.(c .* (s .^ 2 .- big)), collect(s))
end

"""
    hit_upper_probability(entry_gap, profit_take, stop_loss, theta, sigma) -> Float64

Probability the OU deviation hits the upper (profit-take) barrier before the lower
(stop-loss) barrier. Mirrors Python's `hit_upper_probability`.
"""
function hit_upper_probability(entry_gap, profit_take, stop_loss, theta, sigma)
    c = theta / (sigma * sigma)
    y0 = -float(entry_gap)
    a, b = y0 - stop_loss, y0 + profit_take
    b <= a && return NaN
    num = _scale_relative(y0, a, b, c)
    den = _scale_relative(b, a, b, c)
    den <= 0 && return NaN
    return clamp(num / den, 0.0, 1.0)
end

# Overflow-safe E[τ] for very wide barriers (max-subtracted scale-function quadrature).
function _mean_exit_time_quadrature(entry_gap, profit_take, stop_loss, theta, sigma)
    c = theta / (sigma * sigma)
    y0 = -float(entry_gap)
    a, b = y0 - stop_loss, y0 + profit_take
    big = max(a * a, b * b)
    s_hat(x) = (s = range(a, x; length = 3000); _ou_trapz(exp.(c .* (s .^ 2 .- big)), collect(s)))
    s_hat_b, s_hat_y0 = s_hat(b), s_hat(y0)
    ys = collect(range(a, b; length = 3000))
    s_ya = [s_hat(y) for y in ys]
    s_by = s_hat_b .- s_ya
    green = [(ys[i] <= y0 ? s_ya[i] * (s_hat_b - s_hat_y0) : s_hat_y0 * s_by[i]) / s_hat_b for i in eachindex(ys)]
    speed = (2.0 / (sigma * sigma)) .* exp.(c .* (big .- ys .^ 2))
    return _ou_trapz(green .* speed, ys)
end

"""
    mean_exit_time(entry_gap, profit_take, stop_loss, theta, sigma) -> Float64

Closed-form OU mean first-exit time `E[τ]` from the two-barrier interval via the
Karlin–Taylor Green's function. Mirrors Python's `mean_exit_time`.
"""
function mean_exit_time(entry_gap, profit_take, stop_loss, theta, sigma)
    c = theta / (sigma * sigma)
    y0 = -float(entry_gap)
    a, b = y0 - stop_loss, y0 + profit_take
    b <= a && return NaN
    rc = sqrt(c)
    pref = sqrt(pi) / (2.0 * rc)
    ys = collect(range(a, b; length = 400))
    erfi_y = erfi.(rc .* ys)
    erfi_a, erfi_b, erfi_y0 = erfi(rc * a), erfi(rc * b), erfi(rc * y0)
    (all(isfinite, erfi_y) && isfinite(erfi_b)) ||
        return _mean_exit_time_quadrature(entry_gap, profit_take, stop_loss, theta, sigma)
    s_ya = pref .* (erfi_y .- erfi_a)
    s_by = pref .* (erfi_b .- erfi_y)
    s_ba = pref * (erfi_b - erfi_a)
    s_y0a = pref * (erfi_y0 - erfi_a)
    s_by0 = pref * (erfi_b - erfi_y0)
    green = [(ys[i] <= y0 ? s_ya[i] * s_by0 : s_y0a * s_by[i]) / s_ba for i in eachindex(ys)]
    speed = (2.0 / (sigma * sigma)) .* exp.(-c .* ys .^ 2)
    return _ou_trapz(green .* speed, ys)
end

"""
    ou_rule_metrics(profit_take, stop_loss, theta, sigma, entry_gap, cost=0.0)

Exact OU per-trade metrics for a rule entered `entry_gap` from the mean. Returns a
`NamedTuple` `(hit_probability, expected_gain, std_gain, expected_holding_time,
return_rate, time_scaled_sharpe)`. `return_rate = expected_gain / E[τ]` is the
optimization objective. Mirrors Python's `ou_rule_metrics`.
"""
function ou_rule_metrics(profit_take, stop_loss, theta, sigma, entry_gap, cost = 0.0)
    bad = (
        hit_probability = NaN,
        expected_gain = -Inf,
        std_gain = NaN,
        expected_holding_time = NaN,
        return_rate = -Inf,
        time_scaled_sharpe = -Inf,
    )
    (profit_take <= 0 || stop_loss <= 0) && return bad
    u = hit_upper_probability(entry_gap, profit_take, stop_loss, theta, sigma)
    isfinite(u) || return bad
    q = 1.0 - u
    expected_gain = profit_take * u - stop_loss * q - cost
    std_gain = (profit_take + stop_loss) * sqrt(max(u * q, 0.0))
    e_tau = mean_exit_time(entry_gap, profit_take, stop_loss, theta, sigma)
    (!isfinite(e_tau) || e_tau <= 0) && return bad
    return (
        hit_probability = u,
        expected_gain = expected_gain,
        std_gain = std_gain,
        expected_holding_time = e_tau,
        return_rate = expected_gain / e_tau,
        time_scaled_sharpe = std_gain > 0 ? expected_gain / (std_gain * sqrt(e_tau)) : -Inf,
    )
end

"""
    optimal_ou_trading_rule(theta, sigma, entry_gap; cost=0.0, bounds=(0.25, 4.0))

Exact optimal OU profit-take / stop-loss by maximizing the closed-form expected net
return per unit time. Returns a `NamedTuple` `(profit_take, stop_loss, ...)` merged
with the full `ou_rule_metrics` of the optimum.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer the closed-form OU rule over the Monte-Carlo grid whenever the OU model is
used — it reproduces the grid optimum within resolution, 18–350× faster, with no
simulation noise; it degrades in step with the grid off-model (no robustness to
misspecification), so check the OU fit at decision time. Objective: expected net
return per unit time.

Deliberate divergence: a coarse-to-fine grid search replaces SciPy's L-BFGS-B
multi-start (no Optim dependency). Mirrors Python's `optimal_ou_trading_rule`.
"""
function optimal_ou_trading_rule(theta, sigma, entry_gap; cost = 0.0, bounds = (0.25, 4.0))
    lo, hi = bounds
    objective(pt, sl) =
        (v = ou_rule_metrics(pt, sl, theta, sigma, entry_gap, cost).return_rate;
        isfinite(v) ? v : -1e9)

    blo_pt, bhi_pt, blo_sl, bhi_sl = lo, hi, lo, hi
    best_pt, best_sl, best_v = lo, lo, -Inf
    for _ = 1:4
        pts = range(blo_pt, bhi_pt; length = 41)
        sls = range(blo_sl, bhi_sl; length = 41)
        for pt in pts, sl in sls
            v = objective(pt, sl)
            if v > best_v
                best_v, best_pt, best_sl = v, pt, sl
            end
        end
        dpt = (bhi_pt - blo_pt) / 40
        dsl = (bhi_sl - blo_sl) / 40
        blo_pt, bhi_pt = max(lo, best_pt - dpt), min(hi, best_pt + dpt)
        blo_sl, bhi_sl = max(lo, best_sl - dsl), min(hi, best_sl + dsl)
    end
    metrics = ou_rule_metrics(best_pt, best_sl, theta, sigma, entry_gap, cost)
    return merge((profit_take = best_pt, stop_loss = best_sl), metrics)
end

"""
    fit_ornstein_uhlenbeck(series; dt=1.0)

Fit an OU process by AR(1) regression `xₜ = a + b·xₜ₋₁ + e` (the OU
goodness-of-fit check). Returns a `NamedTuple` `(theta, sigma, mu, rho, half_life,
stationary_std, r2)`. Check `r2` / the half-life before trusting the closed-form
rule. Mirrors Python's `fit_ornstein_uhlenbeck`.
"""
function fit_ornstein_uhlenbeck(series::AbstractVector{<:Real}; dt::Real = 1.0)
    x = float.(filter(isfinite, series))
    x0 = @view x[1:(end-1)]
    x1 = @view x[2:end]
    mx0, mx1 = mean(x0), mean(x1)
    b = sum((x0 .- mx0) .* (x1 .- mx1)) / sum((x0 .- mx0) .^ 2)
    a = mx1 - b * mx0
    residual = x1 .- (a .+ b .* x0)
    b = min(max(b, 1e-6), 0.999999)
    theta = -log(b) / dt
    mu = a / (1.0 - b)
    n1 = length(x1)
    var_eps = sum(residual .^ 2) / (n1 - 2)              # ddof = 2
    sigma = sqrt(var_eps * 2.0 * theta / (1.0 - b * b))
    ss_tot = sum((x1 .- mx1) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - sum(residual .^ 2) / ss_tot : 0.0
    return (
        theta = theta,
        sigma = sigma,
        mu = mu,
        rho = b,
        half_life = log(2.0) / theta,
        stationary_std = stationary_std(theta, sigma),
        r2 = r2,
    )
end
