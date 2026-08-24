struct StrategyPerformance
    correct::Float64
    misspecified::Float64
end

struct ColliderCoefficients
    beta_hat::Float64
    theta_hat::Float64
end

struct ColliderDiagnostics
    residual_variance::Float64
    outcome_variance::Float64
    correct_r_squared::Float64
    overcontrolled_r_squared::Float64
    correct_adjusted_r_squared::Float64
    overcontrolled_adjusted_r_squared::Float64
    correct_beta_variance::Float64
    overcontrolled_beta_variance::Float64
    collider_coefficient_variance::Float64
    correct_beta_t_statistic::Float64
    overcontrolled_beta_t_statistic::Float64
    collider_t_statistic::Float64
    adjusted_r_squared_prefers_overcontrolled::Bool
    absolute_beta_t_prefers_overcontrolled::Bool
end

function _finite_output(name::AbstractString, value)
    result = try
        Float64(value)
    catch
        throw(ArgumentError("$name cannot be represented as a finite Float64"))
    end
    isfinite(result) ||
        throw(ArgumentError("$name cannot be represented as a finite Float64"))
    return result
end

_exact_fraction(name, value) = Rational{BigInt}(_finite_real(name, value))

function _coefficient_times_square_root(name, coefficient, radicand)
    iszero(coefficient) && return 0.0
    result = try
        setprecision(BigFloat, 256) do
            Float64(BigFloat(coefficient) * sqrt(BigFloat(radicand)))
        end
    catch
        throw(ArgumentError("$name cannot be represented as a finite Float64"))
    end
    isfinite(result) ||
        throw(ArgumentError("$name cannot be represented as a finite Float64"))
    return result
end

_confounder_coefficient(beta, gamma, delta) = beta + gamma * delta / (1 + delta * delta)

function _collider_coefficients(beta, gamma, delta)
    denominator = 1 + gamma * gamma
    return (beta - delta * gamma) / denominator, gamma / denominator
end

function confounder_undercontrolled_coefficient(beta, gamma, delta)
    beta_value = _exact_fraction("beta", beta)
    gamma_value = _exact_fraction("gamma", gamma)
    delta_value = _exact_fraction("delta", delta)
    return _finite_output(
        "undercontrolled coefficient",
        _confounder_coefficient(beta_value, gamma_value, delta_value),
    )
end

function confounder_factor_return(
    x,
    z,
    beta,
    gamma,
    delta_estimated;
    delta_realized = nothing,
)
    x_value = _exact_fraction("x", x)
    z_value = _exact_fraction("z", z)
    beta_value = _exact_fraction("beta", beta)
    gamma_value = _exact_fraction("gamma", gamma)
    estimated = _exact_fraction("delta_estimated", delta_estimated)
    realized =
        delta_realized === nothing ? estimated :
        _exact_fraction("delta_realized", delta_realized)

    correct_signal = x_value * beta_value + z_value * gamma_value
    estimated_coefficient = _confounder_coefficient(beta_value, gamma_value, estimated)
    realized_coefficient = _confounder_coefficient(beta_value, gamma_value, realized)
    return StrategyPerformance(
        _finite_output("correct factor return", correct_signal * correct_signal),
        _finite_output(
            "undercontrolled factor return",
            x_value * x_value * estimated_coefficient * realized_coefficient,
        ),
    )
end

function confounder_forecast_return(beta, gamma, delta_estimated; delta_realized = nothing)
    beta_value = _exact_fraction("beta", beta)
    gamma_value = _exact_fraction("gamma", gamma)
    estimated = _exact_fraction("delta_estimated", delta_estimated)
    realized =
        delta_realized === nothing ? estimated :
        _exact_fraction("delta_realized", delta_realized)

    estimated_coefficient = _confounder_coefficient(beta_value, gamma_value, estimated)
    realized_coefficient = _confounder_coefficient(beta_value, gamma_value, realized)
    second_signal = beta_value * realized + gamma_value
    return StrategyPerformance(
        _finite_output(
            "correct forecast return",
            beta_value * beta_value + second_signal * second_signal,
        ),
        _finite_output(
            "undercontrolled forecast return",
            (1 + realized * realized) * estimated_coefficient * realized_coefficient,
        ),
    )
end

function collider_overcontrolled_coefficients(beta, gamma, delta)
    beta_value = _exact_fraction("beta", beta)
    gamma_value = _exact_fraction("gamma", gamma)
    delta_value = _exact_fraction("delta", delta)
    beta_hat, theta_hat = _collider_coefficients(beta_value, gamma_value, delta_value)
    return ColliderCoefficients(
        _finite_output("overcontrolled beta coefficient", beta_hat),
        _finite_output("collider coefficient", theta_hat),
    )
end

function collider_factor_return(x, collider_proxy, beta, gamma, delta)
    x_value = _exact_fraction("x", x)
    proxy_value = _exact_fraction("collider_proxy", collider_proxy)
    beta_value = _exact_fraction("beta", beta)
    gamma_value = _exact_fraction("gamma", gamma)
    delta_value = _exact_fraction("delta", delta)

    beta_x = beta_value * x_value
    denominator = 1 + gamma_value * gamma_value
    overcontrolled_signal =
        (beta_value - delta_value * gamma_value) * x_value + gamma_value * proxy_value
    return StrategyPerformance(
        _finite_output("correct collider factor return", beta_x * beta_x),
        _finite_output(
            "overcontrolled factor return",
            beta_x * overcontrolled_signal / denominator,
        ),
    )
end

function collider_forecast_return(beta, gamma, delta)
    beta_value = _exact_fraction("beta", beta)
    gamma_value = _exact_fraction("gamma", gamma)
    delta_value = _exact_fraction("delta", delta)
    denominator = 1 + gamma_value * gamma_value
    return StrategyPerformance(
        _finite_output("correct collider forecast return", beta_value * beta_value),
        _finite_output(
            "overcontrolled forecast return",
            beta_value * (beta_value - gamma_value * delta_value) / denominator,
        ),
    )
end

function collider_model_diagnostics(beta, gamma, delta, n_observations)
    beta_value = _exact_fraction("beta", beta)
    gamma_value = _exact_fraction("gamma", gamma)
    delta_value = _exact_fraction("delta", delta)
    n_observations isa Bool && throw(ArgumentError("n_observations must be an integer"))
    n_observations isa Integer || throw(ArgumentError("n_observations must be an integer"))
    n_integer = BigInt(n_observations)
    n_integer > 3 || throw(ArgumentError("n_observations must be greater than 3"))
    n_value = n_integer // BigInt(1)

    gamma_denominator = 1 + gamma_value * gamma_value
    outcome_variance = 1 + beta_value * beta_value
    residual_variance = 1 / gamma_denominator
    correct_r_squared = 1 - 1 / outcome_variance
    overcontrolled_r_squared = 1 - 1 / (gamma_denominator * outcome_variance)
    correct_adjusted_r_squared =
        1 - (((n_integer - 1) // (n_integer - 2)) / outcome_variance)
    overcontrolled_adjusted_r_squared =
        1 - ((n_integer - 1) // (n_integer - 3)) / (gamma_denominator * outcome_variance)

    diagnostic_sum =
        (beta_value * gamma_value + delta_value)^2 + gamma_value * gamma_value + 1
    correct_beta_variance = 1 / n_value
    overcontrolled_beta_variance = diagnostic_sum / (n_value * gamma_denominator^2)
    collider_coefficient_variance = 1 / (n_value * gamma_denominator^2)
    overcontrolled_numerator = beta_value - delta_value * gamma_value

    return ColliderDiagnostics(
        _finite_output("residual variance", residual_variance),
        _finite_output("outcome variance", outcome_variance),
        _finite_output("correct R-squared", correct_r_squared),
        _finite_output("overcontrolled R-squared", overcontrolled_r_squared),
        _finite_output("correct adjusted R-squared", correct_adjusted_r_squared),
        _finite_output(
            "overcontrolled adjusted R-squared",
            overcontrolled_adjusted_r_squared,
        ),
        _finite_output("correct beta variance", correct_beta_variance),
        _finite_output("overcontrolled beta variance", overcontrolled_beta_variance),
        _finite_output("collider coefficient variance", collider_coefficient_variance),
        _coefficient_times_square_root("correct beta t-statistic", beta_value, n_value),
        _coefficient_times_square_root(
            "overcontrolled beta t-statistic",
            overcontrolled_numerator,
            n_value / diagnostic_sum,
        ),
        _coefficient_times_square_root("collider t-statistic", gamma_value, n_value),
        gamma_value^2 * (n_integer - 3) > 1,
        overcontrolled_numerator^2 > beta_value^2 * diagnostic_sum,
    )
end
