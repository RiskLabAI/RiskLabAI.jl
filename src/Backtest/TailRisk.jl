using Distributions: Chisq, ccdf
import Optim

function _tail_real_scalar(value, name)
    value isa Real && !(value isa Bool) && isfinite(value) || throw(ArgumentError("$name must be a finite real scalar"))
    result = Float64(value)
    isfinite(result) || throw(ArgumentError("$name is outside the Float64 range"))
    return result
end

function _tail_real_vector(values, name)
    values isa AbstractVector && !isempty(values) && all(value -> value isa Real && !(value isa Bool) && isfinite(value), values) || throw(ArgumentError("$name must be a nonempty finite real vector"))
    result = Float64.(values)
    all(isfinite, result) || throw(ArgumentError("$name is outside the Float64 range"))
    return result
end

"""
    expected_shortfall(returns, step_risk_free_rate; confidence_level=0.05)

Return the mean of signed returns at or below their linearly interpolated
empirical quantile. The supplied level lies in `[0,1]`, and ties are included.
The risk-free argument is retained for public API compatibility and is unused,
as in Python 3.2.0. No sign reversal or risk-free subtraction is performed.
"""
function expected_shortfall(returns, step_risk_free_rate; confidence_level = 0.05)
    values = _tail_real_vector(returns, "returns")
    level = _tail_real_scalar(confidence_level, "confidence_level")
    0 <= level <= 1 || throw(ArgumentError("confidence_level must lie in [0, 1]"))
    cutoff = quantile(values, level)
    return mean(values[values .<= cutoff])
end

expected_shortfall(returns, step_risk_free_rate, confidence_level) = expected_shortfall(returns, step_risk_free_rate; confidence_level)

"""
    generalized_pareto_tail_risk(threshold, exceedance_probability, shape, scale, probability)

Return the unconditional loss quantile and upper-tail mean for the supplied
zero-location excess model. Require `0 < rate <= 1`, positive scale and
`1-rate < probability < 1`. Shape zero uses the exponential limit; shape at
least one gives infinite expected shortfall. All loss quantities share units.
No threshold-selection rule or time-series filter is implied.
"""
function generalized_pareto_tail_risk(threshold, exceedance_probability, shape, scale, probability)
    threshold = _tail_real_scalar(threshold, "threshold")
    rate = _tail_real_scalar(exceedance_probability, "exceedance_probability")
    shape = _tail_real_scalar(shape, "shape")
    scale = _tail_real_scalar(scale, "scale")
    probability = _tail_real_scalar(probability, "probability")
    scale > 0 && 0 < rate <= 1 && 1 - rate < probability < 1 || throw(ArgumentError("require positive scale, 0 < rate <= 1 and 1-rate < probability < 1"))
    log_survival = log((1 - probability) / rate)
    excess = scale * (shape == 0 ? -log_survival : expm1(-shape * log_survival) / shape)
    value_at_risk = threshold + excess
    expected_shortfall = shape >= 1 ? Inf : value_at_risk + (scale + shape * excess) / (1 - shape)
    isfinite(value_at_risk) && (shape >= 1 || isfinite(expected_shortfall)) || throw(ArgumentError("tail risk is outside the supported floating-point range"))
    return (; value_at_risk, expected_shortfall)
end

function _pareto_negative_log_likelihood(parameters, excesses)
    shape, scale = parameters
    isfinite(shape) && isfinite(scale) && scale > 0 || return Inf
    value = length(excesses) * log(scale)
    for excess in excesses
        standardized = excess / scale
        support = 1 + shape * standardized
        support > 0 || return Inf
        value += shape == 0 ? standardized : (1 + shape) * log1p(shape * standardized) / shape
    end
    return isfinite(value) ? value : Inf
end

"""
    fit_pot_tail_risk(losses; threshold, probability)

Fit a zero-location generalized Pareto likelihood to strictly positive
threshold excesses, requiring at least two distinct excesses. The threshold
is caller-supplied and must be chosen without held-out outcomes. Native
Nelder–Mead fitting rescales excesses by their mean and checks convergence,
support and finite likelihood. Shape is unrestricted. Numerical convergence
does not guarantee a global optimum or existence of a finite global MLE.
"""
function fit_pot_tail_risk(losses; threshold, probability)
    losses = _tail_real_vector(losses, "losses")
    threshold = _tail_real_scalar(threshold, "threshold")
    excesses = losses[losses .> threshold] .- threshold
    length(excesses) >= 2 && all(isfinite, excesses) && maximum(excesses) > minimum(excesses) || throw(ArgumentError("at least two distinct finite positive excesses are required"))
    unit = maximum(excesses)
    unit *= mean(excesses / unit)
    scaled = excesses / unit
    objective(parameters) = _pareto_negative_log_likelihood(parameters, scaled)
    method = Optim.NelderMead(initial_simplex = Optim.AffineSimplexer(a = 0.0, b = 0.05), parameters = Optim.FixedParameters())
    result = Optim.optimize(objective, [1.0, 1.0], method, Optim.Options(g_tol = 1e-12, iterations = 4000))
    Optim.converged(result) && isfinite(Optim.minimum(result)) || error("Pareto likelihood optimization failed to converge")
    shape, normalized_scale = Optim.minimizer(result)
    scale = normalized_scale * unit
    isfinite(_pareto_negative_log_likelihood([shape, scale], excesses)) || error("fitted Pareto model fails its support/likelihood check")
    rate = length(excesses) / length(losses)
    risk = generalized_pareto_tail_risk(threshold, rate, shape, scale, probability)
    return (; risk..., shape, scale, threshold, exceedance_count = length(excesses), sample_count = length(losses), exceedance_probability = rate)
end

function _tail_bernoulli_likelihood(zeros, ones, probability)
    return (ones == 0 ? 0.0 : ones * log(probability)) + (zeros == 0 ? 0.0 : zeros * log1p(-probability))
end

"""
    quantile_exception_tests(exceptions; exception_probability)

Return Kupiec coverage and Christoffersen independence and conditional coverage
statistics with asymptotic chi-square p-values. Coverage uses all observations;
transition likelihoods condition on the first. When a previous-state row is
absent, independence and conditional coverage fields are `nothing`. The input
is a nonempty binary vector with one for a loss above its forecast quantile.
These tests concern quantile forecasts, not expected-shortfall forecasts.
"""
function quantile_exception_tests(exceptions; exception_probability)
    exceptions isa AbstractVector && !isempty(exceptions) && all(value -> value isa Real && value in (0, 1), exceptions) || throw(ArgumentError("exceptions must be a nonempty binary vector"))
    probability = _tail_real_scalar(exception_probability, "exception_probability")
    0 < probability < 1 || throw(ArgumentError("exception_probability must lie in (0, 1)"))
    hits = Int.(exceptions)
    total, ones = length(hits), sum(hits)
    coverage = max(0.0, 2 * (_tail_bernoulli_likelihood(total - ones, ones, ones / total) - _tail_bernoulli_likelihood(total - ones, ones, probability)))
    transitions = zeros(Int, 2, 2)
    for index in 2:total
        transitions[hits[index-1]+1, hits[index]+1] += 1
    end
    rows = vec(sum(transitions; dims = 2))
    independence = nothing
    if all(>(0), rows)
        conditional = sum(_tail_bernoulli_likelihood(transitions[index, 1], transitions[index, 2], transitions[index, 2] / rows[index]) for index in 1:2)
        transition_ones = sum(transitions[:, 2])
        pooled = _tail_bernoulli_likelihood(sum(transitions[:, 1]), transition_ones, transition_ones / (total - 1))
        independence = max(0.0, 2 * (conditional - pooled))
    end
    combined = isnothing(independence) ? nothing : coverage + independence
    return (; coverage_statistic = coverage, coverage_pvalue = ccdf(Chisq(1), coverage),
        independence_statistic = independence, independence_pvalue = isnothing(independence) ? nothing : ccdf(Chisq(1), independence),
        conditional_coverage_statistic = combined, conditional_coverage_pvalue = isnothing(combined) ? nothing : ccdf(Chisq(2), combined),
        transitions, exception_count = ones, sample_count = total)
end
