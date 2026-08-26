using Distributions: Normal, ccdf, cdf, logcdf, logpdf
using QuadGK: quadgk

"Family error probabilities induced by selecting the maximum of K trials."
struct MaxSelectionFamilyErrors
    n_trials::Int
    trial_null_probability::Float64
    trial_type_i_error::Float64
    trial_type_ii_error::Float64
    family_null_probability::Float64
    family_type_i_error::Float64
    family_type_ii_error::Float64
end

"Log-domain evidence for the source's single-versus-family FDR conditions."
struct FDRComparisonEvidence
    n_trials::Int
    single_trial_false_discovery_rate::Float64
    family_level_false_discovery_rate::Float64
    log_observed_type_i_inflation::Float64
    log_required_type_i_inflation::Float64
    equation_13_log_gap::Float64
    identification_condition_holds::Bool
    single_trial_upper_bounds_family::Bool
end

"Two-component Gaussian model for one candidate specification."
struct GaussianTrialMixture
    trial_null_probability::Float64
    null_standard_deviation::Float64
    alternative_mean::Float64
    alternative_standard_deviation::Float64

    function GaussianTrialMixture(
        trial_null_probability,
        null_standard_deviation,
        alternative_mean,
        alternative_standard_deviation,
    )
        probability = _fdr_probability("trial_null_probability", trial_null_probability)
        null_scale = _fdr_positive("null_standard_deviation", null_standard_deviation)
        alternative_location = _fdr_finite("alternative_mean", alternative_mean)
        alternative_scale =
            _fdr_positive("alternative_standard_deviation", alternative_standard_deviation)
        return new(probability, null_scale, alternative_location, alternative_scale)
    end
end

"Per-threshold and aggregate Gaussian search-adjustment evidence."
struct GaussianSearchAdjustedFDR
    n_trials::Int
    thresholds::Vector{Float64}
    trial_type_i_errors::Vector{Float64}
    trial_type_ii_errors::Vector{Float64}
    family_type_i_errors::Vector{Float64}
    family_type_ii_errors::Vector{Float64}
    mean_family_type_i_error::Float64
    mean_family_type_ii_error::Float64
    family_null_probability::Float64
    false_discovery_rate::Float64

    function GaussianSearchAdjustedFDR(
        n_trials,
        thresholds,
        trial_type_i_errors,
        trial_type_ii_errors,
        family_type_i_errors,
        family_type_ii_errors,
        mean_family_type_i_error,
        mean_family_type_ii_error,
        family_null_probability,
        false_discovery_rate,
    )
        trials = _fdr_positive_integer("n_trials", n_trials)
        vectors = map(
            pair -> _fdr_vector(pair[1], pair[2]),
            (
                ("thresholds", thresholds),
                ("trial_type_i_errors", trial_type_i_errors),
                ("trial_type_ii_errors", trial_type_ii_errors),
                ("family_type_i_errors", family_type_i_errors),
                ("family_type_ii_errors", family_type_ii_errors),
            ),
        )
        isempty(first(vectors)) &&
            throw(ArgumentError("threshold evidence vectors must be nonempty"))
        all(length(vector) == length(first(vectors)) for vector in vectors) ||
            throw(ArgumentError("threshold evidence vectors must have equal lengths"))
        return new(
            trials,
            vectors...,
            _fdr_probability("mean_family_type_i_error", mean_family_type_i_error),
            _fdr_probability("mean_family_type_ii_error", mean_family_type_ii_error),
            _fdr_probability("family_null_probability", family_null_probability),
            _fdr_probability("false_discovery_rate", false_discovery_rate),
        )
    end
end

"Two-trial equal-component construction proving unrestricted nonidentification."
struct FDRNonIdentificationWitness
    n_trials::Int
    target_false_discovery_rate::Float64
    trial_null_probability::Float64
    observable_cdf_at_threshold::Float64
    latent_trial_cdf_at_threshold::Float64
    family_null_probability::Float64
    family_type_i_error::Float64
    family_type_ii_error::Float64
    family_false_discovery_rate::Float64
    reconstructed_observable_cdf_at_threshold::Float64
end

"Probability evidence that a selected maximum is a null candidate."
struct SelectionLevelProbabilityEvidence
    joint_null_and_selection_probability::Float64
    selection_probability::Float64
    conditional_null_probability::Float64
    quadrature_absolute_error::Float64
end

function _fdr_finite(name::AbstractString, value)
    value isa Bool && throw(ArgumentError("$name must be a real scalar"))
    value isa Real || throw(ArgumentError("$name must be a real scalar"))
    result = try
        Float64(value)
    catch
        throw(ArgumentError("$name must be a representable real scalar"))
    end
    isfinite(result) || throw(ArgumentError("$name must be finite"))
    return result
end

function _fdr_probability(name::AbstractString, value; strict::Bool = false)
    result = _fdr_finite(name, value)
    valid = strict ? 0.0 < result < 1.0 : 0.0 <= result <= 1.0
    valid || throw(
        ArgumentError(
            strict ? "$name must be strictly between zero and one" :
            "$name must be between zero and one",
        ),
    )
    return result
end

function _fdr_positive(name::AbstractString, value)
    result = _fdr_finite(name, value)
    result > 0.0 || throw(ArgumentError("$name must be positive"))
    return result
end

function _fdr_nonnegative(name::AbstractString, value)
    result = _fdr_finite(name, value)
    result >= 0.0 || throw(ArgumentError("$name must be nonnegative"))
    return result
end

function _fdr_positive_integer(name::AbstractString, value)
    value isa Bool && throw(ArgumentError("$name must be an integer"))
    value isa Integer || throw(ArgumentError("$name must be an integer"))
    value > 0 || throw(ArgumentError("$name must be positive"))
    value <= typemax(Int) || throw(ArgumentError("$name is too large"))
    return Int(value)
end

function _fdr_vector(name::AbstractString, values)
    values isa AbstractVector ||
        throw(ArgumentError("$name must be a one-dimensional real vector"))
    result = Vector{Float64}(undef, length(values))
    for (index, value) in enumerate(values)
        result[index] = _fdr_finite("$name[$index]", value)
    end
    return result
end

_fdr_log_probability(value::Float64) = iszero(value) ? -Inf : log(value)
_fdr_log_complement(value::Float64) = isone(value) ? -Inf : log1p(-value)

function _fdr_log_one_minus_exp(log_value::Float64)
    log_value <= 0.0 || throw(ArgumentError("log probability must not be positive"))
    log_value == -Inf && return 0.0
    iszero(log_value) && return -Inf
    return log_value < -log(2.0) ? log1p(-exp(log_value)) : log(-expm1(log_value))
end

function _fdr_probability_power(probability::Float64, n_trials::Int)
    iszero(probability) && return 0.0
    isone(probability) && return 1.0
    return exp(n_trials * log(probability))
end

function _fdr_one_minus_probability_power(probability::Float64, n_trials::Int)
    iszero(probability) && return 1.0
    isone(probability) && return 0.0
    return -expm1(n_trials * log(probability))
end

function _fdr_logsumexp(first::Float64, second::Float64)
    maximum_value = max(first, second)
    maximum_value == -Inf && return -Inf
    return maximum_value + log(exp(first - maximum_value) + exp(second - maximum_value))
end

function _fdr_posterior(log_null::Float64, log_alternative::Float64)
    log_null == -Inf &&
        log_alternative == -Inf &&
        throw(ArgumentError("the discovery probability must be positive"))
    log_null == -Inf && return 0.0
    log_alternative == -Inf && return 1.0
    return exp(log_null - _fdr_logsumexp(log_null, log_alternative))
end

function _fdr_log_weighted_pair(weight::Float64, first::Float64, second::Float64)
    iszero(weight) && return second
    isone(weight) && return first
    return _fdr_logsumexp(log(weight) + first, log1p(-weight) + second)
end

"""
    single_trial_false_discovery_rate(type_i_error, type_ii_error, trial_null_probability)

Return the single-trial posterior false-discovery rate from Equation 6.
"""
function single_trial_false_discovery_rate(
    type_i_error,
    type_ii_error,
    trial_null_probability,
)
    alpha = _fdr_probability("type_i_error", type_i_error)
    beta = _fdr_probability("type_ii_error", type_ii_error)
    pi_zero = _fdr_probability("trial_null_probability", trial_null_probability)
    return _fdr_posterior(
        _fdr_log_probability(alpha) + _fdr_log_probability(pi_zero),
        _fdr_log_complement(beta) + _fdr_log_complement(pi_zero),
    )
end

"""
    family_level_false_discovery_rate(family_type_i_error, family_type_ii_error,
                                      trial_null_probability, n_trials)

Return the family-level search-adjusted false-discovery rate from Equations 9-10.
"""
function family_level_false_discovery_rate(
    family_type_i_error,
    family_type_ii_error,
    trial_null_probability,
    n_trials,
)
    alpha_family = _fdr_probability("family_type_i_error", family_type_i_error)
    beta_family = _fdr_probability("family_type_ii_error", family_type_ii_error)
    pi_zero = _fdr_probability("trial_null_probability", trial_null_probability)
    trials = _fdr_positive_integer("n_trials", n_trials)
    log_family_null = trials * _fdr_log_probability(pi_zero)
    return _fdr_posterior(
        _fdr_log_probability(alpha_family) + log_family_null,
        _fdr_log_complement(beta_family) + _fdr_log_one_minus_exp(log_family_null),
    )
end

"Return maximum-selection family errors from Equations 17-18."
function max_selection_family_errors(
    trial_type_i_error,
    trial_type_ii_error,
    trial_null_probability,
    n_trials,
)
    alpha = _fdr_probability("trial_type_i_error", trial_type_i_error)
    beta = _fdr_probability("trial_type_ii_error", trial_type_ii_error)
    pi_zero =
        _fdr_probability("trial_null_probability", trial_null_probability; strict = true)
    trials = _fdr_positive_integer("n_trials", n_trials)

    family_null = _fdr_probability_power(pi_zero, trials)
    family_alpha = _fdr_one_minus_probability_power(1.0 - alpha, trials)
    first_base = pi_zero * (1.0 - alpha) + (1.0 - pi_zero) * beta
    second_base = pi_zero * (1.0 - alpha)
    log_denominator = _fdr_log_one_minus_exp(trials * log(pi_zero))
    family_beta = if iszero(first_base) || first_base == second_base
        0.0
    else
        log_first_power = trials * log(first_base)
        log_numerator = if iszero(second_base)
            log_first_power
        else
            log_ratio = trials * (log(second_base) - log(first_base))
            log_first_power + _fdr_log_one_minus_exp(log_ratio)
        end
        clamp(exp(log_numerator - log_denominator), 0.0, 1.0)
    end
    return MaxSelectionFamilyErrors(
        trials,
        pi_zero,
        alpha,
        beta,
        family_null,
        family_alpha,
        family_beta,
    )
end

"Evaluate the equality and upper-bound conditions in Equations 13-15."
function compare_single_and_family_fdr(
    trial_type_i_error,
    trial_type_ii_error,
    trial_null_probability,
    n_trials;
    family_type_i_error,
    family_type_ii_error,
    relative_tolerance = 1e-12,
    absolute_tolerance = 1e-15,
)
    alpha = _fdr_probability("trial_type_i_error", trial_type_i_error; strict = true)
    beta = _fdr_probability("trial_type_ii_error", trial_type_ii_error; strict = true)
    pi_zero =
        _fdr_probability("trial_null_probability", trial_null_probability; strict = true)
    trials = _fdr_positive_integer("n_trials", n_trials)
    alpha_family =
        _fdr_probability("family_type_i_error", family_type_i_error; strict = true)
    beta_family =
        _fdr_probability("family_type_ii_error", family_type_ii_error; strict = true)
    relative = _fdr_nonnegative("relative_tolerance", relative_tolerance)
    absolute = _fdr_nonnegative("absolute_tolerance", absolute_tolerance)

    log_observed = log(alpha_family) - log(alpha)
    log_family_null = trials * log(pi_zero)
    log_required =
        log1p(-beta_family) - log1p(-beta) + _fdr_log_one_minus_exp(log_family_null) -
        (trials - 1) * log(pi_zero) - log1p(-pi_zero)
    gap = log_observed - log_required
    scale_tolerance = max(absolute, relative * max(abs(log_observed), abs(log_required)))
    equality = isapprox(log_observed, log_required; rtol = relative, atol = absolute)
    return FDRComparisonEvidence(
        trials,
        single_trial_false_discovery_rate(alpha, beta, pi_zero),
        family_level_false_discovery_rate(alpha_family, beta_family, pi_zero, trials),
        log_observed,
        log_required,
        gap,
        equality,
        gap <= scale_tolerance,
    )
end

"Return the CDF of the maximum in Equation 16 from component CDF values."
function maximum_mixture_cdf(null_cdf, alternative_cdf, trial_null_probability, n_trials)
    null_value = _fdr_probability("null_cdf", null_cdf)
    alternative_value = _fdr_probability("alternative_cdf", alternative_cdf)
    pi_zero = _fdr_probability("trial_null_probability", trial_null_probability)
    trials = _fdr_positive_integer("n_trials", n_trials)
    mixture = clamp(pi_zero * null_value + (1.0 - pi_zero) * alternative_value, 0.0, 1.0)
    return _fdr_probability_power(mixture, trials)
end

"Return `P(X >= x | X >= c)` from Equation 26 for `x >= c`."
function conditional_upper_tail_probability(cdf_at_value, cdf_at_threshold)
    at_value = _fdr_probability("cdf_at_value", cdf_at_value)
    at_threshold = _fdr_probability("cdf_at_threshold", cdf_at_threshold)
    isone(at_threshold) && throw(ArgumentError("cdf_at_threshold must be less than one"))
    at_value >= at_threshold ||
        throw(ArgumentError("cdf_at_value must not be below cdf_at_threshold"))
    return (1.0 - at_value) / (1.0 - at_threshold)
end

"Return the Gaussian trial-mixture CDF in Equations 33-35."
function gaussian_trial_mixture_cdf(model::GaussianTrialMixture, x)
    value = _fdr_finite("x", x)
    null_cdf = cdf(Normal(0.0, model.null_standard_deviation), value)
    alternative_cdf =
        cdf(Normal(model.alternative_mean, model.alternative_standard_deviation), value)
    return clamp(
        model.trial_null_probability * null_cdf +
        (1.0 - model.trial_null_probability) * alternative_cdf,
        0.0,
        1.0,
    )
end

"Return the selected Gaussian maximum CDF in Equation 35."
function gaussian_max_selection_cdf(model::GaussianTrialMixture, x, n_trials)
    trials = _fdr_positive_integer("n_trials", n_trials)
    return _fdr_probability_power(gaussian_trial_mixture_cdf(model, x), trials)
end

function _fdr_gaussian_component_logs(model::GaussianTrialMixture, x::Float64)
    null_distribution = Normal(0.0, model.null_standard_deviation)
    alternative_distribution =
        Normal(model.alternative_mean, model.alternative_standard_deviation)
    return (
        logcdf(null_distribution, x),
        logcdf(alternative_distribution, x),
        logpdf(null_distribution, x),
        logpdf(alternative_distribution, x),
    )
end

"Return the log density of the selected Gaussian maximum from Equation 36."
function gaussian_max_selection_log_density(model::GaussianTrialMixture, x, n_trials)
    value = _fdr_finite("x", x)
    trials = _fdr_positive_integer("n_trials", n_trials)
    (null_log_cdf, alternative_log_cdf, null_log_density, alternative_log_density) =
        _fdr_gaussian_component_logs(model, value)
    log_mixture_cdf = _fdr_log_weighted_pair(
        model.trial_null_probability,
        null_log_cdf,
        alternative_log_cdf,
    )
    log_mixture_density = _fdr_log_weighted_pair(
        model.trial_null_probability,
        null_log_density,
        alternative_log_density,
    )
    return log(trials) + (trials - 1) * log_mixture_cdf + log_mixture_density
end

"Return the Equation 37 log-likelihood for selected maxima."
function gaussian_max_selection_log_likelihood(
    model::GaussianTrialMixture,
    observations,
    n_trials,
)
    values = _fdr_vector("observations", observations)
    isempty(values) && throw(ArgumentError("observations must be nonempty"))
    trials = _fdr_positive_integer("n_trials", n_trials)
    result =
        sum(gaussian_max_selection_log_density(model, value, trials) for value in values)
    (isnan(result) || result == Inf) &&
        throw(ArgumentError("log-likelihood could not be evaluated"))
    return result
end

"Apply Equations 38-40 to a vector of Gaussian thresholds."
function gaussian_search_adjusted_false_discovery_rate(
    model::GaussianTrialMixture,
    thresholds,
    n_trials,
)
    threshold_values = _fdr_vector("thresholds", thresholds)
    isempty(threshold_values) && throw(ArgumentError("thresholds must be nonempty"))
    trials = _fdr_positive_integer("n_trials", n_trials)
    null_distribution = Normal(0.0, model.null_standard_deviation)
    alternative_distribution =
        Normal(model.alternative_mean, model.alternative_standard_deviation)
    trial_alpha = [ccdf(null_distribution, threshold) for threshold in threshold_values]
    trial_beta =
        [cdf(alternative_distribution, threshold) for threshold in threshold_values]
    family_alpha = similar(trial_alpha)
    family_beta = similar(trial_beta)
    for index in eachindex(threshold_values)
        evidence = max_selection_family_errors(
            trial_alpha[index],
            trial_beta[index],
            model.trial_null_probability,
            trials,
        )
        family_alpha[index] = evidence.family_type_i_error
        family_beta[index] = evidence.family_type_ii_error
    end
    mean_family_alpha = mean(family_alpha)
    mean_family_beta = mean(family_beta)
    family_null = _fdr_probability_power(model.trial_null_probability, trials)
    false_discovery_rate = family_level_false_discovery_rate(
        mean_family_alpha,
        mean_family_beta,
        model.trial_null_probability,
        trials,
    )
    return GaussianSearchAdjustedFDR(
        trials,
        threshold_values,
        trial_alpha,
        trial_beta,
        family_alpha,
        family_beta,
        mean_family_alpha,
        mean_family_beta,
        family_null,
        false_discovery_rate,
    )
end

"Return the exact unrestricted two-trial witness from Equations 41-50."
function fdr_nonidentification_witness(
    observable_cdf_at_threshold,
    target_false_discovery_rate,
)
    observable_cdf = _fdr_probability(
        "observable_cdf_at_threshold",
        observable_cdf_at_threshold;
        strict = true,
    )
    target = _fdr_probability(
        "target_false_discovery_rate",
        target_false_discovery_rate;
        strict = true,
    )
    trials = 2
    trial_null = sqrt(target)
    latent_cdf = sqrt(observable_cdf)
    family_null = trial_null^trials
    family_alpha = 1.0 - observable_cdf
    family_beta = observable_cdf
    family_fdr =
        family_level_false_discovery_rate(family_alpha, family_beta, trial_null, trials)
    return FDRNonIdentificationWitness(
        trials,
        target,
        trial_null,
        observable_cdf,
        latent_cdf,
        family_null,
        family_alpha,
        family_beta,
        family_fdr,
        latent_cdf^trials,
    )
end

"""
    max_selection_null_probability(threshold, trial_null_probability, n_trials;
                                   null_density, mixture_cdf, ...)

Return the selected-winner null probability from footnote 25. The result is a
selection-level estimand and is deliberately distinct from family-level FDR.
"""
function max_selection_null_probability(
    threshold,
    trial_null_probability,
    n_trials;
    null_density,
    mixture_cdf,
    absolute_tolerance = 1e-12,
    relative_tolerance = 1e-10,
    max_subintervals = 200,
)
    threshold_value = _fdr_finite("threshold", threshold)
    pi_zero = _fdr_probability("trial_null_probability", trial_null_probability)
    trials = _fdr_positive_integer("n_trials", n_trials)
    absolute = _fdr_positive("absolute_tolerance", absolute_tolerance)
    relative = _fdr_positive("relative_tolerance", relative_tolerance)
    subdivisions = _fdr_positive_integer("max_subintervals", max_subintervals)

    checked_cdf(value) = _fdr_probability("mixture_cdf return value", mixture_cdf(value))
    mixture_at_threshold = checked_cdf(threshold_value)
    selection_probability = _fdr_one_minus_probability_power(mixture_at_threshold, trials)
    iszero(selection_probability) &&
        throw(ArgumentError("the selection probability must be positive"))
    if iszero(pi_zero)
        return SelectionLevelProbabilityEvidence(0.0, selection_probability, 0.0, 0.0)
    end

    function integrand(value)
        density = _fdr_finite("null_density return value", null_density(value))
        density >= 0.0 ||
            throw(ArgumentError("null_density return value must be nonnegative"))
        return density * _fdr_probability_power(checked_cdf(value), trials - 1)
    end
    subdivisions <= typemax(Int) ÷ 15 ||
        throw(ArgumentError("max_subintervals is too large"))
    integral, error = quadgk(
        integrand,
        threshold_value,
        Inf;
        atol = absolute,
        rtol = relative,
        maxevals = max(15, 15 * subdivisions),
    )
    (isfinite(integral) && isfinite(error) && integral >= 0.0) ||
        throw(ArgumentError("selection-level quadrature returned invalid evidence"))
    scale = trials * pi_zero
    joint_probability = scale * integral
    absolute_error = scale * error
    comparison_tolerance =
        max(absolute, relative * selection_probability, 8.0 * eps(Float64))
    joint_probability <= selection_probability + comparison_tolerance ||
        throw(ArgumentError("callbacks imply a null probability above one"))
    joint_probability = clamp(joint_probability, 0.0, selection_probability)
    return SelectionLevelProbabilityEvidence(
        joint_probability,
        selection_probability,
        joint_probability / selection_probability,
        absolute_error,
    )
end
