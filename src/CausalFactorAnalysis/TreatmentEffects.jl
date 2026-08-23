struct TreatmentEffectDecomposition
    observed_difference::Float64
    average_treatment_effect_on_treated::Float64
    sample_selection_bias::Float64
end

struct DifferenceInDifferencesEstimate
    treated_change::Float64
    control_change::Float64
    estimate::Float64
end

const _PROBABILITY_TOLERANCE = 1.0e-12

function _finite_real(name::AbstractString, value)
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

function _finite_vector(name::AbstractString, values)
    values isa AbstractString &&
        throw(ArgumentError("$name must be a one-dimensional numeric collection"))
    raw = try
        collect(values)
    catch
        throw(ArgumentError("$name must be a one-dimensional numeric collection"))
    end
    ndims(raw) == 1 && !isempty(raw) ||
        throw(ArgumentError("$name must be a nonempty numeric vector"))
    all(value -> value isa Real && !(value isa Bool), raw) ||
        throw(ArgumentError("$name must contain real values"))
    result = try
        Float64.(raw)
    catch
        throw(ArgumentError("$name must contain representable real values"))
    end
    all(isfinite, result) ||
        throw(ArgumentError("$name must contain only finite values"))
    return result
end

function _probability_vector(name::AbstractString, values)
    result = _finite_vector(name, values)
    all(value -> 0.0 <= value <= 1.0, result) ||
        throw(ArgumentError("$name must contain probabilities in [0, 1]"))
    isapprox(
        sum(result),
        1.0;
        rtol = _PROBABILITY_TOLERANCE,
        atol = _PROBABILITY_TOLERANCE,
    ) || throw(ArgumentError("$name must sum to one"))
    return result
end

function _finite_matrix(name::AbstractString, values)
    if values isa AbstractMatrix
        all(value -> value isa Real && !(value isa Bool), values) ||
            throw(ArgumentError("$name must contain real values"))
        result = try
            Float64.(values)
        catch
            throw(ArgumentError("$name must contain representable real values"))
        end
    else
        rows = try
            collect(values)
        catch
            throw(ArgumentError("$name must be a two-dimensional numeric collection"))
        end
        !isempty(rows) || throw(ArgumentError("$name must be nonempty"))
        converted = [_finite_vector("$name row", row) for row in rows]
        width = length(first(converted))
        all(row -> length(row) == width, converted) ||
            throw(ArgumentError("$name rows must have equal length"))
        result = reduce(vcat, permutedims.(converted))
    end
    ndims(result) == 2 ||
        throw(ArgumentError("$name must be a two-dimensional numeric collection"))
    all(isfinite, result) ||
        throw(ArgumentError("$name must contain only finite values"))
    return Matrix{Float64}(result)
end

function average_treatment_effect(
    treated_interventional_mean,
    control_interventional_mean,
)
    treated = _finite_real(
        "treated_interventional_mean",
        treated_interventional_mean,
    )
    control = _finite_real(
        "control_interventional_mean",
        control_interventional_mean,
    )
    return _finite_real("average_treatment_effect", treated - control)
end

function treatment_effect_decomposition(
    observed_treated_mean,
    observed_control_mean,
    counterfactual_control_mean_for_treated,
)
    treated = _finite_real("observed_treated_mean", observed_treated_mean)
    control = _finite_real("observed_control_mean", observed_control_mean)
    counterfactual = _finite_real(
        "counterfactual_control_mean_for_treated",
        counterfactual_control_mean_for_treated,
    )
    observed = _finite_real("observed_difference", treated - control)
    effect = _finite_real(
        "average_treatment_effect_on_treated",
        treated - counterfactual,
    )
    selection = _finite_real(
        "sample_selection_bias",
        counterfactual - control,
    )
    return TreatmentEffectDecomposition(observed, effect, selection)
end

function randomized_mean_difference(treated_outcomes, control_outcomes)
    treated = _finite_vector("treated_outcomes", treated_outcomes)
    control = _finite_vector("control_outcomes", control_outcomes)
    return _finite_real(
        "randomized_mean_difference",
        sum(treated) / length(treated) - sum(control) / length(control),
    )
end

function difference_in_differences(
    treated_before,
    treated_after,
    control_before,
    control_after,
)
    treated_change = _finite_real(
        "treated_change",
        _finite_real("treated_after", treated_after) -
        _finite_real("treated_before", treated_before),
    )
    control_change = _finite_real(
        "control_change",
        _finite_real("control_after", control_after) -
        _finite_real("control_before", control_before),
    )
    estimate = _finite_real(
        "difference_in_differences",
        treated_change - control_change,
    )
    return DifferenceInDifferencesEstimate(
        treated_change,
        control_change,
        estimate,
    )
end

function backdoor_adjusted_expectation(
    conditional_outcome_means,
    adjustment_probabilities,
)
    means = _finite_vector("conditional_outcome_means", conditional_outcome_means)
    probabilities = _probability_vector(
        "adjustment_probabilities",
        adjustment_probabilities,
    )
    length(means) == length(probabilities) || throw(
        ArgumentError(
            "conditional_outcome_means and adjustment_probabilities must have the same length",
        ),
    )
    return _finite_real("backdoor_adjusted_expectation", dot(means, probabilities))
end

function backdoor_adjusted_average_treatment_effect(
    treated_conditional_means,
    control_conditional_means,
    adjustment_probabilities,
)
    treated = _finite_vector(
        "treated_conditional_means",
        treated_conditional_means,
    )
    control = _finite_vector(
        "control_conditional_means",
        control_conditional_means,
    )
    probabilities = _probability_vector(
        "adjustment_probabilities",
        adjustment_probabilities,
    )
    length(treated) == length(control) == length(probabilities) || throw(
        ArgumentError(
            "treated means, control means, and adjustment probabilities must have the same length",
        ),
    )
    return _finite_real(
        "backdoor_adjusted_average_treatment_effect",
        dot(treated - control, probabilities),
    )
end

function frontdoor_adjusted_probability(
    mediator_probabilities_given_treatment,
    outcome_probabilities_given_mediator_and_treatment,
    treatment_probabilities,
)
    mediator_probabilities = _probability_vector(
        "mediator_probabilities_given_treatment",
        mediator_probabilities_given_treatment,
    )
    treatment_weights = _probability_vector(
        "treatment_probabilities",
        treatment_probabilities,
    )
    conditional = _finite_matrix(
        "outcome_probabilities_given_mediator_and_treatment",
        outcome_probabilities_given_mediator_and_treatment,
    )
    all(value -> 0.0 <= value <= 1.0, conditional) || throw(
        ArgumentError(
            "outcome_probabilities_given_mediator_and_treatment must contain probabilities in [0, 1]",
        ),
    )
    size(conditional) ==
    (length(mediator_probabilities), length(treatment_weights)) || throw(
        ArgumentError(
            "outcome probability matrix shape must match mediator and treatment state counts",
        ),
    )
    probability = _finite_real(
        "frontdoor_adjusted_probability",
        dot(mediator_probabilities, conditional * treatment_weights),
    )
    if -_PROBABILITY_TOLERANCE <= probability < 0.0
        return 0.0
    elseif 1.0 < probability <= 1.0 + _PROBABILITY_TOLERANCE
        return 1.0
    end
    return probability
end

function linear_instrumental_variable_effect(
    outcome_instrument_covariance,
    treatment_instrument_covariance,
)
    numerator = _finite_real(
        "outcome_instrument_covariance",
        outcome_instrument_covariance,
    )
    denominator = _finite_real(
        "treatment_instrument_covariance",
        treatment_instrument_covariance,
    )
    denominator == 0.0 &&
        throw(ArgumentError("treatment_instrument_covariance must be nonzero"))
    return _finite_real("instrumental_variable_effect", numerator / denominator)
end
