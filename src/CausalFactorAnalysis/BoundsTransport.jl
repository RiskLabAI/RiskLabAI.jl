"""
    manski_ate_bounds(outcomes, treatment; lower, upper)

Return worst-case finite-population ATE bounds under consistency and supplied
finite potential-outcome limits. No exchangeability or sampling confidence
interval is assumed. All-treated and all-control populations are allowed.
"""
function manski_ate_bounds(outcomes, treatment; lower, upper)
    outcomes = _real_array(outcomes, "outcomes", 1)
    lower = _finite_real("lower", lower)
    upper = _finite_real("upper", upper)
    treatment isa AbstractVector &&
    eltype(treatment) <: Real &&
    size(treatment) == size(outcomes) &&
    all(value -> value == 0 || value == 1, treatment) &&
    lower <= upper &&
    all(value -> lower <= value <= upper, outcomes) || throw(
        ArgumentError(
            "require binary matching treatment and outcomes within ordered bounds",
        ),
    )
    lower_effect = [
        treatment[i] == 1 ? outcomes[i] - upper : lower - outcomes[i] for
        i in eachindex(outcomes)
    ]
    upper_effect = [
        treatment[i] == 1 ? outcomes[i] - lower : upper - outcomes[i] for
        i in eachindex(outcomes)
    ]
    return (
        lower = mean(lower_effect),
        upper = mean(upper_effect),
        treated_fraction = mean(treatment),
        sample_count = length(outcomes),
    )
end

"""
    transported_ate(treated_means, control_means, target_probabilities,
                    source_treated_counts, source_control_counts)

Standardize supported stratum effects to a supplied target population. Every
target-positive stratum needs source observations of both treatments. Counts
may be nonnegative real weights. Zero-target strata contribute nothing, but
all supplied means must be finite. Source exchangeability, consistency, shared
treatment definitions and transportability of conditional means are caller
assumptions and are not certified by these numerical checks.
"""
function transported_ate(
    treated_means,
    control_means,
    target_probabilities,
    source_treated_counts,
    source_control_counts,
)
    treated = _real_array(treated_means, "treated_means", 1)
    control = _real_array(control_means, "control_means", 1)
    probabilities = _probabilities(target_probabilities, "target_probabilities", 1)
    count1 = _real_array(source_treated_counts, "source_treated_counts", 1)
    count0 = _real_array(source_control_counts, "source_control_counts", 1)
    all(
        values -> size(values) == size(treated),
        (control, probabilities, count1, count0),
    ) &&
    all(>=(0), count1) &&
    all(>=(0), count0) ||
        throw(ArgumentError("stratum arrays must match with nonnegative source counts"))
    any((probabilities .> 0) .& ((count1 .== 0) .| (count0 .== 0))) &&
        throw(ArgumentError("target-positive strata lack source treatment support"))
    return backdoor_adjusted_average_treatment_effect(treated, control, probabilities)
end
