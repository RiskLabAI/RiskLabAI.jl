function _allocation_real_matrix(name::AbstractString, values)
    values isa AbstractMatrix || throw(ArgumentError("$name must be a numeric matrix"))
    all(value -> value isa Real && !(value isa Bool), values) ||
        throw(ArgumentError("$name must be real-valued and must not contain booleans"))
    result = try
        Matrix{Float64}(values)
    catch
        throw(ArgumentError("$name must be a numeric matrix"))
    end
    all(isfinite, result) || throw(ArgumentError("$name must contain only finite values"))
    return result
end

function _allocation_real_vector(name::AbstractString, values)
    values isa AbstractVector || throw(ArgumentError("$name must be a numeric vector"))
    all(value -> value isa Real && !(value isa Bool), values) ||
        throw(ArgumentError("$name must be real-valued and must not contain booleans"))
    result = try
        Vector{Float64}(values)
    catch
        throw(ArgumentError("$name must be a numeric vector"))
    end
    all(isfinite, result) || throw(ArgumentError("$name must contain only finite values"))
    return result
end

function _allocation_float_snapshot(name::AbstractString, values)
    return _allocation_real_vector(name, values)
end

function _allocation_bool_snapshot(name::AbstractString, values)
    values isa AbstractVector || throw(ArgumentError("$name must be a boolean vector"))
    all(value -> value isa Bool, values) ||
        throw(ArgumentError("$name must be a boolean vector"))
    return Vector{Bool}(values)
end

function _allocation_nonnegative_scalar(name::AbstractString, value)
    value isa Real && !(value isa Bool) ||
        throw(ArgumentError("$name must be a nonnegative finite scalar"))
    result = try
        Float64(value)
    catch
        throw(ArgumentError("$name must be a nonnegative finite scalar"))
    end
    isfinite(result) && result >= 0.0 ||
        throw(ArgumentError("$name must be a nonnegative finite scalar"))
    return result
end

"""
    AllocationMisspecificationDiagnostics

Evidence comparing minimum-variance allocations built from reference and
misspecified exposure systems. Every vector field is an independent concrete
snapshot. `reference_exposure_error` is the reference-system exposure of the
misspecified weights minus `reference_target_exposures`.
"""
struct AllocationMisspecificationDiagnostics
    reference_weights::Vector{Float64}
    misspecified_weights::Vector{Float64}
    reference_target_exposures::Vector{Float64}
    reference_achieved_exposures::Vector{Float64}
    misspecified_reference_exposures::Vector{Float64}
    misspecified_model_target_exposures::Vector{Float64}
    misspecified_model_achieved_exposures::Vector{Float64}
    reference_exposure_error::Vector{Float64}
    weight_sign_reversals::Vector{Bool}
    reference_variance::Float64
    misspecified_variance::Float64

    function AllocationMisspecificationDiagnostics(
        reference_weights,
        misspecified_weights,
        reference_target_exposures,
        reference_achieved_exposures,
        misspecified_reference_exposures,
        misspecified_model_target_exposures,
        misspecified_model_achieved_exposures,
        reference_exposure_error,
        weight_sign_reversals,
        reference_variance,
        misspecified_variance,
    )
        return new(
            _allocation_float_snapshot("reference_weights", reference_weights),
            _allocation_float_snapshot("misspecified_weights", misspecified_weights),
            _allocation_float_snapshot(
                "reference_target_exposures",
                reference_target_exposures,
            ),
            _allocation_float_snapshot(
                "reference_achieved_exposures",
                reference_achieved_exposures,
            ),
            _allocation_float_snapshot(
                "misspecified_reference_exposures",
                misspecified_reference_exposures,
            ),
            _allocation_float_snapshot(
                "misspecified_model_target_exposures",
                misspecified_model_target_exposures,
            ),
            _allocation_float_snapshot(
                "misspecified_model_achieved_exposures",
                misspecified_model_achieved_exposures,
            ),
            _allocation_float_snapshot(
                "reference_exposure_error",
                reference_exposure_error,
            ),
            _allocation_bool_snapshot("weight_sign_reversals", weight_sign_reversals),
            _allocation_nonnegative_scalar("reference_variance", reference_variance),
            _allocation_nonnegative_scalar("misspecified_variance", misspecified_variance),
        )
    end
end

function _validate_allocation_shapes(
    covariance::Matrix{Float64},
    reference_factor_exposures::Matrix{Float64},
    reference_target_exposures::Vector{Float64},
    misspecified_factor_exposures::Matrix{Float64},
    misspecified_target_exposures::Vector{Float64},
)
    n_assets, covariance_columns = size(covariance)
    n_assets > 0 && covariance_columns == n_assets ||
        throw(ArgumentError("covariance must be a nonempty square matrix"))

    for (name, exposures, targets) in (
        ("reference", reference_factor_exposures, reference_target_exposures),
        ("misspecified", misspecified_factor_exposures, misspecified_target_exposures),
    )
        size(exposures, 1) == n_assets || throw(
            ArgumentError(
                "$(name)_factor_exposures must have the same number of rows as covariance",
            ),
        )
        n_factors = size(exposures, 2)
        n_factors > 0 || throw(
            ArgumentError(
                "$(name)_factor_exposures must contain at least one factor column",
            ),
        )
        n_factors <= n_assets || throw(
            ArgumentError("$(name)_factor_exposures cannot have more columns than rows"),
        )
        length(targets) == n_factors || throw(
            ArgumentError(
                "$(name)_target_exposures length must equal the number of $(name) factor columns",
            ),
        )
    end
    return nothing
end

function _allocation_portfolio_variance(
    covariance::Matrix{Float64},
    weights::Vector{Float64},
    name::AbstractString,
)
    result = dot(weights, covariance, weights)
    isfinite(result) && result >= 0.0 ||
        throw(ArgumentError("$name cannot be represented as a nonnegative finite Float64"))
    return result
end

"""
    allocation_misspecification_diagnostics(
        covariance,
        reference_factor_exposures,
        reference_target_exposures,
        misspecified_factor_exposures,
        misspecified_target_exposures,
    )

Solve the two equality-constrained minimum-variance allocations and evaluate
both against the reference covariance and exposure system. The two exposure
matrices must have the same asset rows but may have different factor counts.
A sign reversal is recorded only when both paired weights are nonzero and have
opposite signs.
"""
function allocation_misspecification_diagnostics(
    covariance,
    reference_factor_exposures,
    reference_target_exposures,
    misspecified_factor_exposures,
    misspecified_target_exposures,
)
    covariance_array = _allocation_real_matrix("covariance", covariance)
    reference_exposure_array =
        _allocation_real_matrix("reference_factor_exposures", reference_factor_exposures)
    reference_target_array =
        _allocation_real_vector("reference_target_exposures", reference_target_exposures)
    misspecified_exposure_array = _allocation_real_matrix(
        "misspecified_factor_exposures",
        misspecified_factor_exposures,
    )
    misspecified_target_array = _allocation_real_vector(
        "misspecified_target_exposures",
        misspecified_target_exposures,
    )
    _validate_allocation_shapes(
        covariance_array,
        reference_exposure_array,
        reference_target_array,
        misspecified_exposure_array,
        misspecified_target_array,
    )

    reference_weights = minimum_variance_factor_weights(
        covariance_array,
        reference_exposure_array,
        reference_target_array,
    )
    misspecified_weights = minimum_variance_factor_weights(
        covariance_array,
        misspecified_exposure_array,
        misspecified_target_array,
    )

    reference_achieved_exposures = transpose(reference_exposure_array) * reference_weights
    misspecified_reference_exposures =
        transpose(reference_exposure_array) * misspecified_weights
    misspecified_model_achieved_exposures =
        transpose(misspecified_exposure_array) * misspecified_weights
    reference_exposure_error = misspecified_reference_exposures - reference_target_array
    weight_sign_reversals =
        map(reference_weights, misspecified_weights) do reference, misspecified
            !iszero(reference) &&
                !iszero(misspecified) &&
                signbit(reference) != signbit(misspecified)
        end

    return AllocationMisspecificationDiagnostics(
        reference_weights,
        misspecified_weights,
        reference_target_array,
        reference_achieved_exposures,
        misspecified_reference_exposures,
        misspecified_target_array,
        misspecified_model_achieved_exposures,
        reference_exposure_error,
        weight_sign_reversals,
        _allocation_portfolio_variance(
            covariance_array,
            reference_weights,
            "reference_variance",
        ),
        _allocation_portfolio_variance(
            covariance_array,
            misspecified_weights,
            "misspecified_variance",
        ),
    )
end
