const _SYMMETRY_RTOL = 1.0e-10
const _SYMMETRY_ATOL = 1.0e-12
const _FEASIBILITY_RTOL = 1.0e-10
const _FEASIBILITY_ATOL = 1.0e-12

function _real_matrix(name::AbstractString, values)
    values isa AbstractMatrix ||
        throw(ArgumentError("$name must be a numeric matrix"))
    all(value -> value isa Real, values) ||
        throw(ArgumentError("$name must be real-valued"))
    result = try
        Matrix{Float64}(values)
    catch
        throw(ArgumentError("$name must be a numeric matrix"))
    end
    all(isfinite, result) ||
        throw(ArgumentError("$name must contain only finite values"))
    return result
end

function _real_vector(name::AbstractString, values)
    values isa AbstractVector ||
        throw(ArgumentError("$name must be a numeric vector"))
    all(value -> value isa Real, values) ||
        throw(ArgumentError("$name must be real-valued"))
    result = try
        Vector{Float64}(values)
    catch
        throw(ArgumentError("$name must be a numeric vector"))
    end
    all(isfinite, result) ||
        throw(ArgumentError("$name must contain only finite values"))
    return result
end

function minimum_variance_factor_weights(
    covariance,
    factor_exposures,
    target_exposures,
)
    covariance_array = _real_matrix("covariance", covariance)
    exposure_array = _real_matrix("factor_exposures", factor_exposures)
    target_array = _real_vector("target_exposures", target_exposures)

    n_assets, covariance_columns = size(covariance_array)
    n_assets > 0 && covariance_columns == n_assets ||
        throw(ArgumentError("covariance must be a nonempty square matrix"))
    size(exposure_array, 1) == n_assets || throw(
        ArgumentError(
            "factor_exposures must have the same number of rows as covariance",
        ),
    )
    n_factors = size(exposure_array, 2)
    n_factors > 0 ||
        throw(ArgumentError("factor_exposures must contain a factor column"))
    n_factors <= n_assets || throw(
        ArgumentError("factor_exposures cannot have more columns than rows"),
    )
    length(target_array) == n_factors || throw(
        ArgumentError(
            "target_exposures length must equal the number of factor columns",
        ),
    )

    covariance_scale = maximum(abs, covariance_array)
    covariance_scale > 0.0 ||
        throw(ArgumentError("covariance must be positive definite"))
    scaled_covariance = covariance_array / covariance_scale
    isapprox(
        scaled_covariance,
        transpose(scaled_covariance);
        rtol = _SYMMETRY_RTOL,
        atol = _SYMMETRY_ATOL,
    ) || throw(ArgumentError("covariance must be symmetric"))
    scaled_covariance =
        0.5 .* scaled_covariance .+ 0.5 .* transpose(scaled_covariance)
    covariance_cholesky = try
        cholesky(Symmetric(scaled_covariance); check = true)
    catch
        throw(ArgumentError("covariance must be positive definite"))
    end

    maximum_exposures = vec(maximum(abs.(exposure_array); dims = 1))
    all(value -> value > 0.0, maximum_exposures) ||
        throw(ArgumentError("factor_exposures must have full column rank"))
    target_scales = abs.(target_array) ./ (floatmax(Float64) / 2.0)
    constraint_scales = max.(maximum_exposures, target_scales)
    scaled_exposures = exposure_array ./ transpose(constraint_scales)
    scaled_targets = target_array ./ constraint_scales
    any((target_array .!= 0.0) .& (scaled_targets .== 0.0)) && throw(
        ArgumentError(
            "factor constraints could not be represented at numerical precision",
        ),
    )
    all(isfinite, scaled_exposures) && all(isfinite, scaled_targets) || throw(
        ArgumentError(
            "factor constraints could not be represented at numerical precision",
        ),
    )

    lower = covariance_cholesky.L
    whitened_exposures = lower \ scaled_exposures
    all(isfinite, whitened_exposures) || throw(
        ArgumentError("factor_exposures produce non-finite whitened values"),
    )
    rank(whitened_exposures) == n_factors ||
        throw(ArgumentError("factor_exposures must have full column rank"))
    if n_factors == n_assets
        weights = transpose(scaled_exposures) \ scaled_targets
    else
        factorization = qr(whitened_exposures)
        orthonormal_basis = Matrix(factorization.Q)[:, 1:n_factors]
        triangular_factor = Matrix(factorization.R)[1:n_factors, 1:n_factors]
        transformed_targets = transpose(triangular_factor) \ scaled_targets
        whitened_weights = orthonormal_basis * transformed_targets
        weights = transpose(lower) \ whitened_weights
    end
    all(isfinite, weights) ||
        throw(ArgumentError("the allocation produced non-finite weights"))

    achieved_exposures = transpose(scaled_exposures) * weights
    residuals = abs.(achieved_exposures - scaled_targets)
    tolerances = map(scaled_targets) do target
        target != 0.0 ? _FEASIBILITY_RTOL * abs(target) : _FEASIBILITY_ATOL
    end
    all(isfinite, achieved_exposures) && all(residuals .<= tolerances) || throw(
        ArgumentError(
            "factor constraints could not be resolved to numerical precision",
        ),
    )
    return Vector{Float64}(weights)
end
