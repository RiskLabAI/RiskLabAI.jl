struct PopulationRegressionDiagnostics
    coefficient_names::Tuple{Vararg{String}}
    coefficients::Tuple{Vararg{Float64}}
    r_squared::Float64
end

struct RegressionDiagnostics
    coefficient_names::Tuple{Vararg{String}}
    coefficients::Tuple{Vararg{Float64}}
    standard_errors::Tuple{Vararg{Float64}}
    t_statistics::Tuple{Vararg{Float64}}
    r_squared::Float64
    adjusted_r_squared::Float64
    residual_variance::Float64
    n_observations::Int
end

struct SpecificationPopulationResult
    structure::String
    conditioning_variable_role::String
    unconditioned::PopulationRegressionDiagnostics
    conditioned::PopulationRegressionDiagnostics
end

struct SpecificationExperimentResult
    structure::String
    conditioning_variable_role::String
    unconditioned::RegressionDiagnostics
    conditioned::RegressionDiagnostics
end

const _MAX_SAMPLE_SIZE = 1_000_000

function _sample_size(value)
    value isa Bool && throw(ArgumentError("sample_size must be an integer"))
    value isa Integer || throw(ArgumentError("sample_size must be an integer"))
    result = Int(value)
    result > 3 || throw(ArgumentError("sample_size must be greater than 3"))
    result <= _MAX_SAMPLE_SIZE ||
        throw(ArgumentError("sample_size must not exceed $_MAX_SAMPLE_SIZE"))
    return result
end

function _random_seed(value)
    value isa Bool && throw(ArgumentError("seed must be an integer"))
    value isa Integer || throw(ArgumentError("seed must be an integer"))
    0 <= value <= typemax(UInt32) ||
        throw(ArgumentError("seed must be between 0 and 2^32 - 1"))
    return UInt32(value)
end

function _population_regression(names, coefficients, r_squared)
    return PopulationRegressionDiagnostics(
        Tuple(String.(names)),
        Tuple(Float64.(coefficients)),
        Float64(r_squared),
    )
end

function _ols_diagnostics(outcome::Vector{Float64}, regressors)
    n_observations = length(outcome)
    columns = Any[ones(Float64, n_observations)]
    names = String["intercept"]
    for (name, values) in regressors
        length(values) == n_observations ||
            throw(ArgumentError("regressor length must equal outcome length"))
        push!(columns, values)
        push!(names, String(name))
    end
    design = hcat(columns...)
    n_parameters = size(design, 2)
    n_observations > n_parameters ||
        throw(ArgumentError("the regression requires positive residual degrees of freedom"))
    rank(design) == n_parameters ||
        throw(ArgumentError("the generated regression design is rank deficient"))

    coefficients = design \ outcome
    fitted = design * coefficients
    residuals = outcome - fitted
    residual_sum_squares = dot(residuals, residuals)
    centered = outcome .- mean(outcome)
    total_sum_squares = dot(centered, centered)
    isfinite(total_sum_squares) && total_sum_squares > 0.0 ||
        throw(ArgumentError("the generated outcome must have positive finite variance"))

    residual_degrees_of_freedom = n_observations - n_parameters
    residual_variance = residual_sum_squares / residual_degrees_of_freedom
    coefficient_covariance = residual_variance * inv(transpose(design) * design)
    variances = diag(coefficient_covariance)
    all(value -> value >= -1.0e-12, variances) ||
        throw(ArgumentError("the coefficient variance calculation is invalid"))
    standard_errors = sqrt.(max.(variances, 0.0))
    t_statistics = map(coefficients, standard_errors) do coefficient, standard_error
        standard_error > 0.0 ? coefficient / standard_error : NaN
    end
    all(isfinite, coefficients) && all(isfinite, standard_errors) ||
        throw(ArgumentError("the regression diagnostics must be finite"))

    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    if -1.0e-12 <= r_squared < 0.0
        r_squared = 0.0
    elseif 1.0 < r_squared <= 1.0 + 1.0e-12
        r_squared = 1.0
    end
    adjusted_r_squared =
        1.0 - (1.0 - r_squared) * (n_observations - 1) / residual_degrees_of_freedom
    return RegressionDiagnostics(
        Tuple(names),
        Tuple(Float64.(coefficients)),
        Tuple(Float64.(standard_errors)),
        Tuple(Float64.(t_statistics)),
        Float64(r_squared),
        Float64(adjusted_r_squared),
        Float64(residual_variance),
        n_observations,
    )
end

function fork_population_diagnostics()
    return SpecificationPopulationResult(
        "fork",
        "confounder",
        _population_regression(("intercept", "X"), (0.0, 0.5), 0.25),
        _population_regression(("intercept", "X", "Z"), (0.0, 0.0, 1.0), 0.5),
    )
end

function collider_population_diagnostics()
    return SpecificationPopulationResult(
        "collider",
        "collider",
        _population_regression(("intercept", "X"), (0.0, 0.0), 0.0),
        _population_regression(("intercept", "X", "Z"), (0.0, -0.5, 0.5), 0.5),
    )
end

function confounded_mediator_population_diagnostics()
    return SpecificationPopulationResult(
        "confounded_mediator",
        "confounded_mediator",
        _population_regression(("intercept", "X"), (0.0, 1.0), 1.0 / 7.0),
        _population_regression(("intercept", "X", "Z"), (0.0, -0.5, 1.5), 11.0 / 14.0),
    )
end

function fork_specification_experiment(sample_size = 5_000, seed = 0)
    n_observations = _sample_size(sample_size)
    random_state = MersenneTwister(_random_seed(seed))
    z = randn(random_state, n_observations)
    x = z + randn(random_state, n_observations)
    y = z + randn(random_state, n_observations)
    return SpecificationExperimentResult(
        "fork",
        "confounder",
        _ols_diagnostics(y, (("X", x),)),
        _ols_diagnostics(y, (("X", x), ("Z", z))),
    )
end

function collider_specification_experiment(sample_size = 5_000, seed = 0)
    n_observations = _sample_size(sample_size)
    random_state = MersenneTwister(_random_seed(seed))
    x = randn(random_state, n_observations)
    y = randn(random_state, n_observations)
    z = x + y + randn(random_state, n_observations)
    return SpecificationExperimentResult(
        "collider",
        "collider",
        _ols_diagnostics(y, (("X", x),)),
        _ols_diagnostics(y, (("X", x), ("Z", z))),
    )
end

function confounded_mediator_specification_experiment(sample_size = 5_000, seed = 0)
    n_observations = _sample_size(sample_size)
    random_state = MersenneTwister(_random_seed(seed))
    x = randn(random_state, n_observations)
    w = randn(random_state, n_observations)
    z = x + w + randn(random_state, n_observations)
    y = z + w + randn(random_state, n_observations)
    return SpecificationExperimentResult(
        "confounded_mediator",
        "confounded_mediator",
        _ols_diagnostics(y, (("X", x),)),
        _ols_diagnostics(y, (("X", x), ("Z", z))),
    )
end
