"""
    KernelDensityEstimate

An immutable one-dimensional fitted kernel density with equal observation
weights. Evaluate log densities with [`score_samples`](@ref). The fitted
observations are copied into a tuple, so later input mutations have no effect.
"""
struct KernelDensityEstimate
    observations::Tuple{Vararg{Float64}}
    bandwidth::Float64
    kernel::String
    function KernelDensityEstimate(observations, bandwidth, kernel)
        values = _density_values(observations, "observations")
        bandwidth isa Real && isfinite(bandwidth) && bandwidth > 0 || throw(ArgumentError("bandwidth must be finite and positive"))
        width = Float64(bandwidth)
        isfinite(width) && width > 0 || throw(ArgumentError("bandwidth must be representable as a positive Float64"))
        kernel isa AbstractString && kernel in ("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine") || throw(ArgumentError("unsupported one-dimensional kernel"))
        return new(Tuple(values), width, String(kernel))
    end
end

function _density_values(values, name)
    values isa AbstractArray && !isempty(values) && all(value -> value isa Real && isfinite(value), values) || throw(ArgumentError("$name must contain finite real observations"))
    result = vec(Float64.(values))
    all(isfinite, result) || throw(ArgumentError("$name is outside the Float64 range"))
    return result
end

"""
    fit_kde(observations; bandwidth=0.01, kernel="gaussian")

Fit the public univariate, equally weighted density contract. Supported kernels
are Gaussian, tophat, Epanechnikov, exponential, linear and cosine. Arrays are
flattened to univariate observations, and bandwidth is supplied in data units.
Use `score_samples(estimate, query)` for log densities on a vector or one-column
matrix. Finite-support kernels exclude points exactly one bandwidth away.
"""
fit_kde(observations; bandwidth = 0.01, kernel = "gaussian") = KernelDensityEstimate(observations, bandwidth, kernel)
fit_kde(observations, bandwidth, kernel = "gaussian") = fit_kde(observations; bandwidth, kernel)

function _density_log_kernel(distance, kernel)
    kernel == "gaussian" && return -0.5 * distance^2 - 0.5 * log(2pi)
    kernel == "exponential" && return -distance - log(2)
    distance < 1 || return -Inf
    kernel == "tophat" && return -log(2)
    kernel == "epanechnikov" && return log(0.75) + log1p(-distance^2)
    kernel == "linear" && return log1p(-distance)
    return log(pi / 4) + log(cos(pi * distance / 2))
end

"""
    score_samples(estimate::KernelDensityEstimate, query)

Return the log density at each query using a stable log-sum-exp mixture.
Queries are a nonempty finite vector or one-column matrix; values outside all
compact kernel supports have log density `-Inf`.
"""
function score_samples(estimate::KernelDensityEstimate, query)
    query isa AbstractVector || query isa AbstractMatrix && size(query, 2) == 1 || throw(ArgumentError("query must be a vector or one-column matrix"))
    points = _density_values(query, "query")
    result = similar(points)
    normalizer = log(length(estimate.observations)) + log(estimate.bandwidth)
    terms = Vector{Float64}(undef, length(estimate.observations))
    for (index, point) in enumerate(points)
        for (observation, value) in enumerate(estimate.observations)
            distance = abs(point - value)
            standardized = isfinite(distance) ? distance / estimate.bandwidth : abs(point / estimate.bandwidth - value / estimate.bandwidth)
            terms[observation] = _density_log_kernel(standardized, estimate.kernel)
        end
        largest = maximum(terms)
        result[index] = largest == -Inf ? -Inf : largest + log(sum(exp(value - largest) for value in terms)) - normalizer
    end
    return result
end

"""
    denoised_corr2(eigenvalues, eigenvectors, num_factors; alpha=0.0)

Retain the leading supplied eigenpairs and shrink only the off-diagonal noise
contribution by `alpha`. `eigenvalues` is a square diagonal matrix, with entries
off its diagonal allowed only within absolute tolerance `1e-12`. The caller
supplies the eigenpair order and signal count. This construction preserves the
diagonal contribution and does not normalize the result to unit diagonal.
"""
function denoised_corr2(eigenvalues, eigenvectors, num_factors; alpha = 0.0)
    eigenvalues isa AbstractMatrix && size(eigenvalues, 1) == size(eigenvalues, 2) && !isempty(eigenvalues) || throw(ArgumentError("eigenvalues must be a nonempty square diagonal matrix"))
    eigenvectors isa AbstractMatrix && size(eigenvectors) == size(eigenvalues) || throw(ArgumentError("eigenvectors must be square and match eigenvalues"))
    for array in (eigenvalues, eigenvectors)
        all(value -> value isa Real && !(value isa Bool) && isfinite(value), array) || throw(ArgumentError("eigenpairs must contain finite real numbers"))
    end
    values, vectors = Float64.(eigenvalues), Float64.(eigenvectors)
    all(isfinite, values) && all(isfinite, vectors) || throw(ArgumentError("eigenpairs are outside the Float64 range"))
    all(abs.(values - Diagonal(diag(values))) .<= 1e-12) || throw(ArgumentError("eigenvalues must be diagonal"))
    num_factors isa Integer && !(num_factors isa Bool) && 0 <= num_factors <= size(values, 1) || throw(ArgumentError("num_factors must be an integer between zero and the dimension"))
    alpha isa Real && !(alpha isa Bool) && isfinite(alpha) && 0 <= alpha <= 1 || throw(ArgumentError("alpha must be finite and lie in [0, 1]"))
    signal_vectors, noise_vectors = vectors[:, 1:num_factors], vectors[:, num_factors+1:end]
    signal = signal_vectors * values[1:num_factors, 1:num_factors] * signal_vectors'
    noise = noise_vectors * values[num_factors+1:end, num_factors+1:end] * noise_vectors'
    return signal + Float64(alpha) * noise + (1 - Float64(alpha)) * Diagonal(diag(noise))
end

denoised_corr2(eigenvalues, eigenvectors, num_factors, alpha) = denoised_corr2(eigenvalues, eigenvectors, num_factors; alpha)
