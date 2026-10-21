using Distributions: Normal, cdf

function _feature_probability_rows(values, name, dimensions)
    probabilities = _feature_real_array(values, name, dimensions)
    all(0 .<= probabilities .<= 1) && all(abs.(sum(probabilities; dims = dimensions) .- 1) .<= 2e-12) || throw(ArgumentError("$name must contain nonnegative probability rows summing to one"))
    return probabilities
end

"""
    probability_ambiguity(probabilities; prior_weights=nothing, bin_width=nothing)

Compute the sum of prior-mean bin probabilities times their population
variances over the supplied finite prior distribution. Rows are priors and
columns are exhaustive mutually exclusive bins. Weights are explicit or
uniform and are never normalized automatically. Optional `bin_width` in
`(0,1)` divides the measure by `bin_width*(1-bin_width)`. This is the approved
supplied-prior measure, separate from return variance and raw intraday
probability estimation.
"""
function probability_ambiguity(probabilities; prior_weights = nothing, bin_width = nothing)
    probabilities = _feature_probability_rows(probabilities, "probabilities", 2)
    n = size(probabilities, 1)
    weights = isnothing(prior_weights) ? fill(1 / n, n) : _feature_probability_rows(prior_weights, "prior_weights", 1)
    length(weights) == n || throw(ArgumentError("there must be one weight per prior"))
    means = vec(weights' * probabilities)
    variances = vec(weights' * (probabilities .- permutedims(means)).^2)
    value = sum(means .* variances)
    width = isnothing(bin_width) ? nothing : _feature_real_scalar(bin_width, "bin_width"; minimum = 0, strict = true)
    isnothing(width) || width < 1 || throw(ArgumentError("bin_width must be less than one"))
    return (unscaled = value, scaled = isnothing(width) ? nothing : value / (width * (1 - width)),
        mean_probabilities = means, probability_variances = variances, prior_weights = weights, bin_width = width)
end

"""
    normal_prior_bin_probabilities(means, standard_deviations; edges=nothing)

Convert finite normal priors with positive standard deviations to exhaustive
bin probabilities, including both tails. Means, scales and edges share return
units. Default finite edges range from -0.06 to 0.06 in steps of 0.002.
Positive-tail intervals use survival probabilities to avoid cancellation.
"""
function normal_prior_bin_probabilities(means, standard_deviations; edges = nothing)
    means = _feature_real_array(means, "means", 1)
    scales = _feature_real_array(standard_deviations, "standard_deviations", 1)
    edges = isnothing(edges) ? collect(range(-0.06, 0.06; length = 61)) : _feature_real_array(edges, "edges", 1)
    length(scales) == length(means) && all(>(0), scales) && all(>(0), diff(edges)) || throw(ArgumentError("scales must match means and be positive; edges must increase"))
    result = Matrix{Float64}(undef, length(means), length(edges) + 1)
    normal = Normal()
    for i in eachindex(means)
        z = (edges .- means[i]) / scales[i]
        result[i, 1] = cdf(normal, z[1])
        for j in 1:length(z)-1
            result[i, j + 1] = z[j] >= 0 ? cdf(normal, -z[j]) - cdf(normal, -z[j + 1]) : cdf(normal, z[j + 1]) - cdf(normal, z[j])
        end
        result[i, end] = cdf(normal, -z[end])
    end
    return result
end
