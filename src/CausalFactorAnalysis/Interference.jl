function _column_indices(values, row_count, column_count, name)
    values isa AbstractVector &&
    eltype(values) <: Integer &&
    !(eltype(values) <: Bool) &&
    length(values) == row_count &&
    all(value -> 1 <= value <= column_count, values) ||
        throw(ArgumentError("$name must be matching one-based integer column indices"))
    return Int.(values)
end

"""
    exposure_mean_contrast(outcomes, exposures, exposure_probabilities; first, second)

Return the Horvitz-Thompson mean contrast `first - second` under the supplied
randomized exposure design. Labels and contrast columns use one-based indices.
Every unit needs positive probability for both contrasted exposures and its
observed exposure. Row probability sums may be below one for omitted categories.
The exposure mapping and potential-outcome interpretation are caller premises.
No standard error or confidence interval is returned without joint exposure
probabilities.
"""
function exposure_mean_contrast(outcomes, exposures, exposure_probabilities; first, second)
    outcomes = _real_array(outcomes, "outcomes", 1)
    probabilities = _real_array(exposure_probabilities, "exposure_probabilities", 2)
    size(probabilities, 1) == length(outcomes) ||
        throw(ArgumentError("probability rows must match outcomes"))
    labels =
        _column_indices(exposures, length(outcomes), size(probabilities, 2), "exposures")
    for value in (first, second)
        value isa Integer && !(value isa Bool) && 1 <= value <= size(probabilities, 2) ||
            throw(ArgumentError("contrast indices must name probability columns"))
    end
    first != second &&
    all(>=(0), probabilities) &&
    all(<=(1 + 1e-12), sum(probabilities; dims = 2)) ||
        throw(ArgumentError("require distinct contrasts and valid probability mass"))
    all(>(0), probabilities[:, [first, second]]) &&
    all(probabilities[i, labels[i]] > 0 for i in eachindex(outcomes)) ||
        throw(ArgumentError("the requested exposure contrast lacks positivity"))
    first_scores = (labels .== first) .* outcomes ./ probabilities[:, first]
    second_scores = (labels .== second) .* outcomes ./ probabilities[:, second]
    contributions = first_scores - second_scores
    all(isfinite, first_scores) &&
    all(isfinite, second_scores) &&
    all(isfinite, contributions) ||
        throw(ArgumentError("exposure scores exceed floating-point range"))
    return (
        first_mean = mean(first_scores),
        second_mean = mean(second_scores),
        contrast = mean(contributions),
        unit_contributions = contributions,
    )
end
