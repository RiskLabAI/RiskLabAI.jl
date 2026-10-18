function _timegan_lag(array)
    features = size(array, 3)
    left = reshape(permutedims(array[:, 1:end-1, :], (2, 1, 3)), :, features)
    right = reshape(permutedims(array[:, 2:end, :], (2, 1, 3)), :, features)
    centered_left = left .- mean(left; dims = 1)
    centered_right = right .- mean(right; dims = 1)
    numerator = vec(sum(centered_left .* centered_right; dims = 1))
    denominator = sqrt.(vec(sum(abs2, centered_left; dims = 1)) .* vec(sum(abs2, centered_right; dims = 1)))
    return [denominator[index] > 0 ? numerator[index] / denominator[index] : NaN for index in eachindex(denominator)]
end

function _timegan_flatten(array)
    return permutedims(reshape(permutedims(array, (3, 2, 1)), :, size(array, 1)))
end

function _timegan_scaler(features)
    center = vec(mean(features; dims = 1))
    centered = features .- center'
    variance = vec(sum(abs2, centered; dims = 1)) ./ size(features, 1)
    bound = size(features, 1) .* eps(Float64) .* variance .+ (size(features, 1) .* center .* eps(Float64)).^2
    scale = sqrt.(variance)
    scale[variance .<= bound] .= 1.0
    return center, scale
end

function _timegan_predictive(train, validation)
    features = _timegan_flatten(train[:, 1:end-1, :])
    outcomes = train[:, end, :]
    center, scale = _timegan_scaler(features)
    standardized = (features .- center') ./ scale'
    input_center = vec(mean(standardized; dims = 1))
    outcome_center = vec(mean(outcomes; dims = 1))
    design = standardized .- input_center'
    weights = (design' * design + LinearAlgebra.I) \ (design' * (outcomes .- outcome_center'))
    test = (_timegan_flatten(validation[:, 1:end-1, :]) .- center') ./ scale'
    prediction = (test .- input_center') * weights .+ outcome_center'
    return mean(abs2, prediction .- validation[:, end, :])
end

function _timegan_logistic_loss!(gradient, parameters, features, outcomes, weights)
    coefficients, intercept = parameters[1:end-1], parameters[end]
    scores = features * coefficients .+ intercept
    signed = (1 .- 2 .* outcomes) .* scores
    loss = sum(weights .* (max.(signed, 0.0) .+ log1p.(exp.(-abs.(signed)))))
    residuals = weights .* (_timegan_sigmoid.(scores) .- outcomes)
    total_weight = sum(weights)
    gradient[1:end-1] .= (features' * residuals .+ coefficients) ./ total_weight
    gradient[end] = sum(residuals) / total_weight
    return (loss + sum(abs2, coefficients) / 2) / total_weight
end

function _timegan_discrimination(real_train, real_validation, synthetic_train, synthetic_validation)
    train = vcat(_timegan_flatten(real_train), _timegan_flatten(synthetic_train))
    real_count, fake_count = size(real_train, 1), size(synthetic_train, 1)
    labels = [ones(real_count); zeros(fake_count)]
    weights = [fill(length(labels) / (2real_count), real_count); fill(length(labels) / (2fake_count), fake_count)]
    center, scale = _timegan_scaler(train)
    features = (train .- center') ./ scale'
    objective(parameters) = _timegan_logistic_loss!(zeros(length(parameters)), parameters, features, labels, weights)
    gradient!(gradient, parameters) = (_timegan_logistic_loss!(gradient, parameters, features, labels, weights); gradient)
    fit = Optim.optimize(objective, gradient!, zeros(size(features, 2) + 1), Optim.LBFGS(), Optim.Options(g_tol = 1e-10, f_reltol = 1e-14, iterations = 1000))
    Optim.converged(fit) && isfinite(Optim.minimum(fit)) && all(isfinite, Optim.minimizer(fit)) || error("TimeGAN descriptive discriminator failed to converge")
    parameters = Optim.minimizer(fit)
    test = (vcat(_timegan_flatten(real_validation), _timegan_flatten(synthetic_validation)) .- center') ./ scale'
    predictions = test * parameters[1:end-1] .+ parameters[end] .> 0
    split = size(real_validation, 1)
    return (mean(predictions[1:split]) + mean(.!predictions[split+1:end])) / 2
end

"""
    timegan_quality_report(real_training, real_validation,
        synthetic_training, synthetic_validation)

Return descriptive lag-one correlation gaps, train-synthetic/test-real and
real-trained next-step MSE, and held-out balanced real/fake accuracy. Ridge
uses alpha one after training-only standardization; logistic discrimination
uses balanced class weights, fixed C one and an unpenalized intercept.
The native logistic optimizer uses tighter stopping criteria for the same
objective. All arrays use the same units and window/time/feature shapes.

Real validation must come from a disjoint chronological segment and synthetic
validation from fresh noise. The caller supplies this separation. These are
light descriptive diagnostics, not learned GRU evaluation, uncertainty for
overlapping windows, certification of realistic dynamics or parameter selection.
"""
function timegan_quality_report(real_training, real_validation, synthetic_training, synthetic_validation)
    arrays = map(value -> _timegan_array(value, "windows", 3), (real_training, real_validation, synthetic_training, synthetic_validation))
    all(array -> size(array)[2:3] == size(first(arrays))[2:3] && size(array, 1) >= 2, arrays) && size(first(arrays), 2) >= 2 || throw(ArgumentError("All window shapes must agree and need at least two samples/times"))
    rt, rv, st, sv = arrays
    real_lag, synthetic_lag = _timegan_lag(rv), _timegan_lag(sv)
    return (real_lag_one_correlation = real_lag, synthetic_lag_one_correlation = synthetic_lag,
        lag_one_absolute_gap = abs.(real_lag .- synthetic_lag), synthetic_trained_prediction_mse = _timegan_predictive(st, rv),
        real_trained_prediction_mse = _timegan_predictive(rt, rv), balanced_discriminator_accuracy = _timegan_discrimination(rt, rv, st, sv))
end
