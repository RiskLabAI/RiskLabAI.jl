import DataFrames: transform

function _sizing_values(values, name)
    if values isa Real
        result = Float64(values)
        isfinite(result) || throw(ArgumentError("$name must be finite"))
        return result
    end
    values isa AbstractArray && all(value -> value isa Real && isfinite(value), values) || throw(ArgumentError("$name must contain finite real values"))
    result = Float64.(values)
    all(isfinite, result) || throw(ArgumentError("$name is outside the Float64 range"))
    return result
end

"""
    kelly_bet_fraction(probability; payout_ratio=1.0, max_fraction=1.0)

Return the binary-bet fraction `(p*(b+1)-1)/b`, clipped to `[0,max_fraction]`,
where `b` is the positive net win payout and a loss costs one unit staked.
Scalar and array shapes are retained. The cap must be finite and nonnegative.
"""
function kelly_bet_fraction(probability; payout_ratio = 1.0, max_fraction = 1.0)
    values = _sizing_values(probability, "probability")
    payout = _tail_real_scalar(payout_ratio, "payout_ratio")
    cap = _tail_real_scalar(max_fraction, "max_fraction")
    payout > 0 && cap >= 0 || throw(ArgumentError("payout_ratio must be positive and max_fraction nonnegative"))
    return clamp.((values .* (payout + 1) .- 1) ./ payout, 0.0, cap)
end

kelly_bet_fraction(probability, payout_ratio, max_fraction = 1.0) = kelly_bet_fraction(probability; payout_ratio, max_fraction)

"""
    distributionally_robust_kelly_fraction(probability_estimate, n_observations;
        payout_ratio=1.0, max_fraction=1.0, confidence_z=1.0)

Evaluate long-only Kelly at the pessimistic edge of the supplied standard-error
box. Preserve the public clipping of estimates and lower bounds to
`[1e-4,1-1e-4]`, and the denominator floor of one observation. The confidence
multiplier is finite and nonnegative. This is the public box-ambiguity model;
no general ambiguity-set optimizer or interval-coverage guarantee is implied.
"""
function distributionally_robust_kelly_fraction(probability_estimate, n_observations;
    payout_ratio = 1.0, max_fraction = 1.0, confidence_z = 1.0)
    values = clamp.(_sizing_values(probability_estimate, "probability_estimate"), 1e-4, 1 - 1e-4)
    n_observations isa Integer && !(n_observations isa Bool) || throw(ArgumentError("n_observations must be an integer"))
    count = max(_tail_real_scalar(n_observations, "n_observations"), 1.0)
    multiplier = _tail_real_scalar(confidence_z, "confidence_z")
    multiplier >= 0 || throw(ArgumentError("confidence_z must be nonnegative"))
    standard_error = sqrt.(values .* (1 .- values) ./ count)
    lower = clamp.(values .- multiplier .* standard_error, 1e-4, 1 - 1e-4)
    return kelly_bet_fraction(lower; payout_ratio, max_fraction)
end

distributionally_robust_kelly_fraction(probability_estimate, n_observations, payout_ratio, max_fraction = 1.0, confidence_z = 1.0) = distributionally_robust_kelly_fraction(probability_estimate, n_observations; payout_ratio, max_fraction, confidence_z)

"""
    expected_calibration_error(probabilities, outcomes; n_bins=10)

Return the bin-mass-weighted absolute confidence/accuracy gap. Equal-width bins
are right-closed, with zero included in the first bin, matching the public
floating-point edge construction. Probabilities and binary outcomes are
nonempty vectors of equal length.
"""
function expected_calibration_error(probabilities, outcomes; n_bins = 10)
    probabilities isa AbstractVector && outcomes isa AbstractVector && !isempty(probabilities) && length(probabilities) == length(outcomes) || throw(ArgumentError("probabilities and outcomes must be nonempty equal-length vectors"))
    values = _sizing_values(probabilities, "probabilities")
    all(value -> 0 <= value <= 1, values) && all(value -> value isa Real && value in (0, 1), outcomes) || throw(ArgumentError("require probabilities in [0,1] and binary outcomes"))
    n_bins isa Integer && !(n_bins isa Bool) && n_bins > 0 || throw(ArgumentError("n_bins must be a positive integer"))
    edges = [index * (1.0 / n_bins) for index in 0:n_bins]
    edges[end] = 1.0
    error = 0.0
    for index in 1:n_bins
        selected = (index == 1 ? values .>= edges[index] : values .> edges[index]) .& (values .<= edges[index+1])
        count = sum(selected)
        count == 0 && continue
        error += count / length(values) * abs(mean(outcomes[selected]) - mean(values[selected]))
    end
    return error
end

expected_calibration_error(probabilities, outcomes, n_bins) = expected_calibration_error(probabilities, outcomes; n_bins)

"""
    PlattCalibrator()

Native logistic calibration of supplied probabilities on a separate calibration
block. Fit with `fit!` and evaluate with `transform`. The predictor is the
logit clipped at probabilities `1e-4` and `1-1e-4`; slope regularization matches
the public logistic model's `C=1e6`, with an unpenalized intercept. An unfit
calibrator or one-class calibration block uses the identity transformation.
Training/validation separation is the caller's responsibility.
"""
mutable struct PlattCalibrator
    coefficients::Union{Nothing, NTuple{2, Float64}}
end

PlattCalibrator() = PlattCalibrator(nothing)

_platt_sigmoid(value) = value >= 0 ? 1 / (1 + exp(-value)) : exp(value) / (1 + exp(value))

function _platt_logits(values)
    clipped = clamp.(_sizing_values(values, "predicted_probabilities"), 1e-4, 1 - 1e-4)
    logarithms = log.(clipped ./ (1 .- clipped))
    return logarithms isa AbstractVector ? logarithms : vec(permutedims(logarithms, Tuple(ndims(logarithms):-1:1)))
end

function _platt_loss_gradient!(gradient, parameters, logits, outcomes)
    slope, intercept = parameters
    loss, slope_score, intercept_score = 0.0, 0.0, 0.0
    for (value, outcome) in zip(logits, outcomes)
        score = slope * value + intercept
        signed_score = (1 - 2outcome) * score
        loss += max(signed_score, 0) + log1p(exp(-abs(signed_score)))
        residual = _platt_sigmoid(score) - outcome
        slope_score += residual * value
        intercept_score += residual
    end
    count = length(logits)
    penalty = 1e-6 / count
    gradient .= (slope_score / count + penalty * slope, intercept_score / count)
    return loss / count + penalty * slope^2 / 2
end

"""Fit the public two-parameter penalized logistic map and return the same calibrator."""
function fit!(calibrator::PlattCalibrator, predicted_probabilities, outcomes)
    predicted_probabilities isa AbstractArray && !isempty(predicted_probabilities) && outcomes isa AbstractVector && length(predicted_probabilities) == length(outcomes) || throw(ArgumentError("calibration probabilities and outcomes must have matching nonempty observations"))
    probabilities = _sizing_values(predicted_probabilities, "predicted_probabilities")
    all(value -> value isa Real && value in (0, 1), outcomes) || throw(ArgumentError("calibration outcomes must be binary"))
    if all(==(first(outcomes)), outcomes)
        calibrator.coefficients = nothing
        return calibrator
    end
    logits = _platt_logits(probabilities)
    objective(parameters) = _platt_loss_gradient!(zeros(2), parameters, logits, outcomes)
    gradient!(gradient, parameters) = (_platt_loss_gradient!(gradient, parameters, logits, outcomes); gradient)
    result = Optim.optimize(objective, gradient!, zeros(2), Optim.LBFGS(), Optim.Options(g_tol = 1e-10, f_reltol = 1e-14, iterations = 1000))
    Optim.converged(result) && isfinite(Optim.minimum(result)) && all(isfinite, Optim.minimizer(result)) || error("Platt calibration failed to converge")
    calibrator.coefficients = Tuple(Optim.minimizer(result))
    return calibrator
end

"""Map supplied probabilities to a row-major flattened prediction vector, or retain their shape under identity."""
function transform(calibrator::PlattCalibrator, predicted_probabilities)
    values = _sizing_values(predicted_probabilities, "predicted_probabilities")
    isnothing(calibrator.coefficients) && return values
    all(isfinite, calibrator.coefficients) || throw(ArgumentError("calibrator coefficients must be finite"))
    inputs = values isa Number ? [values] : values
    slope, intercept = calibrator.coefficients
    return _platt_sigmoid.(slope .* _platt_logits(inputs) .+ intercept)
end
