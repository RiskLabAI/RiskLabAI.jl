struct _ADWINSummary
    count::Int
    total::Float64
    centered_squares::Float64
end

mutable struct _ADWINWindow
    delta::Float64
    summaries::Vector{_ADWINSummary}
    width::Int
    total::Float64
    centered_squares::Float64
    clock::Int
    drift_detected::Bool
end

function _ADWINWindow(delta)
    probability = _adaptation_scalar(delta, "delta")
    0 < probability < 1 || throw(ArgumentError("delta must lie strictly between zero and one"))
    return _ADWINWindow(probability, _ADWINSummary[], 0, 0.0, 0.0, 0, false)
end

function _adwin_compress!(window)
    size = 1
    while true
        positions = findall(summary -> summary.count == size, window.summaries)
        length(positions) <= 5 && return window
        first, second = positions[1:2]
        older, newer = window.summaries[first], window.summaries[second]
        mean_difference = older.total / size - newer.total / size
        combined_squares = older.centered_squares + newer.centered_squares + size * 0.5 * mean_difference * mean_difference
        window.summaries[first] = _ADWINSummary(2size, older.total + newer.total, combined_squares)
        deleteat!(window.summaries, second)
        size *= 2
    end
end

function _adwin_has_cut(window)
    older_count, newer_count = 0, window.width
    older_total, newer_total = 0.0, window.total
    for index in 1:(length(window.summaries)-1)
        summary = window.summaries[index]
        older_count += summary.count
        newer_count -= summary.count
        older_total += summary.total
        newer_total -= summary.total
        min(older_count, newer_count) >= 5 || continue
        mean_difference = older_total / older_count - newer_total / newer_count
        confidence_term = log(2.0 * log(Float64(window.width)) / window.delta)
        inverse_count = 1.0 / (older_count - 4) + 1.0 / (newer_count - 4)
        variance_term = 2.0 * inverse_count * (window.centered_squares / window.width) * confidence_term
        threshold = sqrt(variance_term >= 0 ? variance_term : NaN) + 2.0 / 3.0 * confidence_term * inverse_count
        abs(mean_difference) > threshold && return true
    end
    return false
end

function _adwin_remove_oldest!(window)
    oldest = popfirst!(window.summaries)
    window.width -= oldest.count
    window.total -= oldest.total
    difference = oldest.total / oldest.count - window.total / window.width
    removed_squares = oldest.centered_squares + Float64(oldest.count) * window.width * difference * difference / (oldest.count + window.width)
    window.centered_squares -= removed_squares
    return window
end

function _adwin_update!(window::_ADWINWindow, value)
    loss = _adaptation_scalar(value, "loss")
    0 <= loss <= 1 || throw(ArgumentError("ADWIN losses must be in [0,1]"))
    if window.drift_detected
        empty!(window.summaries)
        window.width, window.total, window.centered_squares, window.clock = 0, 0.0, 0.0, 0
    end
    window.drift_detected = false
    push!(window.summaries, _ADWINSummary(1, loss, 0.0))
    window.width += 1
    if window.width > 1
        difference = loss - window.total / (window.width - 1)
        window.centered_squares += (window.width - 1) * difference * difference / window.width
    end
    window.total += loss
    _adwin_compress!(window)
    window.clock += 1
    if window.clock % 32 == 0 && window.width > 10
        while _adwin_has_cut(window)
            window.drift_detected = true
            _adwin_remove_oldest!(window)
        end
    end
    return window.drift_detected
end

function _adaptation_scalar(value, name)
    value isa Real && !(value isa Bool) && isfinite(value) || throw(ArgumentError("$name must be a finite real scalar"))
    result = Float64(value)
    isfinite(result) || throw(ArgumentError("$name is outside the Float64 range"))
    return result
end

function _adaptation_array(value, name, dimension)
    value isa AbstractArray && ndims(value) == dimension && !isempty(value) || throw(ArgumentError("$name must be a nonempty real array of dimension $dimension"))
    return map(item -> _adaptation_scalar(item, name), value)
end

"""
    prequential_adaptation(features, targets, estimator; fit_estimator,
        predict_estimator, clone_estimator, loss_function, window_size,
        min_samples, delta)

Predict each observation before revealing its target and then update the
public ADWIN2 bounded-loss detector. Warm-up predictions/losses are `NaN`.
The first fit uses the first `min_samples` observations. Each alarm triggers a
fresh fit on at most `window_size` latest observed rows, including the alarm
row; this fit can affect only later predictions. Returned alarm/refit indices
are one-based, including the initial refit index `min_samples`.

Native estimator callbacks preserve the caller's model family:
`clone_estimator(estimator)` returns an independent unfit estimator with the
same parameters, `fit_estimator(clone, x, y)` returns the fitted model, and
`predict_estimator(model, x_row)` returns a one-element vector. A fresh clone
is made for every fit. The caller controls estimator random state. Training
and prediction callbacks receive independent data slices.

The detector follows the public fixed defaults: checks every 32 losses,
five buckets per size, minimum subwindow length five, grace period ten and
reset on the observation after an alarm. `loss_function(target,prediction)`
must return a finite scalar in `[0,1]`. The detector's window and the caller's
rolling training window have distinct roles.
"""
function prequential_adaptation(features, targets, estimator;
    fit_estimator, predict_estimator, clone_estimator, loss_function,
    window_size, min_samples, delta)
    x = _adaptation_array(features, "features", 2)
    y = _adaptation_array(targets, "targets", 1)
    for (value, name) in ((window_size, "window_size"), (min_samples, "min_samples"))
        value isa Integer && !(value isa Bool) && value > 0 || throw(ArgumentError("$name must be a positive integer"))
    end
    size(x, 1) == length(y) && min_samples <= window_size && min_samples < length(y) || throw(ArgumentError("Require matching rows, min_samples <= window_size, and enough data"))
    applicable(loss_function, first(y), 0.0) || throw(ArgumentError("loss_function must accept a target and prediction"))
    applicable(clone_estimator, estimator) || throw(ArgumentError("clone_estimator must accept the supplied estimator"))
    detector = _ADWINWindow(delta)
    predictions, losses = fill(NaN, length(y)), fill(NaN, length(y))
    alarms, refits = Int[], [Int(min_samples)]
    function fit_observed(indices)
        fresh = clone_estimator(estimator)
        training_x, training_y = x[indices, :], y[indices]
        applicable(fit_estimator, fresh, training_x, training_y) || throw(ArgumentError("fit_estimator must accept a clone and training arrays"))
        return fit_estimator(fresh, training_x, training_y)
    end
    model = fit_observed(1:Int(min_samples))
    for index in (Int(min_samples)+1):length(y)
        row = x[index:index, :]
        applicable(predict_estimator, model, row) || throw(ArgumentError("predict_estimator must accept the fitted model and a row matrix"))
        raw_prediction = predict_estimator(model, row)
        raw_prediction isa AbstractVector && length(raw_prediction) == 1 || throw(ArgumentError("Estimator must return one scalar prediction per row"))
        prediction = _adaptation_scalar(only(raw_prediction), "prediction")
        applicable(loss_function, y[index], prediction) || throw(ArgumentError("loss_function must accept a target and prediction"))
        loss = _adaptation_scalar(loss_function(y[index], prediction), "loss")
        0 <= loss <= 1 || throw(ArgumentError("ADWIN losses must be in [0,1]"))
        predictions[index], losses[index] = prediction, loss
        if _adwin_update!(detector, loss)
            push!(alarms, index)
            beginning = Int(max(1, index + 1 - window_size))
            model = fit_observed(beginning:index)
            push!(refits, index)
        end
    end
    return (predictions = predictions, losses = losses, alarm_indices = alarms, refit_indices = refits)
end
