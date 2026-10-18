"""
    SampleWeightedPipeline(steps; clone_step, fit_step, transform_step,
        predict_step, predict_proba_step=nothing, classes_step,
        configure_step, fit_transform_step=nothing)

Compose named native estimators without selecting their model family. Callbacks
receive `(name, step, ...)`. `fit_step` receives training features, labels and a
string-keyed options dictionary and returns the fitted step. Transformer steps
use `transform_step`; an optional `fit_transform_step` can instead return the
fitted step and training transformation together. Prediction callbacks receive
the fitted final step and transformed features; `classes_step` returns its
ordered class labels. `clone_step` must return a fresh unfitted estimator and
may accept a third seed argument. `configure_step` applies a parameter dictionary
to one step. Intermediate `nothing` or `:passthrough` steps are supported.

Top-level fitting weights route only to the final estimator. Explicit
`name__option` fitting arguments route to the named step. Global weights override
an explicit final-step weight, preserving the public SampleWeightedPipeline
contract. No estimator is silently replaced or retried without weights.
"""
mutable struct SampleWeightedPipeline
    steps::Vector{Pair{String,Any}}
    callbacks::NamedTuple
    fitted::Bool
end

function SampleWeightedPipeline(steps; clone_step, fit_step, transform_step, predict_step, predict_proba_step = nothing, classes_step, configure_step = (name, step, options) -> isempty(options) ? step : throw(ArgumentError("configure_step is required for nonempty parameters")), fit_transform_step = nothing)
    named = Pair{String,Any}[string(first(step)) => last(step) for step in steps]
    !isempty(named) && allunique(first.(named)) && all(!isempty(first(step)) && !occursin("__", first(step)) for step in named) || throw(ArgumentError("pipeline step names must be nonempty, unique and exclude __"))
    callbacks = (; clone_step, fit_step, transform_step, predict_step, predict_proba_step, classes_step, configure_step, fit_transform_step)
    return SampleWeightedPipeline(named, callbacks, false)
end

_pipeline_passthrough(step) = step === nothing || step === :passthrough
_pipeline_options(options) = Dict{String,Any}(string(key) => deepcopy(value) for (key, value) in pairs(options))

"""Clone every pipeline step through its caller-supplied unfitted-estimator contract."""
function clone_pipeline(pipeline::SampleWeightedPipeline; seed = nothing)
    callback = pipeline.callbacks.clone_step
    steps = map(pipeline.steps) do (name, step)
        cloned = _pipeline_passthrough(step) ? step : seed !== nothing && applicable(callback, name, step, seed) ? callback(name, step, seed) : callback(name, step)
        return name => cloned
    end
    return SampleWeightedPipeline(steps; pipeline.callbacks...)
end

"""Apply `step__parameter` options to native steps, or replace a named step."""
function configure_pipeline!(pipeline::SampleWeightedPipeline, parameters)
    options = _pipeline_options(parameters)
    names = first.(pipeline.steps)
    for key in keys(options)
        first(split(key, "__"; limit = 2)) in names || throw(ArgumentError("unknown pipeline step in parameter $key"))
    end
    for (position, (name, step)) in enumerate(pipeline.steps)
        replacement = get(options, name, step)
        nested = Dict{String,Any}(chopprefix(key, name * "__") => value for (key, value) in options if startswith(key, name * "__"))
        replacement = pipeline.callbacks.configure_step(name, replacement, nested)
        pipeline.steps[position] = name => replacement
    end
    pipeline.fitted = false
    return pipeline
end

"""Fit each transformation on training rows, routing global weights only to the final step."""
function fit_pipeline!(pipeline::SampleWeightedPipeline, x::AbstractMatrix, y::AbstractVector; sample_weight = nothing, fit_params = Dict{String,Any}())
    size(x, 1) == length(y) || throw(DimensionMismatch("pipeline features and labels must align"))
    options = _pipeline_options(fit_params)
    haskey(options, "sample_weight") && (sample_weight === nothing && (sample_weight = options["sample_weight"]); delete!(options, "sample_weight"))
    if sample_weight !== nothing
        length(sample_weight) == length(y) || throw(DimensionMismatch("pipeline weights must align with labels"))
        options[first(pipeline.steps[end]) * "__sample_weight"] = copy(sample_weight)
    end
    routed = Dict(name => Dict{String,Any}() for (name, _) in pipeline.steps)
    for (key, value) in options
        parts = split(key, "__"; limit = 2)
        length(parts) == 2 && haskey(routed, parts[1]) && !isempty(parts[2]) || throw(ArgumentError("fit options must use a known step__option name"))
        routed[parts[1]][parts[2]] = value
    end
    pipeline.fitted = false
    data, labels = copy(x), copy(y)
    for position in eachindex(pipeline.steps)
        name, step = pipeline.steps[position]
        final = position == length(pipeline.steps)
        if !final && _pipeline_passthrough(step)
            continue
        elseif !final && pipeline.callbacks.fit_transform_step !== nothing
            fitted, data = pipeline.callbacks.fit_transform_step(name, step, data, labels, routed[name])
        else
            fitted = pipeline.callbacks.fit_step(name, step, data, labels, routed[name])
            final || (data = pipeline.callbacks.transform_step(name, fitted, data))
        end
        pipeline.steps[position] = name => fitted
        size(data, 1) == length(y) || throw(DimensionMismatch("pipeline transformations must preserve observation count"))
    end
    pipeline.fitted = true
    return pipeline
end

function _pipeline_transform(pipeline, x)
    pipeline.fitted || throw(ArgumentError("pipeline must be fitted before prediction"))
    data = copy(x)
    for (name, step) in pipeline.steps[1:end-1]
        _pipeline_passthrough(step) || (data = pipeline.callbacks.transform_step(name, step, data))
        size(data, 1) == size(x, 1) || throw(DimensionMismatch("pipeline transformations must preserve observation count"))
    end
    return data
end

"""Predict labels with the fitted pipeline, preserving the caller's estimator."""
function predict_pipeline(pipeline::SampleWeightedPipeline, x::AbstractMatrix)
    data = _pipeline_transform(pipeline, x)
    name, fitted = pipeline.steps[end]
    return pipeline.callbacks.predict_step(name, fitted, data)
end

"""Predict class probabilities in the order returned by `pipeline_classes`."""
function predict_pipeline_proba(pipeline::SampleWeightedPipeline, x::AbstractMatrix)
    pipeline.callbacks.predict_proba_step !== nothing || throw(ArgumentError("pipeline has no probability callback"))
    data = _pipeline_transform(pipeline, x)
    name, fitted = pipeline.steps[end]
    return pipeline.callbacks.predict_proba_step(name, fitted, data)
end

"""Return the fitted final estimator's ordered class labels."""
function pipeline_classes(pipeline::SampleWeightedPipeline)
    pipeline.fitted || throw(ArgumentError("pipeline must be fitted before reading classes"))
    return collect(pipeline.callbacks.classes_step(pipeline.steps[end]...))
end
