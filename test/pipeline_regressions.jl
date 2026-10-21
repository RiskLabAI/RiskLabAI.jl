using Test
using TOML
using Statistics
using Random
using Distributions: Uniform
using RiskLabAI

pipeline_matrix(values) = permutedims(reduce(hcat, [Float64.(row) for row in values]))
pipeline_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isapprox(left, right; rtol = 1e-11, atol = 1e-13)
pipeline_equal(left::AbstractArray, right::AbstractArray) = size(left) == size(right) && all(pipeline_equal(a, b) for (a, b) in zip(left, right))

function pipeline_diagnostic_probability(model, data)
    get(model, "score_fail", false) && throw(ArgumentError("requested diagnostic scoring failure"))
    signal = get(model, "strength", 1.) .* (.45 .* data[:, 1] .+ .15 .* vec(sum(data; dims = 2))) .+ get(model, "bias", 0.) .+ .01*model["feature"] .+ (get(model, "random_state", 0)%97)*1e-5
    scores = log.(model["prior"])' .+ signal .* collect(0:length(model["classes"])-1)'
    exponential = exp.(scores .- maximum(scores; dims = 2))
    return exponential ./ sum(exponential; dims = 2)
end

function pipeline_diagnostic(traces = Any[]; fit_transform = false, probability = true)
    clone_step = function(name, model, seed = nothing)
        options = Dict{String,Any}(key => deepcopy(value) for (key, value) in model if key in ("strength", "bias", "fail", "score_fail", "with_std", "random_state"))
        name == "model" && seed !== nothing && (options["random_state"] = seed)
        return options
    end
    transform = (name, model, data) -> (data .- model["mean"]') ./ model["scale"]'
    fit_step = function(name, model, data, labels, params)
        weights = Float64.(get(params, "sample_weight", ones(length(labels))))
        if name == "scale"
            center = vec(sum(data .* weights; dims = 1)) ./ sum(weights)
            deviation = get(model, "with_std", true) ? sqrt.(vec(sum((data .- center').^2 .* weights; dims = 1)) ./ sum(weights)) : ones(size(data, 2))
            deviation[deviation .== 0] .= 1
            merge!(model, Dict("mean" => center, "scale" => deviation))
            push!(traces, Dict("step" => name, "x" => copy(data), "weights" => weights, "weighted" => haskey(params, "sample_weight"), "mean" => center, "scale" => deviation))
        else
            get(model, "fail", false) && throw(ArgumentError("requested diagnostic fit failure"))
            !isempty(labels) && sum(weights) > 0 || throw(ArgumentError("empty diagnostic fit"))
            classes = sort(unique(labels))
            prior = [1+sum(weights[labels .== code]) for code in classes]
            feature = sum(weights .* vec(sum(data; dims = 2))) / sum(weights)
            visits = get(model, "visits", 0)+1
            merge!(model, Dict("classes" => classes, "prior" => prior, "feature" => feature, "visits" => visits))
            push!(traces, Dict("step" => name, "x" => copy(data), "y" => copy(labels), "weights" => weights, "weighted" => haskey(params, "sample_weight"), "visits" => visits, "row_tag" => get(params, "row_tag", Int[]), "token" => get(params, "token", "fixed"), "seed" => get(model, "random_state", 0)))
        end
        return model
    end
    combined = fit_transform ? (name, model, data, labels, options) -> begin
        fitted = fit_step(name, model, data, labels, options)
        return fitted, transform(name, fitted, data)
    end : nothing
    prediction = (name, model, data) -> [model["classes"][argmax(row)] for row in eachrow(pipeline_diagnostic_probability(model, data))]
    return SampleWeightedPipeline(["scale" => Dict{String,Any}(), "model" => Dict{String,Any}()]; clone_step, fit_step, transform_step = transform, predict_step = prediction, predict_proba_step = probability ? (_, model, data) -> pipeline_diagnostic_probability(model, data) : nothing, classes_step = (_, model) -> model["classes"], configure_step = (_, model, params) -> merge(model, params), fit_transform_step = combined)
end

function pipeline_check_traces(actual, expected)
    @test length(actual) == length(expected)
    for (left, right) in zip(actual, expected)
        @test Set(keys(left)) == Set(keys(right))
        for key in keys(right)
            if key == "x"
                @test pipeline_equal(left[key], pipeline_matrix(right[key]))
            elseif right[key] isa AbstractVector
                @test pipeline_equal(left[key], right[key])
            else
                @test left[key] == right[key]
            end
        end
    end
end

@testset "Shared pipeline routing and estimator state" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "pipeline_python_320.toml"))
    for case in fixture["pipeline"], combined in (false, true)
        traces = Any[]
        model = pipeline_diagnostic(traces; fit_transform = combined)
        x, y = pipeline_matrix(case["x"]), Int.(case["y"])
        original = deepcopy((x, y, case["fit_params"]))
        @test fit_pipeline!(model, x, y; fit_params = case["fit_params"]) === model
        @test fit_pipeline!(model, x, y; fit_params = case["fit_params"]) === model
        @test pipeline_equal(predict_pipeline_proba(model, x), pipeline_matrix(case["probabilities"]))
        @test predict_pipeline(model, x) == case["predictions"]
        @test pipeline_classes(model) == sort(unique(y))
        pipeline_check_traces(traces, case["traces"])
        cloned = clone_pipeline(model)
        @test !cloned.fitted
        @test !haskey(last(cloned.steps[end]), "visits")
        @test last(model.steps[end])["visits"] == 2
        @test (x, y, case["fit_params"]) == original
    end
end

@testset "Shared weighted F1 and log-loss pipeline search" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "pipeline_python_320.toml"))
    for case in fixture["search"]
        traces = Any[]
        prototype = pipeline_diagnostic(traces)
        x, y = pipeline_matrix(case["x"]), Int.(case["y"])
        cv = PurgedKFoldCV(case["n_splits"], Int.(case["starts"]), Int.(case["ends"]); embargo = case["embargo"])
        original = deepcopy((x, y, case["fit_params"], case["grid"]))
        result = clf_hyper_fit(cv, x, y, prototype, case["grid"]; fit_params = case["fit_params"], rnd_search_iter = case["iterations"], candidate_indices = case["iterations"] == 0 ? nothing : Int.(case["candidate_indices"]), return_search = true)
        @test result.candidates == case["candidates"]
        @test result.best_index == case["best_index"]
        @test result.best_params == case["candidates"][case["best_index"]]
        @test result.scoring == (Set(y) == Set([0, 1]) ? :f1 : :neg_log_loss)
        @test isempty(result.failures)
        @test pipeline_equal(result.fold_scores, pipeline_matrix(case["scores"]))
        @test pipeline_equal(result.mean_scores, Float64.(case["mean_scores"]))
        @test pipeline_equal(predict_pipeline_proba(result.estimator, x), pipeline_matrix(case["probabilities"]))
        @test predict_pipeline(result.estimator, x) == case["predictions"]
        pipeline_check_traces(traces, case["traces"])
        @test all(trace["visits"] == 1 for trace in traces if trace["step"] == "model")
        @test size(traces[end]["x"], 1) == length(y)
        @test !prototype.fitted
        @test all(isempty(last(step)) for step in prototype.steps)
        @test (x, y, case["fit_params"], case["grid"]) == original
    end
end

@testset "Shared post-search weighted bagging" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "pipeline_python_320.toml"))
    for case in fixture["bagging"]
        traces = Any[]
        prototype = pipeline_diagnostic(traces)
        x, y = pipeline_matrix(case["x"]), Int.(case["y"])
        cv = PurgedKFoldCV(3, Int.(case["starts"]), Int.(case["ends"]); embargo = 0.)
        plans = case["plans"]
        result = clf_hyper_fit(cv, x, y, prototype, case["grid"]; fit_params = case["fit_params"], rnd_search_iter = case["iterations"], candidate_indices = Int.(case["candidate_indices"]), bagging = case["bagging"], bagging_sample_indices = [Int.(plan["samples"]) for plan in plans], bagging_feature_indices = [Int.(plan["features"]) for plan in plans], bagging_estimator_seeds = [plan["seed"] for plan in plans], return_search = true)
        @test result.best_index == case["best_index"]
        @test result.best_params == case["best_params"]
        @test first.(result.estimator.steps) == ["bag"]
        @test length(last(result.estimator.steps[end]).members) == case["bagging"][1]
        @test pipeline_equal(predict_pipeline_proba(result.estimator, x), pipeline_matrix(case["probabilities"]))
        @test predict_pipeline(result.estimator, x) == case["predictions"]
        @test pipeline_classes(result.estimator) == sort(unique(y))
        pipeline_check_traces(traces, case["traces"])
        @test !prototype.fitted
    end
end

@testset "Search failures, first ties and random candidate contracts" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "pipeline_python_320.toml"))
    source = fixture["search"][1]
    x, y = pipeline_matrix(source["x"]), Int.(source["y"])
    cv = PurgedKFoldCV(3, Int.(source["starts"]), Int.(source["ends"]); embargo = 0.)
    for case in fixture["errors"]
        if case["raises"]
            @test_throws ArgumentError clf_hyper_fit(cv, x, y, pipeline_diagnostic(), case["grid"])
        else
            result = clf_hyper_fit(cv, x, y, pipeline_diagnostic(), case["grid"]; return_search = true)
            @test result.best_index == case["best_index"]
            @test pipeline_equal(result.fold_scores, pipeline_matrix(case["scores"]))
            @test pipeline_equal(result.mean_scores, Float64.(case["mean_scores"]))
            @test all(failure.phase == (case["name"] == "partial_fit" ? :fit : :score) for failure in result.failures)
            @test result.estimator.fitted
        end
    end
    @test_throws ArgumentError clf_hyper_fit(cv, x, y, pipeline_diagnostic(), Dict("model__fail" => [true, false]); error_score = :raise)
    draw_case = only(fixture["distribution"])
    result = clf_hyper_fit(cv, x, y, pipeline_diagnostic(), Dict("model__strength" => Uniform(-2, 2), "model__bias" => [-.4, .3]); rnd_search_iter = 4, candidate_draws = draw_case["candidates"], return_search = true)
    @test result.best_index == draw_case["best_index"]
    @test pipeline_equal(result.fold_scores, pipeline_matrix(draw_case["scores"]))
    result_a = clf_hyper_fit(cv, x, y, pipeline_diagnostic(), source["grid"]; rnd_search_iter = 3, rng = MersenneTwister(46), return_search = true)
    result_b = clf_hyper_fit(cv, x, y, pipeline_diagnostic(), source["grid"]; rnd_search_iter = 3, rng = MersenneTwister(46), return_search = true)
    @test result_a.candidates == result_b.candidates
    @test result_a.fold_scores == result_b.fold_scores
    @test allunique(result_a.candidates)
    result_all = clf_hyper_fit(cv, x, y, pipeline_diagnostic(), source["grid"]; rnd_search_iter = 20, rng = MersenneTwister(46), return_search = true)
    @test length(result_all.candidates) == 8
    @test allunique(result_all.candidates)
    continuous = Dict("model__strength" => Uniform(-2, 2), "model__bias" => [-.4, .3])
    native_a = clf_hyper_fit(cv, x, y, pipeline_diagnostic(), continuous; rnd_search_iter = 7, rng = MersenneTwister(4), return_search = true)
    native_b = clf_hyper_fit(cv, x, y, pipeline_diagnostic(), continuous; rnd_search_iter = 7, rng = MersenneTwister(4), return_search = true)
    @test native_a.candidates == native_b.candidates
    @test length(native_a.candidates) == 7
    @test all(-2 <= row["model__strength"] <= 2 && row["model__bias"] in (-.4, .3) for row in native_a.candidates)
    @test pipeline_parameter_grid([Dict("b" => [1, 2], "a" => [4, 5]), Dict()]) == [Dict("a" => 4, "b" => 1), Dict("a" => 4, "b" => 2), Dict("a" => 5, "b" => 1), Dict("a" => 5, "b" => 2), Dict()]
    @test_throws ArgumentError pipeline_parameter_grid(Dict("a" => []))
    @test_throws ArgumentError pipeline_parameter_grid(Dict("a" => 1))
    @test_throws ArgumentError clf_hyper_fit(cv, x, y, pipeline_diagnostic(), source["grid"]; rnd_search_iter = 3, candidate_indices = [1, 1, 2])
    @test_throws ArgumentError clf_hyper_fit(cv, x, y, pipeline_diagnostic(), source["grid"]; candidate_indices = [1])
    @test_throws DimensionMismatch clf_hyper_fit(cv, x, y, pipeline_diagnostic(), continuous; rnd_search_iter = 3, candidate_draws = [Dict()])
    @test_throws ArgumentError clf_hyper_fit(cv, x, y, pipeline_diagnostic(), continuous; rnd_search_iter = 3, candidate_indices = [1, 2, 3])
    @test_throws ArgumentError clf_hyper_fit(cv, x, y, pipeline_diagnostic(), source["grid"]; sample_weight = -ones(length(y)))
    @test_throws ArgumentError clf_hyper_fit(cv, x, y, pipeline_diagnostic(), Dict(); folds = [([1, 2], [2, 3])])
    @test_throws ArgumentError clf_hyper_fit(cv, x, y, pipeline_diagnostic(), Dict(); fit_params = Dict("model__sample_weight" => ones(length(y))), bagging = [2, .8, 1.])
    @test clf_hyper_fit(x, y, Int.(source["starts"]), Int.(source["ends"]), pipeline_diagnostic(), Dict()) isa SampleWeightedPipeline
    @test clf_hyper_fit(cv, x, y, pipeline_diagnostic(; probability = false), Dict(); bagging = [2, .8, 1.], rng = MersenneTwister(6)) isa SampleWeightedPipeline
end

@testset "Pipeline boundary and passthrough contracts" begin
    prototype = pipeline_diagnostic()
    @test_throws ArgumentError predict_pipeline(prototype, ones(3, 2))
    @test_throws ArgumentError pipeline_classes(prototype)
    @test_throws ArgumentError configure_pipeline!(prototype, Dict("unknown__option" => 1))
    @test_throws ArgumentError fit_pipeline!(prototype, ones(3, 2), [0, 1, 0]; fit_params = Dict("unknown__sample_weight" => ones(3)))
    @test_throws ArgumentError fit_pipeline!(prototype, ones(3, 2), [0, 1, 0]; fit_params = Dict("row_tag" => ones(3)))
    @test_throws DimensionMismatch fit_pipeline!(prototype, ones(3, 2), [0, 1])
    @test_throws DimensionMismatch fit_pipeline!(prototype, ones(3, 2), [0, 1, 0]; sample_weight = [1.])
    @test_throws ArgumentError SampleWeightedPipeline(["same" => nothing, "same" => nothing]; prototype.callbacks...)
    @test_throws ArgumentError SampleWeightedPipeline(["bad__name" => nothing]; prototype.callbacks...)
    passthrough = SampleWeightedPipeline(["scale" => :passthrough, "model" => Dict{String,Any}()]; prototype.callbacks...)
    fit_pipeline!(passthrough, [1. 2.; 2. 3.; 4. 1.], [0, 1, 0])
    @test predict_pipeline(passthrough, [1. 2.; 2. 3.]) == [1, 1]
    @test predict_pipeline_proba(passthrough, [1. 2.])[1, 2] ≈ 2exp(.9+13/300) / (3+2exp(.9+13/300))
    @test sum(predict_pipeline_proba(passthrough, [1. 2.])) ≈ 1.
    @test first(clone_pipeline(passthrough).steps) == ("scale" => :passthrough)
    unicode = SampleWeightedPipeline(["modèle" => Dict{String,Any}()]; prototype.callbacks...)
    configure_pipeline!(unicode, Dict("modèle__bias" => 1.))
    @test last(unicode.steps[end])["bias"] == 1.
end
