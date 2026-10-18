using Test
using TOML
using Random
using RiskLabAI

bagged_matrix(values) = permutedims(reduce(hcat, [Float64.(row) for row in values]))

function bagged_diagnostic_probability(model, data)
    scores = log.(model.prior)' .+ (0.03 .* sum(data; dims = 2) .+ 0.02 * model.feature .+ mod(model.seed, 97) * 1e-5) .* (model.classes .+ 1)'
    exponential = exp.(scores .- maximum(scores; dims = 2))
    return exponential ./ sum(exponential; dims = 2)
end

function bagged_validator(case)
    return BaggedCombinatorialPurgedCV(case["groups"], case["test_groups"], case["starts"], case["ends"]; embargo = case["embargo"], classifier = case["classifier"], n_estimators = case["n_estimators"], max_samples = case["max_samples"], max_features = case["max_features"], bootstrap = case["bootstrap"], bootstrap_features = case["bootstrap_features"], random_seed = case["random_seed"])
end

function bagged_callbacks(classifier, calls)
    fit_estimator = function(model, x, y, sample_weights)
        weights = sample_weights === nothing ? ones(length(y)) : sample_weights
        feature = sum(weights .* vec(sum(x; dims = 2))) / sum(weights)
        classes = classifier ? sort(unique(y)) : Int[]
        push!(calls, (x = copy(x), y = copy(y), weights = copy(weights), seed = model.seed, classes = classes, weight_argument_missing = sample_weights === nothing))
        if classifier
            prior = [1.0 + sum(weights[y .== code]) for code in classes]
            return (feature = feature, prior = prior, classes = classes, seed = model.seed)
        end
        return (feature = feature, mean = sum(weights .* y) / sum(weights), seed = model.seed)
    end
    predict = classifier ? (model, data) -> [model.classes[argmax(row)] for row in eachrow(bagged_diagnostic_probability(model, data))] : (model, data) -> model.mean .+ .003 * model.feature .+ .01 .* vec(sum(data; dims = 2)) .+ mod(model.seed, 97)*1e-5
    return fit_estimator, predict
end

@testset "Shared bagged CPCV draws, fitted inputs and predictions" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "bagged_validation_python_320.toml"))
    for case in fixture["cases"]
        validator = bagged_validator(case)
        x = bagged_matrix(case["x"])
        y = case["classifier"] ? String.(case["y"]) : Float64.(case["y"])
        weights = Float64.(case["weights"])
        originals = deepcopy((x, y, weights, case["sample_indices"], case["feature_indices"], case["estimator_seeds"]))
        calls = []
        fit_estimator, predict = bagged_callbacks(case["classifier"], calls)
        prototype = (seed = 991,)
        arguments = (clone_estimator = (model, seed) -> (seed = seed,), fit_estimator = fit_estimator, predict = predict, predict_proba = case["learner"] == "probability" ? bagged_diagnostic_probability : nothing, estimator_classes = case["classifier"] ? model -> model.classes : nothing, fit_sample_weights = case["consumes_weights"], sample_weights = case["weight_mode"] == "omitted" ? nothing : weights, sample_indices = case["sample_indices"], feature_indices = case["feature_indices"], estimator_seeds = case["estimator_seeds"])
        predictions = backtest_predictions(validator, prototype, x, y; arguments...)
        expected_traces = reduce(vcat, case["traces"])
        @test length(calls) == length(expected_traces) == 18
        for (observed, expected) in zip(calls, expected_traces)
            @test observed.x == bagged_matrix(expected["x"])
            @test observed.y == expected["y"]
            @test observed.weights == expected["weights"]
            @test observed.seed == expected["seed"]
            @test observed.classes == expected["classes"]
            @test observed.weight_argument_missing == !case["consumes_weights"]
        end
        @test get_n_splits(validator) == 6
        @test cv_split(validator) == cv_split(validator.base)
        @test backtest_paths(validator) == backtest_paths(validator.base)
        @test Set(keys(predictions)) == Set(1:3)
        for (path, actual) in predictions
            expected = case["predictions"][string(path)]
            @test case["classifier"] ? actual == expected : isapprox(actual, Float64.(expected); rtol = 1e-12, atol = 1e-13)
        end
        if case["classifier"]
            probabilities = backtest_predictions(validator, prototype, x, y; arguments..., predict_probability = true)
            @test length(calls) == 36
            for (path, actual) in probabilities
                @test isapprox(actual, bagged_matrix(case["probabilities"][string(path)]); rtol = 1e-12, atol = 1e-13)
                @test all(isapprox.(vec(sum(actual; dims = 2)), 1.0; atol = 1e-14))
            end
        end
        @test prototype.seed == 991
        @test (x, y, weights, case["sample_indices"], case["feature_indices"], case["estimator_seeds"]) == originals
    end
end

@testset "Bagged class-column alignment, native draws and failures" begin
    V = RiskLabAI.Validation
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "bagged_validation_python_320.toml"))
    case = first(filter(item -> item["learner"] == "probability" && !item["consumes_weights"], fixture["cases"]))
    validator = bagged_validator(case)
    x, y, weights = bagged_matrix(case["x"]), String.(case["y"]), Float64.(case["weights"])
    fit_estimator, predict = bagged_callbacks(true, [])
    common = (clone_estimator = (model, seed) -> (seed = seed,), fit_estimator = fit_estimator, predict = predict, sample_weights = weights, fit_sample_weights = false)
    replay = (sample_indices = case["sample_indices"], feature_indices = case["feature_indices"], estimator_seeds = case["estimator_seeds"])
    reversed = backtest_predictions(validator, nothing, x, y; common..., replay..., predict_proba = (model, data) -> reverse(bagged_diagnostic_probability(model, data); dims = 2), estimator_classes = model -> reverse(model.classes), predict_probability = true)
    @test all(isapprox(actual, bagged_matrix(case["probabilities"][string(path)]); rtol = 1e-12, atol = 1e-13) for (path, actual) in reversed)
    native_first = backtest_predictions(validator, nothing, x, y; common..., predict_proba = bagged_diagnostic_probability)
    native_second = backtest_predictions(validator, nothing, x, y; common..., predict_proba = bagged_diagnostic_probability)
    @test native_first == native_second
    @test all(length(prediction) == length(y) for prediction in values(native_first))
    @test V._bagging_sizes(validator, [1.5, .5, 1.], 4) == (1, 4)
    @test V._bagging_weighted_draws(MersenneTwister(7), [0., 1., 2., 3.], 3, false) |> sort == [2, 3, 4]
    draws = V._bagging_weighted_draws(MersenneTwister(19), [0., 1., 3., 6.], 20000, true)
    @test all(index != 1 for index in draws)
    @test maximum(abs.([count(==(index), draws)/length(draws) for index in 1:4] .- [0., .1, .3, .6])) < .02
    @test_throws ArgumentError BaggedCombinatorialPurgedCV(4, 2, 1:24, 1:24; n_estimators = 0)
    @test_throws ArgumentError BaggedCombinatorialPurgedCV(4, 2, 1:24, 1:24; max_samples = 0.)
    @test_throws ArgumentError BaggedCombinatorialPurgedCV(4, 2, 1:24, 1:24; max_features = 1.2)
    @test_throws ArgumentError backtest_predictions(BaggedCombinatorialPurgedCV(4, 2, 1:24, 1:24; classifier = false), nothing, x[1:24, :], collect(1.:24.); common..., predict_probability = true)
    @test_throws ArgumentError backtest_predictions(validator, nothing, x, y; clone_estimator = common.clone_estimator, fit_estimator = fit_estimator, predict = predict, sample_weights = zeros(length(y)))
    @test_throws DimensionMismatch backtest_predictions(validator, nothing, x, y; common..., sample_indices = case["sample_indices"][1:5])
    invalid_features = deepcopy(case["feature_indices"])
    invalid_features[1][1][1] = 10
    @test_throws ArgumentError backtest_predictions(validator, nothing, x, y; common..., sample_indices = case["sample_indices"], feature_indices = invalid_features, estimator_seeds = case["estimator_seeds"])
    invalid_samples = deepcopy(case["sample_indices"])
    invalid_samples[1][1][1] = 1000
    @test_throws ArgumentError backtest_predictions(validator, nothing, x, y; common..., sample_indices = invalid_samples, feature_indices = case["feature_indices"], estimator_seeds = case["estimator_seeds"])
    @test_throws DimensionMismatch backtest_predictions(validator, nothing, x, y; common..., replay..., predict_proba = (model, data) -> zeros(size(data, 1), 9))
    @test_throws ArgumentError backtest_predictions(validator, nothing, x, y; common..., replay..., predict_proba = bagged_diagnostic_probability, estimator_classes = model -> [-1])
    failures = Ref(0)
    failing_fit = function(model, x, y, weights)
        failures[] += 1
        throw(ArgumentError("unsupported estimator fit"))
    end
    @test_throws ArgumentError backtest_predictions(validator, nothing, x, y; clone_estimator = common.clone_estimator, fit_estimator = failing_fit, predict = predict, replay...)
    @test failures[] == 1
end
