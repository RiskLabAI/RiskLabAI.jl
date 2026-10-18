using Test
using TOML
using Dates
using RiskLabAI

validation_matrix(values) = permutedims(reduce(hcat, [Float64.(row) for row in values]))
validation_splits(values) = [(Int[item["train"]...], Int[item["test"]...]) for item in values]
validation_paths(values) = Dict(parse(Int, key) => validation_splits(parts) for (key, parts) in values)
validation_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isapprox(left, right; rtol = 1e-12, atol = 1e-13)
validation_equal(left::AbstractArray, right::AbstractArray) = size(left) == size(right) && all(validation_equal(a, b) for (a, b) in zip(left, right))

function validation_validator(case)
    V = RiskLabAI.Validation
    groups = case["groups"]
    kind = case["kind"]
    if kind == "kfold"
        return V.KFoldCV(groups; shuffle = case["shuffle"], random_seed = case["seed"])
    elseif kind == "walk"
        return V.WalkForwardCV(groups; gap = case["gap"], max_train_size = case["max_train_size"])
    elseif kind == "purged"
        return V.PurgedKFoldCV(groups, case["starts"], case["ends"]; embargo = case["embargo"])
    elseif kind == "combinatorial"
        return V.CombinatorialPurgedCV(groups, case["test_groups"], case["starts"], case["ends"]; embargo = case["embargo"])
    end
    return V.AdaptiveCombinatorialPurgedCV(groups, case["test_groups"], case["starts"], case["ends"]; embargo = case["embargo"], external_feature = case["feature"], n_subsplits = case["subsplits"], lower_quantile = case["lower"], upper_quantile = case["upper"], subtract_border_adjustments = case["subtract"])
end

function validation_predict_probability(model, x)
    probability = 1.0 ./ (1.0 .+ exp.(-(model.alpha .* x[:, 1] .+ 0.2 * model.score)))
    return hcat(1.0 .- probability, probability)
end

validation_predict(model, x) = Int.(validation_predict_probability(model, x)[:, 2] .> 0.5)

@testset "Shared exhaustive inclusive purge and observation embargo" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "validation_python_320.toml"))
    for case in fixture["purge"]
        actual = filtered_training_indices_with_embargo(case["starts"], case["ends"], Int[case["test"]...]; embargo = case["embargo"], continuous = case["continuous"])
        @test actual == case["train"]
    end
end

@testset "Shared combinatorial occurrence paths and adaptive boundaries" begin
    V = RiskLabAI.Validation
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "validation_python_320.toml"))
    for case in fixture["geometry"]
        validator = validation_validator(case)
        @test V._validation_segments(validator) == case["segments"]
        combinations = collect(V.combinations(1:case["groups"], case["test_groups"]))
        expected_locations = Dict(parse(Int, path) => [(pair[1], pair[2]) for pair in locations] for (path, locations) in case["locations"])
        @test V._path_locations(case["groups"], combinations) == expected_locations
        @test cv_split(validator) == validation_splits(case["splits"])
        @test get_n_splits(validator) == length(case["splits"])
        @test backtest_paths(validator) == validation_paths(case["paths"])
        @test length(backtest_paths(validator)) == binomial(case["groups"]-1, case["test_groups"]-1)
        if case["kind"] == "combinatorial" || case["subsplits"] >= 3
            @test all(reduce(vcat, [test for (_, test) in path]) == collect(1:length(case["starts"])) for path in values(backtest_paths(validator)))
        end
    end
    ordering = V._path_locations(4, collect(V.combinations(1:4, 2)))
    @test ordering[2] == [(1, 2), (2, 4), (3, 4), (4, 5)]
    reversed = V._path_locations(4, reverse(collect(V.combinations(1:4, 2))))
    @test reversed[1] == [(1, 4), (2, 2), (3, 1), (4, 1)]
end

@testset "Shared K-fold, purged and walk-forward index contracts" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "validation_python_320.toml"))
    for case in fixture["basic"]
        validator = validation_validator(case)
        count = length(case["starts"])
        expected = validation_splits(case["splits"])
        if case["kind"] == "kfold"
            permutation = case["shuffle"] ? case["permutation"] : nothing
            @test cv_split(validator, count; permutation) == expected
            @test backtest_paths(validator, count; permutation) == Dict(1 => expected)
        elseif case["kind"] == "walk"
            @test cv_split(validator, count) == expected
            @test backtest_paths(validator, count) == Dict(1 => expected)
        else
            @test cv_split(validator) == expected
            @test backtest_paths(validator) == Dict(1 => expected)
        end
        @test get_n_splits(validator) == case["groups"]
    end
end

@testset "Shared caller-estimator predictions preserve rows, weights and model reuse" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "validation_python_320.toml"))
    for case in fixture["predictions"]
        validator = validation_validator(case)
        x, y, weights = validation_matrix(case["x"]), Int.(case["y"]), Float64.(case["weights"])
        originals = deepcopy((x, y, weights))
        prototype = (alpha = 0.7, visits = 7)
        calls = []
        fit_estimator = function(model, train_x, train_y, train_weights)
            push!(calls, (train = Int.(train_x[:, 3]), weights = copy(train_weights)))
            visits = model.visits + 1
            score = sum(train_weights .* (train_y .+ 0.1 .* train_x[:, 2])) / sum(train_weights) + 0.01 * visits
            return (alpha = model.alpha, visits = visits, score = score)
        end
        clone_estimator = case["kind"] == "adaptive" ? model -> (alpha = model.alpha, visits = 0) : deepcopy
        permutation = case["kind"] == "kfold" && case["shuffle"] ? case["permutation"] : nothing
        arguments = (clone_estimator = clone_estimator, fit_estimator = fit_estimator, predict = validation_predict, predict_proba = validation_predict_probability, sample_weights = case["weight_mode"] == "omitted" ? nothing : weights, permutation = permutation)
        actual = backtest_predictions(validator, prototype, x, y; arguments...)
        @test length(calls) == length(case["calls"])
        for (observed, expected) in zip(calls, case["calls"])
            @test observed.train == expected["train"]
            @test observed.weights == expected["weights"]
        end
        probabilities = backtest_predictions(validator, prototype, x, y; arguments..., predict_probability = true)
        @test Set(keys(actual)) == Set(parse.(Int, collect(keys(case["predictions"]))))
        for key in keys(actual)
            @test validation_equal(actual[key], Float64.(case["predictions"][string(key)]))
            @test validation_equal(probabilities[key], validation_matrix(case["probabilities"][string(key)]))
        end
        @test prototype == (alpha = 0.7, visits = 7)
        @test (x, y, weights) == originals
    end
end

@testset "Validator callback failures, snapshots and multiple datasets" begin
    starts = collect(1:24)
    ends = starts .+ 1
    feature = sin.(starts)
    adaptive = AdaptiveCombinatorialPurgedCV(4, 2, starts, ends; external_feature = feature, n_subsplits = 3)
    original = cv_split(adaptive)
    feature .= 100
    starts[1] = -100
    ends[1] = 100
    @test cv_split(adaptive) == original
    dates = Date(2020, 1, 1) .+ Day.(0:23)
    @test filtered_training_indices_with_embargo(dates, dates .+ Day(2), [5, 6, 7]; embargo = .1) == filtered_training_indices_with_embargo(collect(0:23), collect(2:25), [5, 6, 7]; embargo = .1)
    x = hcat(sin.(1:24), cos.(1:24), collect(1:24))
    y = Int.(iseven.(1:24))
    purged = PurgedKFoldCV(4, collect(1:24), collect(2:25))
    prototype = (alpha = .7, visits = 0)
    fit_estimator = (model, x, y, weights) -> (alpha = model.alpha, score = sum(weights .* y) / sum(weights))
    arguments = (clone_estimator = deepcopy, fit_estimator = fit_estimator, predict = validation_predict, predict_proba = validation_predict_probability)
    validators = Dict("purged" => purged, "adaptive" => adaptive)
    datasets = Dict(key => copy(x) for key in keys(validators))
    labels = Dict(key => copy(y) for key in keys(validators))
    estimators = Dict(key => prototype for key in keys(validators))
    weights = Dict(key => collect(1.0:24.0) for key in keys(validators))
    predictions = backtest_predictions(validators, estimators, datasets, labels; sample_weights = weights, arguments...)
    @test Set(keys(predictions)) == Set(keys(validators))
    @test all(predictions[key] == backtest_predictions(validators[key], prototype, x, y; sample_weights = weights[key], arguments...) for key in keys(validators))
    @test backtest_paths(validators) == Dict(key => backtest_paths(value) for (key, value) in validators)
    @test backtest_paths(validators, datasets) == backtest_paths(validators)
    @test backtest_paths(Dict("fold" => KFoldCV(4)), Dict("fold" => x))["fold"] == backtest_paths(KFoldCV(4), 24)
    @test_throws ArgumentError backtest_predictions(validators, Dict("wrong" => prototype), datasets, labels; arguments...)
    @test_throws ArgumentError backtest_predictions(validators, estimators, datasets, labels; sample_weights = Dict("wrong" => ones(24)), arguments...)
    @test_throws DimensionMismatch backtest_predictions(purged, prototype, x, y[1:23]; arguments...)
    @test_throws DimensionMismatch backtest_predictions(purged, prototype, x, y; sample_weights = ones(23), arguments...)
    @test_throws ArgumentError backtest_predictions(purged, prototype, x, y; sample_weights = fill(-1.0, 24), arguments...)
    @test_throws ArgumentError backtest_predictions(purged, prototype, x, y; predict_probability = true, clone_estimator = deepcopy, fit_estimator = fit_estimator, predict = validation_predict)
    @test_throws DimensionMismatch backtest_predictions(purged, prototype, x, y; clone_estimator = deepcopy, fit_estimator = fit_estimator, predict = (model, data) -> zeros(size(data, 1), 1))
    @test_throws DimensionMismatch backtest_predictions(purged, prototype, x, y; clone_estimator = deepcopy, fit_estimator = fit_estimator, predict = (model, data) -> zeros(size(data, 1)+1))
    failures = Ref(0)
    failing_fit = function(model, x, y, weights)
        failures[] += 1
        throw(ArgumentError("estimator does not support these weights"))
    end
    @test_throws ArgumentError backtest_predictions(purged, prototype, x, y; clone_estimator = deepcopy, fit_estimator = failing_fit, predict = validation_predict)
    @test failures[] == 1
    @test_throws ArgumentError cv_split(KFoldCV(4; shuffle = true), 24; permutation = fill(1, 24))
    @test_throws ArgumentError cv_split(KFoldCV(4), 24; permutation = collect(1:24))
    @test_throws ArgumentError cv_split(KFoldCV(0), 24)
    @test_throws ArgumentError cv_split(WalkForwardCV(4; gap = -1), 24)
    @test_throws ArgumentError cv_split(WalkForwardCV(4; max_train_size = 0), 24)
    @test_throws DimensionMismatch filtered_training_indices_with_embargo([1, 2], [1], [1])
    @test_throws ArgumentError filtered_training_indices_with_embargo([2, 1], [2, 1], [1])
    @test_throws ArgumentError filtered_training_indices_with_embargo([1, 2], [1, 2], [3])
    @test_throws ArgumentError filtered_training_indices_with_embargo([1, 2], [1, 2], [1]; embargo = -0.1)
    @test_throws ArgumentError AdaptiveCombinatorialPurgedCV(4, 2, collect(1:8), collect(1:8); external_feature = ones(8))
    @test_throws ArgumentError AdaptiveCombinatorialPurgedCV(4, 2, collect(1:24), collect(1:24); external_feature = fill(NaN, 24))
    @test_throws ArgumentError AdaptiveCombinatorialPurgedCV(4, 2, collect(1:24), collect(1:24); external_feature = ones(24), lower_quantile = .8, upper_quantile = .2)
end
