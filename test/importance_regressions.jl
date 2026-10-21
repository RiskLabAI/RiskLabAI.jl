using Test
using TOML
using LinearAlgebra
using RiskLabAI

importance_matrix(values) = permutedims(reduce(hcat, [Float64.(row) for row in values]))
importance_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isapprox(left, right; rtol = 1e-11, atol = 1e-13)
importance_equal(left::AbstractArray, right::AbstractArray) = size(left) == size(right) && all(importance_equal(a, b) for (a, b) in zip(left, right))

function importance_probability(model, data)
    signal = .05 .* (data * collect(1:size(data, 2))) .+ .01*model.feature .+ .002*model.visits
    scores = log.(model.prior)' .+ signal .* (model.classes .+ 1)'
    exponential = exp.(scores .- maximum(scores; dims = 2))
    return exponential ./ sum(exponential; dims = 2)
end

@testset "Shared PCA column filtering and invariant eigenspaces" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "importance_python_320.toml"))
    for case in fixture["pca"]
        x = importance_matrix(case["x"])
        original = copy(x)
        transformed, values, vectors, cumulative = orthogonal_features(x; variance_threshold = case["threshold"])
        normalized = importance_matrix(case["normalized"])
        if case["threshold"] == 1.0 && size(normalized, 2) > size(normalized, 1)-1
            retained_rank = rank(normalized)
            @test retained_rank <= length(values) <= size(normalized, 2)
            @test isapprox(values[1:retained_rank], Float64.(case["eigenvalues"][1:retained_rank]); rtol = 1e-10, atol = 1e-10)
            @test isapprox(cumulative[1:retained_rank], Float64.(case["cumulative"][1:retained_rank]); rtol = 1e-10, atol = 1e-10)
            @test all(abs.(values[(retained_rank+1):end]) .< 1e-10)
            @test all(abs.(case["eigenvalues"][(retained_rank+1):end]) .< 1e-10)
        else
            @test length(values) == length(case["eigenvalues"])
            @test isapprox(values, Float64.(case["eigenvalues"]); rtol = 1e-10, atol = 1e-10)
            @test isapprox(cumulative, Float64.(case["cumulative"]); rtol = 1e-10, atol = 1e-10)
        end
        @test size(vectors, 1) == length(case["retained_columns"])
        @test isapprox(vectors' * vectors, I; atol = 1e-10)
        @test isapprox(normalized' * normalized * vectors, vectors * Diagonal(values); rtol = 1e-10, atol = 1e-10)
        @test isapprox(transformed, normalized * vectors; rtol = 1e-10, atol = 1e-10)
        @test isapprox(transformed * transformed', importance_matrix(case["output_gram"]); rtol = 1e-10, atol = 1e-10)
        if case["separated"]
            @test isapprox(vectors * vectors', importance_matrix(case["projector"]); rtol = 1e-10, atol = 1e-10)
        end
        @test isequal(x, original)
    end
    @test_throws ArgumentError orthogonal_features(ones(12, 3))
    @test_throws ArgumentError orthogonal_features(ones(1, 3))
end

@testset "Shared weighted ranks with ties and missing scores" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "importance_python_320.toml"))
    for case in fixture["tau"]
        actual = calculate_weighted_tau(Float64[case["values"]...], Float64[case["ranks"]...])
        @test importance_equal(actual, case["result"])
    end
    @test_throws DimensionMismatch calculate_weighted_tau([1., 2.], [1.])
end

@testset "Shared per-tree and clustered impurity aggregation" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "importance_python_320.toml"))
    for case in fixture["mdi"]
        per_tree = importance_matrix(case["per_tree"])
        original = copy(per_tree)
        actual = feature_importance_mdi(per_tree)
        @test importance_equal(actual.mean, Float64.(case["mean"]))
        @test importance_equal(actual.std, Float64.(case["std"]))
        clusters = [name => Int[group...] for (name, group) in zip(case["cluster_names"], case["clusters"])]
        grouped = clustered_feature_importance_mdi(per_tree, clusters)
        @test grouped.names == "C_" .* case["cluster_names"]
        @test importance_equal(grouped.mean, Float64.(case["clustered_mean"]))
        @test importance_equal(grouped.std, Float64.(case["clustered_std"]))
        calls = []
        fit_estimator = function(model, x, y, weights)
            push!(calls, (x = copy(x), y = copy(y), weights = copy(weights)))
            return (per_tree = model.per_tree,)
        end
        x, y, weights = importance_matrix(case["x"]), Int.(case["y"]), Float64.(case["weights"])
        arguments = (fit_estimator = fit_estimator, tree_importances = model -> model.per_tree, sample_weights = weights)
        fitted = feature_importance_mdi((per_tree = per_tree,), x, y; arguments...)
        fitted_grouped = clustered_feature_importance_mdi((per_tree = per_tree,), x, y, clusters; arguments...)
        @test importance_equal(fitted.mean, actual.mean)
        @test importance_equal(fitted.std, actual.std)
        @test importance_equal(fitted_grouped.mean, grouped.mean)
        @test importance_equal(fitted_grouped.std, grouped.std)
        @test length(calls) == 2
        @test all(call.x == x && call.y == y && call.weights == weights for call in calls)
        @test fitted.estimator.per_tree == per_tree
        @test per_tree == original
    end
end

@testset "Shared weighted permutation and single-feature importance" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "importance_python_320.toml"))
    for case in fixture["importance"]
        x, y = importance_matrix(case["x"]), Int.(case["y"])
        originals = deepcopy((x, y, case["train_weights"], case["score_weights"]))
        calls = []
        fit_estimator = function(model, data, labels, weights)
            visits = model.visits + 1
            classes = sort(unique(labels))
            prior = [1 + sum(weights[labels .== label]) for label in classes]
            feature = sum(weights .* (data * collect(1:size(data, 2)))) / sum(weights)
            push!(calls, (x = copy(data), y = copy(labels), weights = copy(weights), visits = visits))
            return (classes = classes, prior = prior, feature = feature, visits = visits)
        end
        predict = (model, data) -> [model.classes[argmax(row)] for row in eachrow(importance_probability(model, data))]
        folds = [(Int.(fold["train"]), Int.(fold["test"])) for fold in case["folds"]]
        weights = (train_sample_weights = case["weight_mode"] == "omitted" ? nothing : Float64.(case["train_weights"]), score_sample_weights = case["weight_mode"] == "omitted" ? nothing : Float64.(case["score_weights"]))
        arguments = (fit_estimator = fit_estimator, predict_proba = importance_probability, estimator_classes = model -> model.classes, folds = folds)
        prototype = (visits = 0,)
        if case["method"] == "mda"
            actual = feature_importance_mda(prototype, x, y; arguments..., weights..., permutations = case["permutations"])
        elseif case["method"] == "clustered_mda"
            clusters = [name => Int[group...] for (name, group) in zip(case["cluster_names"], case["clusters"])]
            actual = clustered_feature_importance_mda(prototype, x, y, clusters; arguments..., weights..., permutations = case["permutations"])
            @test actual.names == "C_" .* case["cluster_names"]
        else
            scoring = case["method"] == "sfi_accuracy" ? :accuracy : :log_loss
            actual = feature_importance_sfi(prototype, x, y; arguments..., weights..., scoring, predict)
        end
        @test importance_equal(actual.mean, Float64.(case["mean"]))
        @test importance_equal(actual.std, Float64.(case["std"]))
        @test length(calls) == length(case["traces"])
        for (observed, expected) in zip(calls, case["traces"])
            @test observed.x == importance_matrix(expected["x"])
            @test observed.y == expected["y"]
            @test observed.weights == expected["weights"]
            @test observed.visits == expected["visits"]
        end
        @test actual.estimator.visits == length(calls)
        @test (x, y, case["train_weights"], case["score_weights"]) == originals
    end
end

@testset "Importance clipping, weighting and input contracts" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "importance_python_320.toml"))
    for case in fixture["loss"]
        actual = F._importance_log_loss(case["labels"], importance_matrix(case["probabilities"]), case["weights"], case["classes"])
        @test importance_equal(actual, case["output"])
    end
    @test_throws ArgumentError F._importance_log_loss([0, 1], [.2 .8; .1 .9], [0., 0.], [0, 1])
    @test_throws ArgumentError F._importance_log_loss([0, 1], [.2 .8; -.1 1.1], [1., 1.], [0, 1])
    @test_throws ArgumentError F._importance_log_loss([0, 2], [.2 .8; .1 .9], [1., 1.], [0, 1])
    @test_throws DimensionMismatch F._importance_log_loss([0, 1], ones(2, 3)./3, [1., 1.], [0, 1])
    @test_throws ArgumentError feature_importance_mdi(zeros(0, 3))
    @test_throws ArgumentError clustered_feature_importance_mdi(ones(2, 3), [[4]])
    @test_throws ArgumentError F._importance_folds(12, 1, nothing, true, 0)
    @test_throws ArgumentError F._importance_folds(4, 2, [([1, 2], [2, 3]), ([3, 4], [1, 4])], false, 0)
    @test_throws ArgumentError F._importance_permutations([[[1, 1]], [[1, 2]]], [([3, 4], [1, 2]), ([1, 2], [3, 4])], [[1]])
    @test_throws ArgumentError F._importance_weights([-1., 1.], 2)
    @test_throws DimensionMismatch F._importance_weights([1.], 2)
    x, y = reshape(collect(1.:24.), 12, 2), repeat([0, 1], 6)
    failures = Ref(0)
    fit_estimator = function(model, x, y, weights)
        failures[] += 1
        throw(ArgumentError("caller fit failed"))
    end
    @test_throws ArgumentError feature_importance_mda(nothing, x, y; fit_estimator, predict_proba = identity, estimator_classes = identity, n_splits = 3)
    @test failures[] == 1
    @test_throws ArgumentError feature_importance_sfi(nothing, x, y; fit_estimator, scoring = :f1)
    @test_throws ArgumentError feature_importance_sfi(nothing, x, y; fit_estimator, scoring = :accuracy)
    @test_throws ArgumentError feature_importance_sfi(nothing, x, y; fit_estimator)
end
