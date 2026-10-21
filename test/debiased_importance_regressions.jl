using Test
using TOML
using Random
using RiskLabAI

debiased_matrix(rows) = reduce(vcat, [permutedims(Vector{Float64}(row)) for row in rows])

@testset "Debiased importance: supplied tree bases and unique in-bag rows" begin
    F = RiskLabAI.Features
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "debiased_importance_python_320.toml"))
    for case in fixtures["mdi"]
        x, y = debiased_matrix(case["x"]), Vector{Int}(case["y"])
        snapshot = copy(x)
        visits = Ref(0)
        fitting = function (model, values, labels, weights)
            visits[] += 1
            @test values == x && values !== x
            @test labels == y && labels !== y
            @test weights === nothing
            return model
        end
        records = [(features = Vector{Int}(row["features"]), thresholds = Vector{Float64}(row["thresholds"]), bootstrap_indices = Vector{Int}(row["bootstrap_indices"])) for row in case["records"]]
        actual = F.mdi_plus_importance(:supplied_forest, x, y; fit_estimator = fitting, tree_records = model -> records, n_trees = case["requested"])
        @test actual ≈ Vector{Float64}(case["output"]) atol = 2e-12 rtol = 2e-10
        @test visits[] == 1
        @test x == snapshot
        @test all(actual .>= -1e-14)
    end
end

@testset "Debiased importance: independent clones and conditional Gaussian losses" begin
    F = RiskLabAI.Features
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "debiased_importance_python_320.toml"))
    for case in fixtures["cpi"]
        x, y = debiased_matrix(case["x"]), Vector{Int}(case["y"])
        snapshot = copy(x)
        template = Dict{Symbol,Any}(:unfitted => true)
        clone_count, fit_count = Ref(0), Ref(0)
        clone_model = function (model)
            @test model === template
            clone_count[] += 1
            return Dict{Symbol,Any}(:unfitted => true, :index => clone_count[], :predictions => 0)
        end
        fit_model = function (model, values, labels, weights)
            fit_count[] += 1
            expected = case["folds"][fit_count[]]["trace"]
            @test model !== template && model[:unfitted]
            @test values == debiased_matrix(expected["x"])
            @test labels == Vector{Int}(expected["y"])
            @test weights === nothing
            model[:unfitted] = false
            model[:classes] = sort(unique(labels))
            model[:prior] = Float64[1 + count(==(label), labels) for label in model[:classes]]
            model[:feature] = sum(values * collect(1:size(values, 2))) / size(values, 1)
            return model
        end
        predict_model = function (model, values)
            model[:predictions] += 1
            expected = case["folds"][model[:index]]["trace"]["predictions"][model[:predictions]]
            @test values ≈ debiased_matrix(expected["x"]) atol = 3e-12 rtol = 3e-10
            scaled = .07 .* (values * collect(1:size(values, 2))) .+ .01 * model[:feature]
            scores = transpose(log.(model[:prior])) .+ scaled * transpose(collect(1:length(model[:classes])))
            exponentials = exp.(scores .- maximum(scores; dims = 2))
            output = exponentials ./ sum(exponentials; dims = 2)
            @test output ≈ debiased_matrix(expected["output"]) atol = 2e-13 rtol = 2e-12
            return output
        end
        folds = [(Vector{Int}(row["train"]), Vector{Int}(row["test"])) for row in case["folds"]]
        draws = [debiased_matrix(row["gaussian"]) for row in case["folds"]]
        actual = F.conditional_predictive_impact(template, x, y; clone_estimator = clone_model, fit_estimator = fit_model, predict_proba = predict_model, estimator_classes = model -> model[:classes], random_state = case["seed"], splits = folds, gaussian_draws = draws)
        @test actual.importance ≈ Vector{Float64}(case["importance"]) atol = 2e-12 rtol = 2e-8
        @test actual.p_value ≈ Vector{Float64}(case["p_value"]) atol = 2e-6 rtol = 1e-5
        @test clone_count[] == fit_count[] == length(folds)
        @test template == Dict{Symbol,Any}(:unfitted => true)
        @test x == snapshot
    end
end

@testset "Debiased importance: population-deviation gate and domains" begin
    F = RiskLabAI.Features
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "debiased_importance_python_320.toml"))
    for case in fixtures["gate"]
        actual = F._cpi_summary([Vector{Float64}(row) for row in case["differences"]])
        @test actual.importance ≈ Vector{Float64}(case["importance"]) atol = 1e-15
        @test actual.p_value ≈ Vector{Float64}(case["p_value"]) atol = 1e-13
    end
    x, y = ones(6, 2), [0, 1, 0, 1, 0, 1]
    @test_throws ArgumentError F.mdi_plus_importance(nothing, x, y; fit_estimator = (m, x, y, w) -> m, tree_records = m -> [], n_trees = -1)
    @test_throws ArgumentError F._cpi_splits(6, 4, 1, [([1, 2, 3], [3, 4, 5])])
    @test_throws ArgumentError F._cpi_splits(6, 4, 1, [([1, 2, 3], [4, 5, 6])])
    @test_throws ArgumentError F._cpi_splits(6, 1, 1, nothing)
    @test_throws ArgumentError F._cpi_summary([[1.]])
    Random.seed!(491)
    expected = rand(5)
    Random.seed!(491)
    folds = F._cpi_splits(9, 4, 19, nothing)
    @test rand(5) == expected
    @test sort(reduce(vcat, (test for (_, test) in folds))) == collect(1:9)
end
