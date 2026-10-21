using Test
using TOML
using Random
using RiskLabAI

ensemble_matrix(rows) = reduce(vcat, [permutedims(Vector{Float64}(row)) for row in rows])

@testset "Ensembles: theoretical tails and fitted caller-member weighting" begin
    E = RiskLabAI.Ensemble
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "ensemble_python_320.toml"))
    for case in fixtures["theory"]
        actual = E.bagging_classifier_accuracy(case["count"], case["probability"])
        @test actual ≈ case["output"] atol = 3e-16 rtol = 2e-12
        @test actual ≈ case["mathematical_tail"] atol = 3e-16 rtol = 2e-12
    end
    for case in fixtures["fitted"]
        x, y = ensemble_matrix(case["x"]), Vector{Int}(case["y"])
        x_test, y_test = ensemble_matrix(case["x_test"]), Vector{Int}(case["y_test"])
        fitting, predicting = Ref(0), Ref(0)
        fit_model = function (model, values, labels, weights)
            fitting[] += 1
            @test values == x && values !== x
            @test labels == y && labels !== y
            @test weights === nothing
            return model
        end
        members = function (model, values)
            predicting[] += 1
            if predicting[] == 1
                @test values == x
                return ensemble_matrix(case["train_predictions"])
            end
            @test values == x_test
            return ensemble_matrix(case["test_predictions"])
        end
        actual = E.bagging_evaluate_schemes(:fitted_members, x, y, x_test, y_test; fit_estimator = fit_model, tree_predictions = members, estimator_classes = model -> case["classes"])
        @test fitting[] == 1 && predicting[] == 4
        weights = E._bagging_member_weights(ensemble_matrix(case["train_predictions"]), y)
        for name in ("uniform", "c_i", "variance")
            @test actual[name] ≈ case["accuracies"][name] atol = 1e-14
            @test weights[name] ≈ Vector{Float64}(case["weights"][name]) atol = 1e-14
            @test E._bagging_member_prediction(ensemble_matrix(case["test_predictions"]), weights[name], case["classes"]) == case["predictions"][name]
        end
    end
    for case in fixtures["votes"]
        weights = E._bagging_member_weights(ensemble_matrix(case["train_predictions"]), Vector{Int}(case["y"]))
        for name in ("uniform", "c_i", "variance")
            @test weights[name] ≈ Vector{Float64}(case["weights"][name]) atol = 1e-14
            @test E._bagging_member_prediction(ensemble_matrix(case["test_predictions"]), weights[name], case["classes"]) == case["predictions"][name]
        end
    end
end

@testset "Ensembles: caller-predictor bootstrap and duplicate-index expansion" begin
    E = RiskLabAI.Ensemble
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "ensemble_python_320.toml"))
    for case in fixtures["bootstrap"]
        x, y = ensemble_matrix(case["x"]), Vector{Int}(case["y"])
        position = Ref(0)
        plans = [Vector{Int}(rows) for rows in case["row_draws"]]
        prediction = function (model, values)
            position[] += 1
            @test values == x[plans[position[]], :]
            return case["predicted"][Int.(values[:, 1]) .+ 1]
        end
        values, average, deviation = E.calculate_bootstrap_accuracy(:fitted_predictor, x, y; predict = prediction, n_bootstraps = length(plans), bootstrap_rows = plans)
        @test values ≈ Vector{Float64}(case["values"]) atol = 1e-14
        @test isapprox(average, case["mean"]; atol = 1e-14, nans = true)
        @test isapprox(deviation, case["std"]; atol = 1e-14, nans = true)
        @test position[] == length(plans)
    end
end

@testset "Ensembles: native sampling isolation and invalid dimensions" begin
    E = RiskLabAI.Ensemble
    x = reshape(collect(1.:12.), 6, 2)
    y = [0, 1, 0, 1, 0, 1]
    predictor = (model, values) -> iseven.(Int.(values[:, 1]))
    Random.seed!(702)
    expected = rand(4)
    Random.seed!(702)
    first = E.calculate_bootstrap_accuracy(nothing, x, y; predict = predictor, n_bootstraps = 8, random_state = 7)
    @test rand(4) == expected
    @test first == E.calculate_bootstrap_accuracy(nothing, x, y; predict = predictor, n_bootstraps = 8, random_state = 7)
    @test_throws ArgumentError E.bagging_classifier_accuracy(2, .5)
    @test_throws ArgumentError E._bagging_member_weights(zeros(6, 0), y)
    @test_throws DimensionMismatch E._bagging_member_weights(zeros(5, 2), y)
    @test_throws ArgumentError E._bagging_member_prediction(ones(6, 2), [.5, .5], [0])
    @test_throws DimensionMismatch E._bagging_member_prediction(ones(6, 2), [1.], [0, 1])
    @test_throws ArgumentError E.calculate_bootstrap_accuracy(nothing, x, y; predict = predictor, n_bootstraps = -1)
    @test_throws DimensionMismatch E.calculate_bootstrap_accuracy(nothing, x, y; predict = predictor, n_bootstraps = 2, bootstrap_rows = [[1, 2]])
    @test_throws ArgumentError E.calculate_bootstrap_accuracy(nothing, x, y; predict = predictor, n_bootstraps = 1, bootstrap_rows = [[0]])
end
