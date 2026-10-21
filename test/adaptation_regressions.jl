using Test
using TOML
using LinearAlgebra
using Statistics
using RiskLabAI
using RiskLabAI.Backtest: prequential_adaptation

mutable struct AdaptationFixtureEstimator
    family::String
    penalty::Float64
    trained::Bool
    center::Float64
    feature_mean::Vector{Float64}
    weights::Vector{Float64}
end

AdaptationFixtureEstimator(family, penalty) = AdaptationFixtureEstimator(family, penalty, false, NaN, Float64[], Float64[])
adaptation_fixture_clone(model) = AdaptationFixtureEstimator(model.family, model.penalty)

function adaptation_fixture_fit!(model, features, targets)
    model.trained && error("Fixture estimator must be a fresh clone")
    model.center = mean(targets)
    if model.family == "ridge"
        model.feature_mean = vec(mean(features; dims = 1))
        centered = features .- model.feature_mean'
        model.weights = (centered' * centered + model.penalty * I) \ (centered' * (targets .- model.center))
    end
    model.trained = true
    return model
end

function adaptation_fixture_predict(model, features)
    model.trained || error("Prediction needs a fitted model")
    return model.family == "mean" ? fill(model.center, size(features, 1)) : model.center .+ (features .- model.feature_mean') * model.weights
end

adaptation_fixture_loss(target, prediction) = min(1.0, abs(target - prediction))
adaptation_fixture_vector(values) = [value == "nan" ? NaN : Float64(value) for value in values]

@testset "ADWIN2 fixed-default streaming traces against River 0.26.1" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "adaptation_python_320.toml"))
    for case in fixture["detectors"]
        window = RiskLabAI.Backtest._ADWINWindow(case["delta"])
        widths, totals, squares, alarms = Int[], Float64[], Float64[], Int[]
        summary_valid = true
        maximum_centered_error = 0.0
        for (index, value) in enumerate(case["values"])
            alarm = RiskLabAI.Backtest._adwin_update!(window, value)
            alarm && push!(alarms, index)
            push!(widths, window.width)
            push!(totals, window.total)
            push!(squares, window.centered_squares)
            counts = getfield.(window.summaries, :count)
            summary_valid &= sum(counts) == window.width && issorted(counts; rev = true)
            summary_valid &= all(ispow2, counts) && all(count(==(size), counts) <= 5 for size in unique(counts))
            summary_valid &= isapprox(sum(item.total for item in window.summaries), window.total; rtol = 1e-10, atol = 1e-11)
            reconstructed = sum(item.centered_squares + item.count * (item.total / item.count - window.total / window.width)^2 for item in window.summaries)
            maximum_centered_error = max(maximum_centered_error, abs(reconstructed - window.centered_squares))
        end
        @test widths == case["widths"]
        @test alarms == case["alarm_indices"] .+ 1
        @test isapprox(totals, case["totals"]; rtol = 1e-10, atol = 1e-12)
        @test isapprox(squares, case["centered_squares"]; rtol = 1e-10, atol = 1e-12)
        @test summary_valid
        @test maximum_centered_error < 1e-9
    end
end

@testset "Prequential forecasts, alarm rows and observed rolling refits" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "adaptation_python_320.toml"))
    for case in fixture["adaptations"]
        features = permutedims(hcat(case["features"]...))
        targets = Float64.(case["targets"])
        original_x, original_y = copy(features), copy(targets)
        template = AdaptationFixtureEstimator(case["family"], case["penalty"])
        history = Tuple{Matrix{Float64}, Vector{Float64}}[]
        function fit_observed!(model, x, y)
            push!(history, (copy(x), copy(y)))
            return adaptation_fixture_fit!(model, x, y)
        end
        settings = (fit_estimator = fit_observed!, predict_estimator = adaptation_fixture_predict,
            clone_estimator = adaptation_fixture_clone, loss_function = adaptation_fixture_loss,
            window_size = case["window_size"], min_samples = case["min_samples"], delta = case["delta"])
        output = prequential_adaptation(features, targets, template; settings...)
        warmup = case["min_samples"]
        @test all(isnan, output.predictions[1:warmup])
        @test all(isnan, output.losses[1:warmup])
        @test isapprox(output.predictions[warmup+1:end], adaptation_fixture_vector(case["predictions"])[warmup+1:end]; rtol = 1e-10, atol = 1e-12)
        @test isapprox(output.losses[warmup+1:end], adaptation_fixture_vector(case["losses"])[warmup+1:end]; rtol = 1e-10, atol = 1e-12)
        @test output.alarm_indices == case["alarm_indices"] .+ 1
        @test output.refit_indices == case["refit_indices"] .+ 1
        @test length(history) == length(case["fit_intervals"])
        for (observed, interval) in zip(history, case["fit_intervals"])
            indices = (interval[1]+1):interval[2]
            @test observed[1] == features[indices, :]
            @test observed[2] == targets[indices]
        end
        @test !template.trained
        @test features == original_x && targets == original_y
        for stop in (warmup + 1, 257, 513, length(targets) - 1)
            prefix = prequential_adaptation(features[1:stop, :], targets[1:stop], template; settings...)
            @test isequal(prefix.predictions, output.predictions[1:stop])
            @test isequal(prefix.losses, output.losses[1:stop])
            @test prefix.alarm_indices == filter(<=(stop), output.alarm_indices)
            @test prefix.refit_indices == filter(<=(stop), output.refit_indices)
        end
    end
end

@testset "Adaptation domains, independent input slices and alarm reset timing" begin
    x, y = reshape(collect(1.0:12.0), :, 1), zeros(12)
    model = AdaptationFixtureEstimator("mean", 0.1)
    settings = (fit_estimator = adaptation_fixture_fit!, predict_estimator = adaptation_fixture_predict,
        clone_estimator = adaptation_fixture_clone, loss_function = adaptation_fixture_loss,
        window_size = 5, min_samples = 2, delta = 0.01)
    for bad_delta in (0, 1, -0.1, NaN, Inf, true, "0.1")
        @test_throws ArgumentError prequential_adaptation(x, y, model; settings..., delta = bad_delta)
    end
    for count in (0, -1, true, 2.5)
        @test_throws ArgumentError prequential_adaptation(x, y, model; settings..., window_size = count)
        @test_throws ArgumentError prequential_adaptation(x, y, model; settings..., min_samples = count)
    end
    @test_throws ArgumentError prequential_adaptation(x, y, model; settings..., min_samples = 6)
    @test_throws ArgumentError prequential_adaptation(x, y, model; settings..., min_samples = 12, window_size = 12)
    @test_throws ArgumentError prequential_adaptation(x, y[1:end-1], model; settings...)
    @test_throws ArgumentError prequential_adaptation(vec(x), y, model; settings...)
    @test_throws ArgumentError prequential_adaptation(x, reshape(y, :, 1), model; settings...)
    @test_throws ArgumentError prequential_adaptation(zeros(12, 0), y, model; settings...)
    @test_throws ArgumentError prequential_adaptation(fill(NaN, size(x)), y, model; settings...)
    @test_throws ArgumentError prequential_adaptation(x, fill(Inf, length(y)), model; settings...)
    @test_throws ArgumentError prequential_adaptation(trues(size(x)), y, model; settings...)
    for bad_loss in (-0.1, 1.1, NaN, Inf, true, [0.1])
        @test_throws ArgumentError prequential_adaptation(x, y, model; settings..., loss_function = (a, b) -> bad_loss)
    end
    for bad_prediction in (1.0, [1.0, 2.0], reshape([1.0], 1, 1), [NaN], [Inf], [true])
        @test_throws ArgumentError prequential_adaptation(x, y, model; settings..., predict_estimator = (m, a) -> bad_prediction)
    end
    @test_throws ArgumentError prequential_adaptation(x, y, model; settings..., loss_function = nothing)
    @test_throws ArgumentError prequential_adaptation(x, y, model; settings..., clone_estimator = nothing)
    @test_throws ArgumentError prequential_adaptation(x, y, model; settings..., fit_estimator = nothing)
    @test_throws ArgumentError prequential_adaptation(x, y, model; settings..., predict_estimator = nothing)
    function destructive_fit!(fresh, training, targets)
        fitted = adaptation_fixture_fit!(fresh, training, targets)
        training .= -100
        targets .= -100
        return fitted
    end
    function destructive_predict(fitted, row)
        prediction = adaptation_fixture_predict(fitted, row)
        row .= -100
        return prediction
    end
    output = prequential_adaptation(x, y, model; settings..., fit_estimator = destructive_fit!, predict_estimator = destructive_predict)
    @test x == reshape(collect(1.0:12.0), :, 1)
    @test y == zeros(12)
    @test output.predictions[3:end] == zeros(10)
    window = RiskLabAI.Backtest._ADWINWindow(0.01)
    for value in [zeros(400); ones(400)]
        RiskLabAI.Backtest._adwin_update!(window, value) && break
    end
    @test window.drift_detected
    @test !RiskLabAI.Backtest._adwin_update!(window, 0.25)
    @test window.width == 1 && window.clock == 1
    @test window.total == 0.25 && window.centered_squares == 0.0
end
