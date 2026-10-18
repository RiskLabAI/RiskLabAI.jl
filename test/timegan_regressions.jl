using Test
using TOML
using Random
using Zygote
using Statistics
using RiskLabAI
using RiskLabAI.Data: chronological_timegan_data, TimeGAN, latent_sequence,
    timegan_losses, sample_timegan, fit_timegan, timegan_quality_report

timegan_fixture_matrix(rows) = Float64.(permutedims(hcat(rows...)))
timegan_fixture_array(windows) = cat((reshape(timegan_fixture_matrix(window), 1, length(window), length(first(window))) for window in windows)...; dims = 1)

function timegan_fixture_parameters(encoded)
    recurrent(prefix, suffix = "") = (weight_ih = timegan_fixture_matrix(encoded[prefix * "weight_ih" * suffix]),
        weight_hh = timegan_fixture_matrix(encoded[prefix * "weight_hh" * suffix]),
        bias_ih = Float64.(encoded[prefix * "bias_ih" * suffix]), bias_hh = Float64.(encoded[prefix * "bias_hh" * suffix]))
    output(prefix) = (weight = timegan_fixture_matrix(encoded[prefix * "weight"]), bias = Float64.(encoded[prefix * "bias"]))
    return (embedding = (forward = recurrent("embedding.recurrent.", "_l0"), output = output("embedding.output.")),
        recovery = (forward = recurrent("recovery.recurrent.", "_l0"), output = output("recovery.output.")),
        generator_cell = recurrent("generator_cell."), generator_output = output("generator_output."),
        discriminator = (forward = recurrent("discriminator.recurrent.", "_l0"), backward = recurrent("discriminator.recurrent.", "_l0_reverse"), output = output("discriminator.output.")))
end

function timegan_test_parameters(actual::NamedTuple, expected::NamedTuple)
    @test keys(actual) == keys(expected)
    for name in keys(actual)
        timegan_test_parameters(getproperty(actual, name), getproperty(expected, name))
    end
end

timegan_test_parameters(actual, expected) = @test isapprox(actual, expected; rtol = 1e-10, atol = 1e-12)

@testset "Chronological TimeGAN windows and training-only scaling" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "timegan_python_320.toml"))
    for case in fixture["splits"]
        data = timegan_fixture_matrix(case["series"])
        original = copy(data)
        actual = chronological_timegan_data(data; training_end = case["training_end"], window_length = case["window_length"])
        @test actual.training == timegan_fixture_array(case["training"])
        @test actual.validation == timegan_fixture_array(case["validation"])
        @test actual.minimum == case["minimum"]
        @test actual.scale == case["scale"]
        @test actual.training_end == case["training_end"]
        @test data == original
        @test all(value -> 0 <= value <= 1, actual.training)
        @test maximum(actual.validation) > 1
        changed = copy(data)
        changed[case["training_end"]+1:end, :] .+= 100
        repeated = chronological_timegan_data(changed; training_end = case["training_end"], window_length = case["window_length"])
        @test actual.training == repeated.training
        @test actual.minimum == repeated.minimum
        @test actual.scale == repeated.scale
        @test repeated.validation != actual.validation
        @test isapprox(actual.training[1, :, :] .* actual.scale' .+ actual.minimum', data[1:case["window_length"], :]; rtol = 1e-10, atol = 1e-12)
    end
    for data in (ones(6), ones(2, 3, 1), zeros(0, 2), zeros(4, 0), fill(NaN, 6, 1), fill(Inf, 6, 1), fill(true, 6, 1), fill(1im, 6, 1))
        @test_throws ArgumentError chronological_timegan_data(data; training_end = 3, window_length = 2)
    end
    for (split, length) in ((0, 2), (1, 2), (5, 2), (3, 1), (3, 0), (3.0, 2), (3, true), (true, 2))
        @test_throws ArgumentError chronological_timegan_data(ones(6, 1); training_end = split, window_length = length)
    end
end

@testset "Shared-weight TimeGAN GRUs, teacher forcing and loss gradients" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "timegan_python_320.toml"))
    for case in fixture["networks"]
        parameters = timegan_fixture_parameters(case["parameters"])
        model = TimeGAN(case["features"], case["hidden_size"]; parameters)
        data, noise = timegan_fixture_array(case["data"]), timegan_fixture_array(case["noise"])
        embedded = D._timegan_sequence_map(model.parameters.embedding, data)
        reconstructed = D._timegan_sequence_map(model.parameters.recovery, embedded)
        free = latent_sequence(model, noise)
        forced = latent_sequence(model, noise; teacher = embedded)
        real_logits = D._timegan_sequence_map(model.parameters.discriminator, embedded; sigmoid = false)
        fake_logits = D._timegan_sequence_map(model.parameters.discriminator, free; sigmoid = false)
        sampled = sample_timegan(model, size(noise, 1), size(noise, 2); seed = 52, normal_draws = noise)
        for (name, output) in pairs((; embedded, reconstructed, free, forced, real_logits, fake_logits, sampled))
            @test isapprox(output, timegan_fixture_array(case["outputs"][string(name)]); rtol = 1e-10, atol = 1e-12)
        end
        losses = timegan_losses(model, data, noise)
        for name in keys(losses)
            @test isapprox(getproperty(losses, name), case["losses"][string(name)]; rtol = 1e-10, atol = 1e-12)
        end
        objective = value -> begin
            losses = D._timegan_losses(value, data, noise)
            losses.reconstruction + 0.7losses.supervised + 0.3losses.discriminator + 0.2losses.generator_adversarial
        end
        gradient = only(Zygote.gradient(objective, model.parameters))
        timegan_test_parameters(gradient, timegan_fixture_parameters(case["gradients"]))
        changed = copy(data)
        changed[:, end, :] .= 1 .- changed[:, end, :]
        changed_embedding = D._timegan_sequence_map(model.parameters.embedding, changed)
        @test embedded[:, 1:end-1, :] == changed_embedding[:, 1:end-1, :]
        @test reconstructed[:, 1:end-1, :] == D._timegan_sequence_map(model.parameters.recovery, changed_embedding)[:, 1:end-1, :]
        changed_noise = copy(noise)
        changed_noise[:, end, :] .+= 3
        @test free[:, 1:end-1, :] == latent_sequence(model, changed_noise)[:, 1:end-1, :]
        @test free[:, 1, :] == forced[:, 1, :]
        @test forced == latent_sequence(model, noise; teacher = changed_embedding)
        changed_teacher = copy(embedded)
        changed_teacher[:, 1, :] .+= 1
        @test forced[:, 1, :] == latent_sequence(model, noise; teacher = changed_teacher)[:, 1, :]
        @test forced[:, 2:end, :] != latent_sequence(model, noise; teacher = changed_teacher)[:, 2:end, :]
        @test real_logits[:, 1, :] != D._timegan_sequence_map(model.parameters.discriminator, changed_embedding; sigmoid = false)[:, 1, :]
        @test all(value -> 0 < value < 1, sampled)
        @test model.parameters.embedding.forward.weight_ih !== parameters.embedding.forward.weight_ih
    end
    @test only(Zygote.gradient(D._timegan_logsigmoid, 0.0)) == 0.5
    @test isfinite(D._timegan_logsigmoid(-1000.0))
    @test D._timegan_logsigmoid(1000.0) == 0.0
end

@testset "TimeGAN fixed phases and shared-minibatch Adam updates" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "timegan_python_320.toml"))
    for case in fixture["networks"]
        model = TimeGAN(case["features"], case["hidden_size"]; parameters = timegan_fixture_parameters(case["parameters"]))
        data = timegan_fixture_array(case["data"])
        draws = [(indices = Int.(draw["indices"]), noise = timegan_fixture_array(draw["noise"])) for draw in case["draws"]]
        original_data, original_draws = copy(data), deepcopy(draws)
        history = fit_timegan(model, data; embedding_steps = 2, supervised_steps = 2, joint_steps = 2, batch_size = 3,
            learning_rate = case["learning_rate"], embedding_weight = case["embedding_weight"], generator_weight = case["generator_weight"], seed = 19, training_draws = draws)
        @test length(history) == 6
        @test getproperty.(history, :phase) == ["embedding", "embedding", "supervised", "supervised", "joint", "joint"]
        for (actual, expected) in zip(history, case["history"])
            for name in (:reconstruction, :supervised, :discriminator, :generator_adversarial)
                @test isapprox(getproperty(actual, name), expected[string(name)]; rtol = 1e-10, atol = 1e-12)
            end
        end
        timegan_test_parameters(model.parameters, timegan_fixture_parameters(case["trained_parameters"]))
        evaluation_noise = timegan_fixture_array(case["evaluation_noise"])
        sample = sample_timegan(model, size(evaluation_noise, 1), size(evaluation_noise, 2); seed = 53, normal_draws = evaluation_noise)
        @test isapprox(sample, timegan_fixture_array(case["evaluation_sample"]); rtol = 1e-10, atol = 1e-12)
        @test data == original_data
        @test draws == original_draws
    end
end

@testset "TimeGAN held-out descriptive quality contracts" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "timegan_python_320.toml"))
    for case in fixture["quality"]
        arrays = Tuple(timegan_fixture_array(case[name]) for name in ("real_training", "real_validation", "synthetic_training", "synthetic_validation"))
        original = deepcopy(arrays)
        actual = timegan_quality_report(arrays...)
        for name in keys(actual)
            result, expected = getproperty(actual, name), case["expected"][string(name)]
            if result isa AbstractVector
                @test length(result) == length(expected)
                @test all(isapprox.(result, expected; rtol = 1e-10, atol = 1e-12, nans = true))
            else
                @test isapprox(result, expected; rtol = 1e-10, atol = 1e-12)
            end
        end
        rt, rv, st, sv = arrays
        train = vcat(D._timegan_flatten(rt), D._timegan_flatten(st))
        center, scale = D._timegan_scaler(train)
        @test isapprox(center, case["logistic"]["center"]; rtol = 1e-10, atol = 1e-12)
        @test isapprox(scale, case["logistic"]["scale"]; rtol = 1e-10, atol = 1e-12)
        features = (train .- center') ./ scale'
        labels = [ones(size(rt, 1)); zeros(size(st, 1))]
        weights = [fill(length(labels) / (2size(rt, 1)), size(rt, 1)); fill(length(labels) / (2size(st, 1)), size(st, 1))]
        parameters = Float64.(case["logistic"]["parameters"])
        gradient = zeros(length(parameters))
        @test isfinite(D._timegan_logistic_loss!(gradient, parameters, features, labels, weights))
        @test maximum(abs, gradient) <= 1e-9
        @test arrays == original
        @test 0 <= actual.balanced_discriminator_accuracy <= 1
    end
    @test D._timegan_flatten(reshape(collect(1.0:12.0), 2, 3, 2)) == [1 7 3 9 5 11; 2 8 4 10 6 12]
    @test D._timegan_scaler([1.0 1e16; 1.0 1e16+2])[2] == [1.0, 1.0]
    for arrays in ((ones(1, 2, 1), ones(2, 2, 1), ones(2, 2, 1), ones(2, 2, 1)),
        (ones(2, 2, 1), ones(2, 3, 1), ones(2, 2, 1), ones(2, 2, 1)),
        ntuple(_ -> ones(2, 1, 1), 4), ntuple(_ -> fill(NaN, 2, 2, 1), 4))
        @test_throws ArgumentError timegan_quality_report(arrays...)
    end
end

@testset "TimeGAN local randomness and invalid training boundaries" begin
    Random.seed!(710)
    expected_global = rand(5)
    Random.seed!(710)
    first = TimeGAN(1, 2; seed = 73)
    @test rand(5) == expected_global
    second = TimeGAN(1, 2; seed = 73)
    @test first.parameters == second.parameters
    @test first.parameters != TimeGAN(1, 2; seed = 74).parameters
    @test sample_timegan(first, 3, 3; seed = 7) == sample_timegan(second, 3, 3; seed = 7)
    @test sample_timegan(first, 3, 3; seed = 7) != sample_timegan(first, 3, 3; seed = 8)
    data = fill(0.5, 4, 2, 1)
    settings = (embedding_steps = 1, supervised_steps = 1, joint_steps = 1, batch_size = 2, seed = 82)
    Random.seed!(711)
    expected_global = rand(5)
    Random.seed!(711)
    sample_timegan(first, 2, 2; seed = 1)
    history = fit_timegan(first, data; settings...)
    @test rand(5) == expected_global
    @test history == fit_timegan(second, data; settings...)
    @test first.parameters == second.parameters
    for (features, hidden) in ((0, 2), (1, 0), (true, 2), (1, 2.0), (-1, 2))
        @test_throws ArgumentError TimeGAN(features, hidden)
    end
    for seed in (true, 1.5, big(1) << 65)
        @test_throws ArgumentError TimeGAN(1, 2; seed)
    end
    @test_throws ArgumentError TimeGAN(1, 2; parameters = (;))
    for noise in (ones(2, 2), ones(2, 1, 2), ones(0, 2, 2), ones(2, 2, 3), ones(Float32, 2, 2, 2), fill(NaN, 2, 2, 2))
        @test_throws ArgumentError latent_sequence(first, noise)
    end
    @test_throws ArgumentError latent_sequence(first, ones(2, 2, 2); teacher = ones(3, 2, 2))
    @test_throws ArgumentError timegan_losses(first, fill(1.1, 2, 2, 1), ones(2, 2, 2))
    @test_throws ArgumentError timegan_losses(first, fill(0.5, 2, 2, 1), ones(3, 2, 2))
    @test_throws ArgumentError sample_timegan(first, 2, 1; seed = 1)
    @test_throws ArgumentError sample_timegan(first, 2, 2; seed = 1, normal_draws = ones(3, 2, 2))
    initial = deepcopy(first.parameters)
    for invalid in ((embedding_steps = 0,), (supervised_steps = 0,), (joint_steps = 0,), (batch_size = true,), (learning_rate = 0.0,), (embedding_weight = -1.0,), (generator_weight = Inf,))
        @test_throws ArgumentError fit_timegan(first, data; merge(settings, invalid)...)
    end
    @test_throws ArgumentError fit_timegan(first, fill(-0.1, 4, 2, 1); settings...)
    @test_throws ArgumentError fit_timegan(first, data; settings..., training_draws = [])
    @test_throws ArgumentError fit_timegan(first, data; settings..., training_draws = fill((indices = [0, 1], noise = ones(2, 2, 2)), 3))
    @test_throws ArgumentError fit_timegan(first, data; settings..., training_draws = fill((indices = [1, 1], noise = ones(2, 2, 3)), 3))
    @test first.parameters == initial
end
