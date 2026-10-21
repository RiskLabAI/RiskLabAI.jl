using Test
using TOML
using Random
using RiskLabAI

const FeatureContracts = RiskLabAI.Features
feature_fixture_matrix(rows) = reduce(vcat, permutedims.(rows))
const feature_contract_fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "features_python_320.toml"))

mutable struct HawkesFixtureRNG <: AbstractRNG
    exponential::Vector{Float64}
    uniform::Vector{Float64}
    exponential_index::Int
    uniform_index::Int
end

function Random.randexp(rng::HawkesFixtureRNG)
    result = rng.exponential[rng.exponential_index]
    rng.exponential_index += 1
    return result
end

function Random.rand(rng::HawkesFixtureRNG)
    result = rng.uniform[rng.uniform_index]
    rng.uniform_index += 1
    return result
end

@testset "Supplied-prior probability ambiguity against Python 3.2.0" begin
    for row in feature_contract_fixtures["ambiguity"]
        probabilities = feature_fixture_matrix(row["probabilities"])
        result = FeatureContracts.probability_ambiguity(probabilities; prior_weights = row["prior_weights"], bin_width = row["bin_width"])
        for name in (:unscaled, :scaled, :mean_probabilities, :probability_variances, :prior_weights, :bin_width)
            @test getproperty(result, name) ≈ row["expected_" * string(name)] rtol = 1e-10 atol = 1e-12
        end
        uniform = FeatureContracts.probability_ambiguity(probabilities)
        explicit = FeatureContracts.probability_ambiguity(probabilities; prior_weights = fill(1 / size(probabilities, 1), size(probabilities, 1)))
        @test uniform == explicit
        @test isnothing(uniform.scaled) && isnothing(uniform.bin_width)
    end
    for row in feature_contract_fixtures["normal"]
        options = haskey(row, "edges") ? (; edges = row["edges"]) : (;)
        result = FeatureContracts.normal_prior_bin_probabilities(row["means"], row["standard_deviations"]; options...)
        @test result ≈ feature_fixture_matrix(row["expected"]) rtol = 1e-10 atol = 1e-12
        @test all(result .>= 0)
        @test all(abs.(sum(result; dims = 2) .- 1) .<= 1e-12)
    end
    @test FeatureContracts.probability_ambiguity([0.2 0.8; 0.2 0.8]).unscaled == 0
    @test FeatureContracts.probability_ambiguity([0.0 1.0; 1.0 0.0]; prior_weights = [1.0, 0.0]).unscaled == 0
    for probabilities in (zeros(0, 2), ones(2, 2), [1.1 -0.1], [NaN 1.0], [true false])
        @test_throws ArgumentError FeatureContracts.probability_ambiguity(probabilities)
    end
    for weights in ([0.2, 0.2], [1.0], [-0.1, 1.1], [NaN, 0.0])
        @test_throws ArgumentError FeatureContracts.probability_ambiguity([0.2 0.8; 0.4 0.6]; prior_weights = weights)
    end
    for width in (0, 1, -0.1, true, NaN)
        @test_throws ArgumentError FeatureContracts.probability_ambiguity([0.2 0.8]; bin_width = width)
    end
    @test_throws ArgumentError FeatureContracts.normal_prior_bin_probabilities([0.0], [0.0])
    @test_throws ArgumentError FeatureContracts.normal_prior_bin_probabilities([0.0], [1.0, 2.0])
    @test_throws ArgumentError FeatureContracts.normal_prior_bin_probabilities([0.0], [1.0]; edges = [0.0, 0.0])
end

@testset "Hawkes intensity, likelihood, score and supplied thinning draws" begin
    for row in feature_contract_fixtures["hawkes"]
        events = Vector{Float64}(row["events"])
        mu, alpha, beta = row["mu"], row["alpha"], row["beta"]
        @test FeatureContracts.hawkes_intensity(row["times"], events, mu, alpha, beta) ≈ row["expected_intensity"] rtol = 1e-10 atol = 1e-12
        @test FeatureContracts.hawkes_integrated_intensity(events, row["horizon"], mu, alpha, beta) ≈ row["expected_integrated"] rtol = 1e-10 atol = 1e-12
        @test FeatureContracts.hawkes_log_likelihood(events, row["horizon"], mu, alpha, beta) ≈ row["expected_likelihood"] rtol = 1e-10 atol = 1e-12
        gradient = zeros(3)
        objective = FeatureContracts._hawkes_scaled_objective_gradient!(gradient, [mu, alpha / beta, beta], events)
        @test objective ≈ -row["expected_likelihood"] rtol = 1e-10 atol = 1e-12
        @test gradient ≈ row["scaled_objective_gradient"] rtol = 1e-5 atol = 1e-7
        @test FeatureContracts.hawkes_intensity(row["times"], [events; 2.0], mu, alpha, beta) == FeatureContracts.hawkes_intensity(row["times"], events, mu, alpha, beta)
    end
    for row in feature_contract_fixtures["simulation"]
        rng = HawkesFixtureRNG(row["exponential"], row["uniform"], 1, 1)
        result = FeatureContracts.simulate_hawkes(row["horizon"], row["mu"], row["alpha"], row["beta"]; random_state = rng)
        @test result ≈ Vector{Float64}(row["expected_events"]) rtol = 1e-10 atol = 1e-12
        @test rng.exponential_index == length(row["exponential"]) + 1
        @test rng.uniform_index == length(row["uniform"]) + 1
        @test all(diff(result) .> 0) && all(0 .< result .<= row["horizon"])
    end
    @test FeatureContracts.hawkes_intensity([0.0], [0.0], 1.0, 0.5, 1.0) == [1.0]
    @test FeatureContracts.hawkes_intensity(Float64[], Float64[], 1.0, 0.5, 1.0) == Float64[]
    @test FeatureContracts.simulate_hawkes(10, 1.0, 0.5, 1.0; random_state = 112) == FeatureContracts.simulate_hawkes(10, 1.0, 0.5, 1.0; random_state = 112)
    @test_throws ErrorException FeatureContracts.simulate_hawkes(1, 1.0, 0.0, 1.0; random_state = HawkesFixtureRNG([0.01, 0.01], [0.1, 0.1], 1, 1), max_events = 1)
    @test_throws ErrorException FeatureContracts.simulate_hawkes(1, 1.0, 0.0, 1.0; random_state = HawkesFixtureRNG([0.0], [0.1], 1, 1))
    for events in ([0.1, 0.1], [0.2, 0.1], [-0.1], [Inf], [NaN], [1.1])
        @test_throws ArgumentError FeatureContracts.hawkes_log_likelihood(events, 1.0, 1.0, 0.5, 1.0)
    end
    for parameters in ((0.0, 0.2, 1.0), (1.0, 1.0, 1.0), (1.0, -0.1, 1.0), (1.0, 0.2, 0.0))
        @test_throws ArgumentError FeatureContracts.hawkes_intensity([0.1], Float64[], parameters...)
    end
    @test_throws ArgumentError FeatureContracts.simulate_hawkes(1.0, 1.0, 0.5, 1.0; max_events = true)
    @test_throws ArgumentError FeatureContracts.hawkes_intensity([-0.1], Float64[], 1.0, 0.5, 1.0)
end

@testset "Bounded Hawkes likelihood fitting and completed-bucket VPIN" begin
    for row in feature_contract_fixtures["fit"]
        events, horizon = row["events"], row["horizon"]
        result = FeatureContracts.fit_hawkes(events, horizon)
        @test result.mu > 0 && result.beta > 0 && 0 <= result.alpha < result.beta
        @test 1 <= result.converged_starts <= 3
        @test abs(result.log_likelihood - row["expected_log_likelihood"]) <= 1e-4
        @test result.log_likelihood ≈ FeatureContracts.hawkes_log_likelihood(events, horizon, result.mu, result.alpha, result.beta) rtol = 1e-12 atol = 1e-12
        @test result.branching_ratio ≈ result.alpha / result.beta rtol = 1e-12
        @test result.log_likelihood >= FeatureContracts.hawkes_log_likelihood(events, horizon, length(events) / horizon, 0.0, 1.0) - 1e-4
        if row["case"] == 1
            scaled = FeatureContracts.fit_hawkes(events * 10, horizon * 10; initial = [result.mu, result.alpha, result.beta] / 10)
            @test scaled.converged_starts == 1
            @test abs(scaled.log_likelihood + length(events) * log(10) - result.log_likelihood) <= 1e-4
        end
    end
    for row in feature_contract_fixtures["vpin"]
        result = FeatureContracts.volume_synchronized_pin(row["buy_volume"], row["sell_volume"]; bucket_volume = row["bucket_volume"], window = row["window"])
        @test result ≈ row["expected"] rtol = 1e-10 atol = 1e-12
        @test all(0 .<= result .<= 1)
    end
    @test FeatureContracts.volume_synchronized_pin([100.0, 0.0], [0.0, 100.0]; bucket_volume = 100, window = 2) == [1.0]
    @test FeatureContracts.volume_synchronized_pin([50.0, 50.0], [50.0, 50.0]; bucket_volume = 100, window = 2) == [0.0]
    @test_throws ArgumentError FeatureContracts.fit_hawkes(Float64[], 1)
    @test_throws ArgumentError FeatureContracts.fit_hawkes([0.1], 1)
    @test_throws ArgumentError FeatureContracts.fit_hawkes([0.1, 0.2], 1; initial = [1, 1, 1])
    @test_throws ArgumentError FeatureContracts.fit_hawkes([0.1, 0.2], 1; initial = [1, 0.5])
    @test_throws ArgumentError FeatureContracts.volume_synchronized_pin([50.0], [49.0]; bucket_volume = 100, window = 1)
    @test_throws ArgumentError FeatureContracts.volume_synchronized_pin([-1.0], [101.0]; bucket_volume = 100, window = 1)
    @test_throws ArgumentError FeatureContracts.volume_synchronized_pin([50.0], [50.0]; bucket_volume = 100, window = 2)
    @test_throws ArgumentError FeatureContracts.volume_synchronized_pin([50.0], [50.0]; bucket_volume = 100, window = true)
end
