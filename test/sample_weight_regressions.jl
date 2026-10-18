using Test
using Dates
using Random
using TOML
using RiskLabAI

mutable struct SampleWeightUniformRNG <: AbstractRNG
    values::Vector{Float64}
    next::Int
end

function Random.rand(generator::SampleWeightUniformRNG)
    value = generator.values[generator.next]
    generator.next += 1
    return value
end

@testset "Sample weights against Python 3.2.0" begin
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "sample_weight_python_320.toml"))
    for fixture in fixtures["concurrency"]
        result = RiskLabAI.Data.expand_label_for_meta_labeling(
            fixture["close_index"], fixture["starts"], fixture["ends"], fixture["molecule"],
        )
        @test result.index == fixture["expected_index"]
        @test result.concurrency == fixture["expected_counts"]
        weights = RiskLabAI.Data.sample_weight_absolute_return_meta_labeling(
            fixture["starts"], fixture["ends"], fixture["close_index"], fixture["prices"], fixture["molecule"],
        )
        @test isapprox(weights, fixture["expected_weights"]; rtol = fixtures["rtol"], atol = fixtures["atol"])
    end
    calendar = Date(2026, 1, 1) .+ Day.(0:4)
    result = RiskLabAI.Data.expand_label_for_meta_labeling(calendar, calendar[1:3], calendar[[4, 3, 4]], calendar[2:3])
    @test result.index == calendar[2:4]
    @test result.concurrency == [2, 3, 2]
    result = RiskLabAI.Data.expand_label_for_meta_labeling(collect(0:4), [0, 1], [missing, 2], [1])
    @test result.index == [1, 2]
    @test result.concurrency == [2, 2]
    result = RiskLabAI.Data.expand_label_for_meta_labeling(collect(0:4), [0, 1], [NaN, 2.0], [1])
    @test result.concurrency == [2, 2]
end

@testset "Sequential bootstrap shared draws and domain checks" begin
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "sample_weight_python_320.toml"))
    bootstrap = RiskLabAI.Data.sequential_bootstrap
    for fixture in fixtures["bootstrap"]
        indicator = reduce(vcat, permutedims.(fixture["indicator_rows"]))
        original = copy(indicator)
        generator = SampleWeightUniformRNG(fixture["uniforms"], 1)
        result = bootstrap(indicator; sample_length = length(fixture["uniforms"]), random_state = generator)
        @test result == fixture["expected_one_based"]
        @test generator.next == length(fixture["uniforms"]) + 1
        @test indicator == original
    end
    indicator = [1 0 0; 1 1 0; 0 1 1; 0 0 1]
    @test bootstrap(indicator; random_state = 19) == bootstrap(indicator; random_state = 19)
    @test length(bootstrap(indicator; random_state = 19)) == size(indicator, 2)
    @test bootstrap(ones(2, 1); sample_length = 4, random_state = 0) == ones(Int, 4)
    generator = Xoshiro(29)
    untouched = copy(generator)
    @test isempty(bootstrap(indicator; sample_length = 0, random_state = generator))
    @test rand(generator) == rand(untouched)
    @test isempty(bootstrap(zeros(3, 0); random_state = 0))
    @test isempty(bootstrap(zeros(0, 0); sample_length = 0, random_state = 0))
    @test_throws ArgumentError bootstrap(zeros(0, 1))
    @test_throws ArgumentError bootstrap(zeros(2, 1))
    @test_throws ArgumentError bootstrap(zeros(3, 0); sample_length = 1)
    @test_throws ArgumentError bootstrap([1.0 NaN; 0.0 1.0])
    @test_throws ArgumentError bootstrap([1 2; 0 1])
    @test_throws ArgumentError bootstrap(ComplexF64[1 0; 0 1])
    for length in [-1, 0.5, true]
        @test_throws ArgumentError bootstrap(indicator; sample_length = length)
    end
    for seed in [-1, 0.5, true, "seed"]
        @test_throws ArgumentError bootstrap(indicator; random_state = seed)
    end
end
