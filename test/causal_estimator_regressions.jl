using Test
using TOML
using RiskLabAI

function causal_fixture_value(value)
    if value isa AbstractVector && !isempty(value) && all(item -> item isa AbstractVector, value)
        if all(item -> all(element -> element isa Number, item), value)
            return reduce(vcat, permutedims.(value))
        end
        matrices = causal_fixture_value.(value)
        return permutedims(cat(matrices...; dims = 3), (3, 1, 2))
    end
    return value
end

function assert_causal_fixture(actual, expected, rtol, atol)
    if expected isa AbstractDict
        @test Set(String.(propertynames(actual))) == Set(keys(expected))
        for (key, value) in expected
            assert_causal_fixture(getproperty(actual, Symbol(key)), value, rtol, atol)
        end
    elseif expected isa Integer
        @test actual == expected
    else
        @test isapprox(actual, causal_fixture_value(expected); rtol, atol)
    end
end

@testset "Causal estimators against Python 3.2.0" begin
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "causal_estimators_python_320.toml"))
    for fixture in fixtures["case"]
        method = getproperty(RiskLabAI.CausalFactorAnalysis, Symbol(fixture["method"]))
        arguments = causal_fixture_value.(fixture["arguments"])
        keywords = (; (Symbol(key) => causal_fixture_value(value) for (key, value) in fixture["keywords"])...)
        result = method(arguments...; keywords...)
        assert_causal_fixture(result, fixture["expected"], fixtures["rtol"], fixtures["atol"])
    end
end

@testset "Causal domains and identification boundaries" begin
    C = RiskLabAI.CausalFactorAnalysis
    @test C.manski_ate_bounds([1.0, 2.0], [1, 1]; lower = 0, upper = 3).lower == -1.5
    @test C.manski_ate_bounds([1.0, 2.0], [0, 0]; lower = 0, upper = 3).upper == 1.5
    @test C.manski_ate_bounds([2.0, 2.0], [true, false]; lower = 2, upper = 2).lower == 0
    @test C.transported_ate([2.0, 4.0], [1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]) == 1.0
    @test_throws ArgumentError C.transported_ate([2.0, 4.0], [1.0, 1.0], [0.5, 0.5], [1.0, 0.0], [1.0, 0.0])
    @test_throws ArgumentError C.transported_ate([2.0], [1.0], [1.0], [-1.0], [1.0])
    @test_throws ArgumentError C.manski_ate_bounds([4.0], [1]; lower = 0, upper = 3)
    @test_throws ArgumentError C.manski_ate_bounds([1.0], [2]; lower = 0, upper = 3)
    @test_throws ArgumentError C.manski_ate_bounds([1.0], [1]; lower = 3, upper = 0)
    @test_throws ArgumentError C.manski_ate_bounds(Float64[], Int[]; lower = 0, upper = 3)
    @test_throws ArgumentError C.average_treatment_effect(true, 1.0)
    @test_throws ArgumentError C.average_treatment_effect(Inf, 1.0)
    @test_throws ArgumentError C.average_treatment_effect(floatmax(Float64), -floatmax(Float64))
    @test_throws ArgumentError C.randomized_mean_difference([true], [false])
    @test_throws ArgumentError C.backdoor_adjusted_expectation([1.0], [0.5])
    @test_throws ArgumentError C.backdoor_adjusted_expectation([1.0, 2.0], [1.0])
    @test_throws ArgumentError C.backdoor_adjusted_average_treatment_effect([1.0], [2.0], [-1.0])
    @test_throws ArgumentError C.frontdoor_adjusted_probability([1.0], [1.1;;], [1.0])
    @test_throws ArgumentError C.frontdoor_adjusted_probability([1.0], [0.1 0.2], [1.0])
    @test_throws ArgumentError C.linear_instrumental_variable_effect(1.0, 0.0)
    @test_throws ArgumentError C.exposure_mean_contrast([1.0], [1], [0.0 1.0]; first = 1, second = 2)
    @test_throws ArgumentError C.exposure_mean_contrast([1.0], [1], [0.7 0.7]; first = 1, second = 2)
    @test_throws ArgumentError C.exposure_mean_contrast([1.0], [true], [0.5 0.5]; first = 1, second = 2)
    @test_throws ArgumentError C.exposure_mean_contrast([1.0], [1], [0.5 0.5]; first = 1, second = 1)
    @test_throws ArgumentError C.exposure_mean_contrast([1.0], [1], [0.5 0.5]; first = true, second = 2)
    @test_throws ArgumentError C.exposure_mean_contrast([1.0], [3], [0.4 0.4 0.0]; first = 1, second = 2)
    @test_throws ArgumentError C.exposure_mean_contrast([floatmax(Float64)], [1], [1e-100 0.5]; first = 1, second = 2)
    omitted = C.exposure_mean_contrast([1.0, 2.0], [1, 2], [0.2 0.3; 0.2 0.3]; first = 1, second = 2)
    @test omitted.contrast ≈ (5 - 2 / 0.3) / 2
    behavior = fill(0.5, 4, 2)
    deterministic = [1.0 0.0; 1.0 0.0; 1.0 0.0; 1.0 0.0]
    zero_weights = C.evaluate_policy([2, 2, 2, 2], ones(4), behavior, deterministic, zeros(4, 2))
    @test zero_weights.effective_sample_size == 0
    @test zero_weights.max_weight == 0
    @test zero_weights.ipw_value == 0
    @test_throws ArgumentError C.evaluate_policy([1], [1.0], [1.0 0.0], [0.0 1.0], zeros(1, 2))
    @test_throws ArgumentError C.evaluate_policy([2], [1.0], [1.0 0.0], [1.0 0.0], zeros(1, 2))
    @test_throws ArgumentError C.evaluate_policy([1.0], [1.0], [0.5 0.5], [0.5 0.5], zeros(1, 2))
    @test_throws ArgumentError C.evaluate_policy([1], [1.0], [0.5 0.5], [0.6 0.6], zeros(1, 2))
    @test_throws ArgumentError C.evaluate_policy([1], [floatmax(Float64)], [1e-100 1.0], [1.0 0.0], zeros(1, 2))
    candidates = permutedims(cat(behavior, deterministic; dims = 3), (3, 1, 2))
    actions = [1, 2, 1, 2]
    first_result = C.select_policy(actions, [1.0, 0.0, 1.0, 0.0], behavior, candidates, zeros(4, 2); training_indices = [1, 2], evaluation_indices = [3, 4])
    perturbed = C.select_policy(actions, [1.0, 0.0, -1e8, 1e8], behavior, candidates, zeros(4, 2); training_indices = [1, 2], evaluation_indices = [3, 4])
    @test first_result.selected_policy == perturbed.selected_policy
    @test first_result.training_values == perturbed.training_values
    tied = C.select_policy(actions, zeros(4), behavior, candidates, zeros(4, 2); training_indices = [1, 2], evaluation_indices = [3, 4])
    @test tied.selected_policy == 1
    for (train, evaluation) in [([1, 1], [3, 4]), ([1, 3], [2, 4]), (Int[], [3, 4]), ([1, 2], [4, 5]), ([true], [3, 4])]
        @test_throws ArgumentError C.select_policy(actions, ones(4), behavior, candidates, zeros(4, 2); training_indices = train, evaluation_indices = evaluation)
    end
end
