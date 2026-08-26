@testset "treatment effects" begin
    @test average_treatment_effect(7.25, 2.0) == 5.25
    @test average_treatment_effect(-2, -5) == 3.0

    decomposition = treatment_effect_decomposition(10.0, 4.0, 6.0)
    @test decomposition.observed_difference == 6.0
    @test decomposition.average_treatment_effect_on_treated == 4.0
    @test decomposition.sample_selection_bias == 2.0

    @test randomized_mean_difference([4.0, 6.0, 8.0], [1.0, 3.0, 5.0]) == 3.0
    did = difference_in_differences(10.0, 15.0, 20.0, 22.0)
    @test did.treated_change == 5.0
    @test did.control_change == 2.0
    @test did.estimate == 3.0

    @test backdoor_adjusted_expectation((0.1, 0.8), (0.75, 0.25)) ≈ 0.275
    @test backdoor_adjusted_average_treatment_effect((0.4, 0.9), (0.1, 0.2), (0.75, 0.25)) ≈
          0.4
    @test frontdoor_adjusted_probability(
        (0.25, 0.75),
        ((0.1, 0.5), (0.4, 0.8)),
        (0.6, 0.4),
    ) ≈ 0.485
    @test linear_instrumental_variable_effect(0.9, 0.3) ≈ 3.0
    @test linear_instrumental_variable_effect(-2.0, 0.5) == -4.0

    @test_throws ArgumentError average_treatment_effect(true, 0.0)
    @test_throws ArgumentError randomized_mean_difference((), (1.0,))
    @test_throws ArgumentError backdoor_adjusted_expectation((1.0,), (0.4, 0.4))
    @test_throws ArgumentError frontdoor_adjusted_probability((1.0,), ((1.2,),), (1.0,))
    @test_throws ArgumentError linear_instrumental_variable_effect(1.0, 0.0)
    @test_throws ArgumentError average_treatment_effect(
        floatmax(Float64),
        -floatmax(Float64),
    )
    @test_throws ArgumentError treatment_effect_decomposition(
        floatmax(Float64),
        -floatmax(Float64),
        0.0,
    )
    @test_throws ArgumentError difference_in_differences(
        -floatmax(Float64),
        floatmax(Float64),
        0.0,
        0.0,
    )
    @test_throws ArgumentError backdoor_adjusted_average_treatment_effect(
        (floatmax(Float64),),
        (-floatmax(Float64),),
        (1.0,),
    )

    @test backdoor_adjusted_expectation((2.0, 4.0), (0.5, 0.5 + 5.0e-13)) ≈ 3.0 + 2.0e-12 atol =
        1.0e-14
    @test_throws ArgumentError backdoor_adjusted_expectation((2.0, 4.0), (0.5, 0.50001))
end
