@testset "additive numeric cross-language fixture" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "additive_numeric_parity.tsv")
    rows = readlines(fixture_path)
    header = split(first(rows), '\t'; keepempty = true)
    cases = [
        Dict(zip(header, split(row, '\t'; keepempty = true))) for
        row in Iterators.drop(rows, 1) if !isempty(row)
    ]
    @test [case["case_id"] for case in cases] == [
        "mirage_confounder_general",
        "mirage_collider_general",
        "fdr_single_source_b",
        "fdr_max_source_a",
        "fdr_max_sensitivity",
        "maximum_mixture_simple",
        "conditional_tail_simple",
        "gaussian_identical_cdf",
        "gaussian_identical_log_density",
        "nonidentification_simple",
        "selection_exponential",
        "allocation_two_asset",
    ]

    number(case, name) = parse(Float64, case[name])
    for case in cases
        @testset "$(case["case_id"])" begin
            method = case["method"]
            if method == "generalized_confounder"
                actual = generalized_confounder_undercontrolled_coefficient(
                    number(case, "arg1"),
                    number(case, "arg2"),
                    number(case, "arg3");
                    confounder_variance = number(case, "arg4"),
                    exposure_noise_variance = number(case, "arg5"),
                )
                @test actual ≈ number(case, "expected1") atol = 1e-15
            elseif method == "generalized_collider"
                actual = generalized_collider_overcontrolled_coefficients(
                    number(case, "arg1"),
                    number(case, "arg2"),
                    number(case, "arg3");
                    outcome_noise_variance = number(case, "arg4"),
                    collider_noise_variance = number(case, "arg5"),
                )
                @test actual.beta_hat ≈ number(case, "expected1") atol = 1e-15
                @test actual.theta_hat ≈ number(case, "expected2") atol = 1e-15
            elseif method == "single_trial_fdr"
                actual = single_trial_false_discovery_rate(
                    number(case, "arg1"),
                    number(case, "arg2"),
                    number(case, "arg3"),
                )
                @test actual ≈ number(case, "expected1") atol = 2e-15
            elseif method == "max_selection_errors"
                actual = max_selection_family_errors(
                    number(case, "arg1"),
                    number(case, "arg2"),
                    number(case, "arg3"),
                    Int(number(case, "arg4")),
                )
                @test actual.family_null_probability ≈ number(case, "expected1") atol =
                    2e-15
                @test actual.family_type_i_error ≈ number(case, "expected2") atol = 2e-15
                @test actual.family_type_ii_error ≈ number(case, "expected3") atol = 2e-14
            elseif method == "maximum_mixture_cdf"
                actual = maximum_mixture_cdf(
                    number(case, "arg1"),
                    number(case, "arg2"),
                    number(case, "arg3"),
                    Int(number(case, "arg4")),
                )
                @test actual ≈ number(case, "expected1") atol = 1e-15
            elseif method == "conditional_upper_tail"
                actual = conditional_upper_tail_probability(
                    number(case, "arg1"),
                    number(case, "arg2"),
                )
                @test actual ≈ number(case, "expected1") atol = 1e-15
            elseif method in ("gaussian_max_cdf", "gaussian_max_log_density")
                model = GaussianTrialMixture(
                    number(case, "arg1"),
                    number(case, "arg2"),
                    number(case, "arg3"),
                    number(case, "arg4"),
                )
                actual = if method == "gaussian_max_cdf"
                    gaussian_max_selection_cdf(
                        model,
                        number(case, "arg5"),
                        Int(number(case, "arg6")),
                    )
                else
                    gaussian_max_selection_log_density(
                        model,
                        number(case, "arg5"),
                        Int(number(case, "arg6")),
                    )
                end
                @test actual ≈ number(case, "expected1") atol = 2e-15
            elseif method == "nonidentification_witness"
                actual = fdr_nonidentification_witness(
                    number(case, "arg1"),
                    number(case, "arg2"),
                )
                @test actual.trial_null_probability ≈ number(case, "expected1") atol = 1e-15
                @test actual.latent_trial_cdf_at_threshold ≈ number(case, "expected2") atol =
                    1e-15
                @test actual.family_false_discovery_rate ≈ number(case, "expected3") atol =
                    1e-15
            elseif method == "selection_exponential"
                density(value) = value >= 0.0 ? exp(-value) : 0.0
                distribution(value) = value > 0.0 ? 1.0 - exp(-value) : 0.0
                actual = max_selection_null_probability(
                    number(case, "arg1"),
                    number(case, "arg2"),
                    Int(number(case, "arg3"));
                    null_density = density,
                    mixture_cdf = distribution,
                )
                @test actual.selection_probability ≈ number(case, "expected1") atol = 2e-14
                @test actual.joint_null_and_selection_probability ≈
                      number(case, "expected2") atol = 2e-11
                @test actual.conditional_null_probability ≈ number(case, "expected3") atol =
                    2e-11
            elseif method == "allocation_two_asset"
                actual = allocation_misspecification_diagnostics(
                    Matrix{Float64}(I, 2, 2),
                    reshape([1.0, 1.0], 2, 1),
                    [1.0],
                    reshape([1.0, -1.0], 2, 1),
                    [1.0],
                )
                @test actual.reference_weights ≈ fill(number(case, "expected1"), 2) atol =
                    1e-14
                @test actual.misspecified_weights ≈
                      [number(case, "expected1"), number(case, "expected2")] atol = 1e-14
                @test actual.reference_exposure_error ≈ [number(case, "expected3")] atol =
                    1e-14
                @test actual.weight_sign_reversals == [false, true]
            else
                error("unhandled parity method: $method")
            end
        end
    end
end
