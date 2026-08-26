using Random

@testset "false-discovery rates" begin
    @testset "single and family Bayes identities" begin
        alpha = 0.05
        beta = 0.20
        pi_zero = 0.80
        expected_single =
            alpha * pi_zero / (alpha * pi_zero + (1.0 - beta) * (1.0 - pi_zero))
        @test single_trial_false_discovery_rate(alpha, beta, pi_zero) ≈ expected_single atol =
            1e-15

        alpha_family = 0.15
        beta_family = 0.40
        n_trials = 3
        family_null = pi_zero^n_trials
        expected_family =
            alpha_family * family_null /
            (alpha_family * family_null + (1.0 - beta_family) * (1.0 - family_null))
        @test family_level_false_discovery_rate(
            alpha_family,
            beta_family,
            pi_zero,
            n_trials,
        ) ≈ expected_family atol = 1e-15

        @test single_trial_false_discovery_rate(0.0, 0.0, 0.5) == 0.0
        @test single_trial_false_discovery_rate(1.0, 1.0, 0.5) == 1.0
        @test_throws ArgumentError single_trial_false_discovery_rate(0.0, 1.0, 0.5)
        @test_throws ArgumentError family_level_false_discovery_rate(0.0, 1.0, 0.5, 4)
    end

    @testset "maximum-selection family errors" begin
        alpha = 0.024997895148220373
        beta = 0.9515427737332771
        evidence = max_selection_family_errors(alpha, beta, 0.95, 10)
        @test evidence.n_trials == 10
        @test evidence.family_null_probability ≈ 0.5987369392383787 atol = 1e-15
        @test evidence.family_type_i_error ≈ 0.22365361940347483 atol = 1e-15
        @test evidence.family_type_ii_error ≈ 0.7531960884251256 atol = 1e-14
        @test family_level_false_discovery_rate(
            evidence.family_type_i_error,
            evidence.family_type_ii_error,
            evidence.trial_null_probability,
            evidence.n_trials,
        ) ≈ 0.5748603608683863 atol = 2e-15

        sensitivity = max_selection_family_errors(0.05, 0.90, 0.90, 5)
        @test sensitivity.family_null_probability ≈ 0.5904900000000001 atol = 1e-15
        @test sensitivity.family_type_i_error ≈ 0.22621906250000023 atol = 1e-15
        @test sensitivity.family_type_ii_error ≈ 0.7245771630882027 atol = 1e-14
        @test family_level_false_discovery_rate(
            sensitivity.family_type_i_error,
            sensitivity.family_type_ii_error,
            0.90,
            5,
        ) ≈ 0.5421963202650197 atol = 2e-15

        direct_alpha = 1.0 - (1.0 - 0.12)^4
        mixture_below = 0.7 * (1.0 - 0.12) + 0.3 * 0.65
        direct_beta = (mixture_below^4 - (0.7 * (1.0 - 0.12))^4) / (1.0 - 0.7^4)
        direct = max_selection_family_errors(0.12, 0.65, 0.7, 4)
        @test direct.family_type_i_error ≈ direct_alpha atol = 1e-15
        @test direct.family_type_ii_error ≈ direct_beta atol = 1e-15

        stable = max_selection_family_errors(1e-12, 1.0 - 1e-12, 0.999, 1_000_000)
        @test 0.0 <= stable.family_type_i_error <= 1.0
        @test 0.0 <= stable.family_type_ii_error <= 1.0
        @test_throws ArgumentError max_selection_family_errors(0.1, 0.2, 0.0, 2)
        @test_throws ArgumentError max_selection_family_errors(0.1, 0.2, 1.0, 2)
    end

    @testset "identification comparison" begin
        alpha = 0.05
        beta = 0.20
        pi_zero = 0.80
        n_trials = 3
        beta_family = 0.40
        required_ratio =
            (1.0 - beta_family) / (1.0 - beta) * (1.0 - pi_zero^n_trials) /
            (pi_zero^(n_trials - 1) * (1.0 - pi_zero))
        alpha_family = alpha * required_ratio
        evidence = compare_single_and_family_fdr(
            alpha,
            beta,
            pi_zero,
            n_trials;
            family_type_i_error = alpha_family,
            family_type_ii_error = beta_family,
        )
        @test evidence.identification_condition_holds
        @test evidence.single_trial_upper_bounds_family
        @test abs(evidence.equation_13_log_gap) <= 1e-14

        violated = compare_single_and_family_fdr(
            alpha,
            beta,
            pi_zero,
            n_trials;
            family_type_i_error = alpha_family / 2,
            family_type_ii_error = beta_family,
        )
        @test !violated.identification_condition_holds
        @test violated.single_trial_upper_bounds_family
    end

    @testset "CDF primitives" begin
        @test maximum_mixture_cdf(0.4, 0.8, 0.25, 3) ≈ (0.25 * 0.4 + 0.75 * 0.8)^3 atol =
            1e-15
        @test maximum_mixture_cdf(0.0, 0.0, 0.3, 5) == 0.0
        @test maximum_mixture_cdf(1.0, 1.0, 0.3, 5) == 1.0
        @test conditional_upper_tail_probability(0.75, 0.25) ≈ 1 / 3 atol = 1e-15
        @test conditional_upper_tail_probability(1.0, 0.25) == 0.0
        @test_throws ArgumentError conditional_upper_tail_probability(0.2, 0.3)
        @test_throws ArgumentError conditional_upper_tail_probability(1.0, 1.0)
    end

    @testset "Gaussian maximum model" begin
        identical = GaussianTrialMixture(0.37, 1.0, 0.0, 1.0)
        @test gaussian_trial_mixture_cdf(identical, 0.0) ≈ 0.5 atol = 1e-15
        @test gaussian_max_selection_cdf(identical, 0.0, 3) ≈ 0.125 atol = 1e-15
        expected_log_density = log(3.0) + 2.0 * log(0.5) - 0.5 * log(2.0 * pi)
        @test gaussian_max_selection_log_density(identical, 0.0, 3) ≈ expected_log_density atol =
            1e-14

        observations = [-1.0, 0.0, 0.5, 2.0]
        expected_likelihood = sum(
            gaussian_max_selection_log_density(identical, value, 7) for
            value in observations
        )
        @test gaussian_max_selection_log_likelihood(identical, observations, 7) ≈
              expected_likelihood atol = 1e-14

        asymmetric = GaussianTrialMixture(0.72, 1.3, 0.9, 0.65)
        point = 0.4
        step = 1e-5
        derivative =
            (
                gaussian_max_selection_cdf(asymmetric, point + step, 5) -
                gaussian_max_selection_cdf(asymmetric, point - step, 5)
            ) / (2.0 * step)
        @test exp(gaussian_max_selection_log_density(asymmetric, point, 5)) ≈ derivative rtol =
            2e-9 atol = 1e-12

        thresholds = [-0.25, 0.5, 1.25]
        adjusted = gaussian_search_adjusted_false_discovery_rate(asymmetric, thresholds, 4)
        @test adjusted.n_trials == 4
        @test adjusted.thresholds == thresholds
        @test all(0.0 .<= adjusted.family_type_i_errors .<= 1.0)
        @test all(0.0 .<= adjusted.family_type_ii_errors .<= 1.0)
        @test adjusted.mean_family_type_i_error ≈ mean(adjusted.family_type_i_errors)
        @test adjusted.mean_family_type_ii_error ≈ mean(adjusted.family_type_ii_errors)
        @test adjusted.family_null_probability ≈ 0.72^4 atol = 1e-15
        copied_thresholds = copy(thresholds)
        thresholds[1] = 99.0
        @test adjusted.thresholds == copied_thresholds

        @test_throws ArgumentError GaussianTrialMixture(0.5, 0.0, 1.0, 1.0)
        @test_throws ArgumentError GaussianTrialMixture(0.5, 1.0, Inf, 1.0)
        @test_throws ArgumentError gaussian_max_selection_log_likelihood(
            identical,
            Float64[],
            2,
        )
        @test_throws ArgumentError gaussian_search_adjusted_false_discovery_rate(
            identical,
            Float64[],
            2,
        )
    end

    @testset "unrestricted nonidentification witness" begin
        for observable_cdf in (0.05, 0.35, 0.90), target_fdr in (0.10, 0.55, 0.95)
            witness = fdr_nonidentification_witness(observable_cdf, target_fdr)
            @test witness.n_trials == 2
            @test witness.family_false_discovery_rate ≈ target_fdr atol = 2e-15
            @test witness.reconstructed_observable_cdf_at_threshold ≈ observable_cdf atol =
                2e-15
            @test witness.trial_null_probability^2 ≈ target_fdr atol = 2e-15
        end
        @test_throws ArgumentError fdr_nonidentification_witness(0.0, 0.5)
        @test_throws ArgumentError fdr_nonidentification_witness(0.5, 1.0)
    end

    @testset "selected-winner null probability" begin
        threshold = 0.4
        n_trials = 5
        pi_zero = 0.63
        exponential_density(x) = x >= 0.0 ? exp(-x) : 0.0
        exponential_cdf(x) = x <= 0.0 ? 0.0 : 1.0 - exp(-x)
        evidence = max_selection_null_probability(
            threshold,
            pi_zero,
            n_trials;
            null_density = exponential_density,
            mixture_cdf = exponential_cdf,
        )
        expected_selection = 1.0 - exponential_cdf(threshold)^n_trials
        @test evidence.selection_probability ≈ expected_selection atol = 1e-14
        @test evidence.joint_null_and_selection_probability ≈ pi_zero * expected_selection atol =
            2e-11
        @test evidence.conditional_null_probability ≈ pi_zero atol = 2e-11
        @test evidence.quadrature_absolute_error >= 0.0

        no_null = max_selection_null_probability(
            threshold,
            0.0,
            n_trials;
            null_density = exponential_density,
            mixture_cdf = exponential_cdf,
        )
        @test no_null.conditional_null_probability == 0.0
        @test_throws ArgumentError max_selection_null_probability(
            threshold,
            0.5,
            n_trials;
            null_density = x -> -1.0,
            mixture_cdf = exponential_cdf,
        )
        @test_throws ArgumentError max_selection_null_probability(
            threshold,
            0.5,
            n_trials;
            null_density = exponential_density,
            mixture_cdf = x -> 1.1,
        )
    end

    @testset "randomized direct-formula properties" begin
        rng = MersenneTwister(928_441)
        for _ = 1:250
            alpha = 0.001 + 0.998 * rand(rng)
            beta = 0.001 + 0.998 * rand(rng)
            pi_zero = 0.001 + 0.998 * rand(rng)
            n_trials = rand(rng, 1:20)
            evidence = max_selection_family_errors(alpha, beta, pi_zero, n_trials)
            mixture_below = pi_zero * (1.0 - alpha) + (1.0 - pi_zero) * beta
            direct_beta =
                (mixture_below^n_trials - (pi_zero * (1.0 - alpha))^n_trials) /
                (1.0 - pi_zero^n_trials)
            @test evidence.family_null_probability ≈ pi_zero^n_trials rtol = 2e-14
            @test evidence.family_type_i_error ≈ 1.0 - (1.0 - alpha)^n_trials rtol = 2e-13 atol =
                2e-15
            @test evidence.family_type_ii_error ≈ direct_beta rtol = 2e-13 atol = 2e-15
        end
    end

    @testset "boundary validation" begin
        @test_throws ArgumentError single_trial_false_discovery_rate(-0.1, 0.2, 0.5)
        @test_throws ArgumentError single_trial_false_discovery_rate(0.1, 1.2, 0.5)
        @test_throws ArgumentError single_trial_false_discovery_rate(0.1, 0.2, NaN)
        @test_throws ArgumentError family_level_false_discovery_rate(0.1, 0.2, 0.5, 0)
        @test_throws ArgumentError family_level_false_discovery_rate(0.1, 0.2, 0.5, true)
        @test_throws ArgumentError maximum_mixture_cdf(0.2, 0.3, 0.5, 1.5)
        @test_throws ArgumentError gaussian_trial_mixture_cdf(
            GaussianTrialMixture(0.5, 1.0, 1.0, 1.0),
            Inf,
        )
    end
end
