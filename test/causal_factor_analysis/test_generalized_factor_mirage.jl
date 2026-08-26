using LinearAlgebra
using Random
using Test

_mirage_fraction(value) = Rational{BigInt}(Float64(value))

function _generalized_confounder_covariance_oracle(
    beta,
    gamma,
    delta,
    confounder_variance,
    exposure_noise_variance,
)
    beta_value = _mirage_fraction(beta)
    gamma_value = _mirage_fraction(gamma)
    delta_value = _mirage_fraction(delta)
    confounder_value = _mirage_fraction(confounder_variance)
    noise_value = _mirage_fraction(exposure_noise_variance)
    exposure_variance = delta_value^2 * confounder_value + noise_value
    outcome_exposure_covariance =
        beta_value * exposure_variance + gamma_value * delta_value * confounder_value
    return Float64(outcome_exposure_covariance / exposure_variance)
end

function _generalized_collider_normal_equation_oracle(
    beta,
    gamma,
    delta,
    outcome_noise_variance,
    collider_noise_variance,
)
    collider_loading = gamma * beta + delta
    regressor_covariance = [
        1.0 collider_loading
        collider_loading collider_loading^2 + gamma^2 * outcome_noise_variance + collider_noise_variance
    ]
    outcome_covariance = [beta, collider_loading * beta + gamma * outcome_noise_variance]
    return regressor_covariance \ outcome_covariance
end

@testset "general-variance factor-mirage coefficients" begin
    @testset "released standardized boundary" begin
        @test confounder_undercontrolled_coefficient(1.0, 2.0, 3.0) == 1.6
        @test collider_overcontrolled_coefficients(1.0, 2.0, 3.0) ==
              ColliderCoefficients(-1.0, 0.4)
        @test_throws MethodError confounder_undercontrolled_coefficient(1.0, 2.0, 3.0, 1.0)
        @test_throws MethodError collider_overcontrolled_coefficients(1.0, 2.0, 3.0, 1.0)
    end

    @testset "confounder covariance-ratio oracle" begin
        cases = (
            (0.75, -1.25, 0.5, 2.0, 3.0),
            (-0.4, 2.5, -0.75, 0.25, 4.0),
            (1.5, 0.0, 8.0, 3.0, 0.5),
            (0.0, -3.0, 0.0, 2.0, 7.0),
        )
        for (beta, gamma, delta, confounder_variance, noise_variance) in cases
            expected = _generalized_confounder_covariance_oracle(
                beta,
                gamma,
                delta,
                confounder_variance,
                noise_variance,
            )
            actual = generalized_confounder_undercontrolled_coefficient(
                beta,
                gamma,
                delta;
                confounder_variance = confounder_variance,
                exposure_noise_variance = noise_variance,
            )
            @test actual == expected
        end

        for (beta, gamma, delta) in ((1.0, 3.0, 1.0), (-0.7, 1.2, -0.4), (2.5, 0.0, -8.0))
            @test generalized_confounder_undercontrolled_coefficient(
                beta,
                gamma,
                delta;
                confounder_variance = 1.0,
                exposure_noise_variance = 1.0,
            ) == confounder_undercontrolled_coefficient(beta, gamma, delta)
        end

        baseline = generalized_confounder_undercontrolled_coefficient(
            0.8,
            -1.4,
            0.6;
            confounder_variance = 2.5,
            exposure_noise_variance = 0.75,
        )
        rescaled = generalized_confounder_undercontrolled_coefficient(
            0.8,
            -1.4,
            0.6;
            confounder_variance = 25.0,
            exposure_noise_variance = 7.5,
        )
        @test rescaled == baseline
    end

    @testset "collider normal-equation oracle" begin
        cases = (
            (0.8, 1.2, -0.4, 2.0, 3.0),
            (-0.5, 0.75, 2.0, 0.25, 4.0),
            (2.0, -1.0, -0.25, 3.5, 0.5),
            (1.25, 0.0, 7.0, 5.0, 2.0),
        )
        for (beta, gamma, delta, outcome_variance, collider_variance) in cases
            expected = _generalized_collider_normal_equation_oracle(
                beta,
                gamma,
                delta,
                outcome_variance,
                collider_variance,
            )
            actual = generalized_collider_overcontrolled_coefficients(
                beta,
                gamma,
                delta;
                outcome_noise_variance = outcome_variance,
                collider_noise_variance = collider_variance,
            )
            @test actual.beta_hat ≈ expected[1] rtol = 2.0e-14 atol = 2.0e-14
            @test actual.theta_hat ≈ expected[2] rtol = 2.0e-14 atol = 2.0e-14
        end

        for (beta, gamma, delta) in ((0.5, 1.0, 0.8), (1.0, -2.0, 0.25), (-0.7, 0.4, -1.2))
            @test generalized_collider_overcontrolled_coefficients(
                beta,
                gamma,
                delta;
                outcome_noise_variance = 1.0,
                collider_noise_variance = 1.0,
            ) == collider_overcontrolled_coefficients(beta, gamma, delta)
        end

        baseline = generalized_collider_overcontrolled_coefficients(
            0.8,
            1.2,
            -0.4;
            outcome_noise_variance = 2.0,
            collider_noise_variance = 3.0,
        )
        rescaled = generalized_collider_overcontrolled_coefficients(
            0.8,
            1.2,
            -0.4;
            outcome_noise_variance = 20.0,
            collider_noise_variance = 30.0,
        )
        @test rescaled == baseline

        no_collider_link = generalized_collider_overcontrolled_coefficients(
            0.7,
            0.0,
            4.0;
            outcome_noise_variance = 8.0,
            collider_noise_variance = 0.5,
        )
        @test no_collider_link == ColliderCoefficients(0.7, 0.0)
    end

    @testset "deterministic randomized normal equations" begin
        rng = MersenneTwister(104729)
        for _ = 1:100
            beta, gamma, delta = randn(rng, 3)
            outcome_variance = exp(randn(rng))
            collider_variance = exp(randn(rng))
            expected = _generalized_collider_normal_equation_oracle(
                beta,
                gamma,
                delta,
                outcome_variance,
                collider_variance,
            )
            actual = generalized_collider_overcontrolled_coefficients(
                beta,
                gamma,
                delta;
                outcome_noise_variance = outcome_variance,
                collider_noise_variance = collider_variance,
            )
            @test [actual.beta_hat, actual.theta_hat] ≈ expected rtol = 5.0e-12 atol =
                5.0e-12
        end
    end

    @testset "validation" begin
        @test_throws UndefKeywordError generalized_confounder_undercontrolled_coefficient(
            1.0,
            2.0,
            3.0,
        )
        @test_throws UndefKeywordError generalized_collider_overcontrolled_coefficients(
            1.0,
            2.0,
            3.0,
        )
        @test_throws MethodError generalized_confounder_undercontrolled_coefficient(
            1.0,
            2.0,
            3.0,
            1.0,
            1.0,
        )

        for variance in (0.0, -1.0)
            @test_throws ArgumentError generalized_confounder_undercontrolled_coefficient(
                0.5,
                1.0,
                -0.25;
                confounder_variance = variance,
                exposure_noise_variance = 1.0,
            )
            @test_throws ArgumentError generalized_collider_overcontrolled_coefficients(
                0.5,
                1.0,
                -0.25;
                outcome_noise_variance = variance,
                collider_noise_variance = 1.0,
            )
        end

        for invalid in (true, NaN, Inf, 1.0 + 2.0im, "1")
            @test_throws ArgumentError generalized_confounder_undercontrolled_coefficient(
                invalid,
                1.0,
                0.5;
                confounder_variance = 1.0,
                exposure_noise_variance = 1.0,
            )
            @test_throws ArgumentError generalized_collider_overcontrolled_coefficients(
                0.5,
                1.0,
                0.5;
                outcome_noise_variance = invalid,
                collider_noise_variance = 1.0,
            )
        end
    end
end
