@testset "factor mirage" begin
    @test confounder_undercontrolled_coefficient(1.0, 2.0, 3.0) ≈ 1.6
    @test confounder_undercontrolled_coefficient(2.0, 0.0, 4.0) == 2.0

    factor = confounder_factor_return(1.0, 1.0, 1.0, 3.0, 1.0)
    @test factor == StrategyPerformance(16.0, 6.25)
    forecast = confounder_forecast_return(1.0, 3.0, 1.0)
    @test forecast.correct ≈ 17.0
    @test forecast.misspecified ≈ 12.5

    coefficients = collider_overcontrolled_coefficients(1.0, 2.0, 3.0)
    @test coefficients.beta_hat ≈ -1.0
    @test coefficients.theta_hat ≈ 0.4
    collider_factor = collider_factor_return(2.0, 1.0, 1.0, 2.0, 3.0)
    @test collider_factor.correct ≈ 4.0
    @test collider_factor.misspecified ≈ -3.2
    collider_forecast = collider_forecast_return(1.0, 2.0, 3.0)
    @test collider_forecast.correct ≈ 1.0
    @test collider_forecast.misspecified ≈ -1.0

    diagnostics = collider_model_diagnostics(1.0, 2.0, 3.0, 100)
    @test diagnostics.residual_variance ≈ 0.2
    @test diagnostics.outcome_variance ≈ 2.0
    @test diagnostics.correct_r_squared ≈ 0.5
    @test diagnostics.overcontrolled_r_squared ≈ 0.9

    @test_throws ArgumentError confounder_undercontrolled_coefficient(Inf, 1.0, 1.0)
    @test_throws ArgumentError collider_model_diagnostics(1.0, 2.0, 3.0, 3)

    @test confounder_undercontrolled_coefficient(0.0, 1.0e308, 1.0e308) ≈
          1.0 rtol = 1.0e-15
    scaled_factor = confounder_factor_return(1.0e200, 0.0, 1.0e-200, 0.0, 1.0)
    @test scaled_factor.correct ≈ 1.0 rtol = 1.0e-15
    @test scaled_factor.misspecified ≈ 1.0 rtol = 1.0e-15
    scaled_collider = collider_overcontrolled_coefficients(1.0, 1.0e308, 1.0)
    @test scaled_collider.beta_hat ≈ -1.0e-308 rtol = 1.0e-15
    @test scaled_collider.theta_hat ≈ 1.0e-308 rtol = 1.0e-15

    large_coefficients = collider_overcontrolled_coefficients(1.0e308, 1.0e163, 0.0)
    @test large_coefficients.beta_hat ≈ 1.0e-18 rtol = 1.0e-14
    large_diagnostics = collider_model_diagnostics(1.0e154, 1.0e200, 0.0, 4)
    @test large_diagnostics.overcontrolled_beta_variance ≈ 2.5e-93 rtol = 1.0e-14

    latent_factor = collider_factor_return(1.0e208, 0.0, 1.0e-54, 1.0e200, 0.0)
    @test latent_factor.correct ≈ 1.0e308 rtol = 1.0e-14
    @test latent_factor.misspecified ≈ 1.0e-92 rtol = 1.0e-14
    latent_forecast = collider_forecast_return(1.0e75, 1.0e200, 0.0)
    @test latent_forecast.correct ≈ 1.0e150 rtol = 1.0e-14
    @test latent_forecast.misspecified ≈ 1.0e-250 rtol = 1.0e-14

    latent_t = collider_model_diagnostics(0.0, 1.0e-200, 1.0e-200, big(10)^200)
    @test latent_t.overcontrolled_beta_t_statistic ≈ -1.0e-300 rtol = 1.0e-14
    @test latent_t.absolute_beta_t_prefers_overcontrolled
    @test_throws ArgumentError collider_forecast_return(1.0e308, 0.0, 0.0)
end
