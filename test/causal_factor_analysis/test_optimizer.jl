@testset "minimum variance factor weights" begin
    covariance = Diagonal([1.0, 2.0, 4.0]) |> Matrix
    exposures = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    targets = [0.0, 1.0]
    weights = minimum_variance_factor_weights(covariance, exposures, targets)
    @test weights ≈ [-2.0 / 7.0, 5.0 / 7.0, 2.0 / 7.0] atol = 1.0e-12
    @test transpose(exposures) * weights ≈ targets atol = 1.0e-12

    @test_throws ArgumentError minimum_variance_factor_weights(
        [1.0 0.0; 0.0 0.0],
        reshape([1.0, 0.0], 2, 1),
        [1.0],
    )
    @test_throws ArgumentError minimum_variance_factor_weights(
        Matrix{Float64}(I, 2, 2),
        [1.0 1.0; 1.0 1.0],
        [1.0, 0.0],
    )

    square_exposures = [1.0 1.0; 1.5 1.0]
    square_targets = [0.0, 1.0]
    expected_square = transpose(square_exposures) \ square_targets
    for square_covariance in ([1.0 0.2; 0.2 2.0], [5.0 -1.0; -1.0 0.5])
        @test minimum_variance_factor_weights(
            square_covariance,
            square_exposures,
            square_targets,
        ) ≈ expected_square atol = 1.0e-13
    end

    @test minimum_variance_factor_weights(
        [2.0 0.5; 0.5 1.0],
        reshape([1.0, 1.0], 2, 1),
        [1.0],
    ) ≈ [0.25, 0.75] atol = 1.0e-14
    @test minimum_variance_factor_weights(
        reshape([2.5], 1, 1),
        reshape([-4.0], 1, 1),
        [3.0],
    ) == [-0.75]

    near_collinear = [1.0 1.0; 1.0 1.0 + 1.0e-9]
    near_targets = [1.0, 0.0]
    near_weights = minimum_variance_factor_weights(
        Matrix{Float64}(I, 2, 2),
        near_collinear,
        near_targets,
    )
    @test transpose(near_collinear) * near_weights ≈ near_targets atol = 1.0e-12

    @test minimum_variance_factor_weights(
        reshape([1.0], 1, 1),
        reshape([1.0e308], 1, 1),
        [1.0],
    ) ≈ [1.0e-308] rtol = 1.0e-15
    @test minimum_variance_factor_weights(
        Matrix{Float64}(I, 2, 2),
        Diagonal([1.0e-200, 1.0e200]) |> Matrix,
        [1.0e-200, 1.0e200],
    ) == [1.0, 1.0]
    @test minimum_variance_factor_weights(
        Matrix{Float64}(I, 2, 2),
        reshape([1.0e-308, 1.0e-308], 2, 1),
        [2.0],
    ) ≈ [1.0e308, 1.0e308] rtol = 1.0e-15

    tiny = nextfloat(0.0)
    @test_throws ArgumentError minimum_variance_factor_weights(
        reshape([1.0], 1, 1),
        reshape([1.0e308], 1, 1),
        [tiny],
    )
    @test minimum_variance_factor_weights(
        reshape([tiny], 1, 1),
        reshape([1.0], 1, 1),
        [1.0],
    ) == [1.0]

    zero_weights = minimum_variance_factor_weights(
        Diagonal([1.0, 2.0, 3.0]) |> Matrix,
        [1.0 0.0; 0.0 1.0; 1.0 1.0],
        [0.0, 0.0],
    )
    @test zero_weights == zeros(3)

    baseline_covariance = [2.0 0.25 0.1; 0.25 1.5 -0.2; 0.1 -0.2 1.0]
    baseline_exposures = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    baseline_targets = [0.2, -0.4]
    baseline_weights = minimum_variance_factor_weights(
        baseline_covariance,
        baseline_exposures,
        baseline_targets,
    )
    for scale in (1.0e-12, 1.0, 1.0e12)
        @test minimum_variance_factor_weights(
            scale .* baseline_covariance,
            baseline_exposures,
            baseline_targets,
        ) ≈ baseline_weights rtol = 1.0e-12 atol = 1.0e-12
    end

    @test_throws ArgumentError minimum_variance_factor_weights(
        [2.0 0.30001; 0.29999 1.0],
        ones(2, 1),
        ones(1),
    )
end
