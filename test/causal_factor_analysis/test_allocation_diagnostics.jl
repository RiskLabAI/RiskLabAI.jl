using LinearAlgebra
using Random
using Test

function _allocation_kkt_weights(covariance, factor_exposures, target_exposures)
    n_assets, n_factors = size(factor_exposures)
    system = [
        2.0 .* covariance factor_exposures
        transpose(factor_exposures) zeros(n_factors, n_factors)
    ]
    right_hand_side = vcat(zeros(n_assets), target_exposures)
    return (system\right_hand_side)[1:n_assets]
end

function _allocation_example_inputs()
    covariance = Diagonal([1.0, 2.0, 4.0]) |> Matrix
    reference_factor_exposures = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    reference_target_exposures = [0.0, 1.0]
    misspecified_factor_exposures = reshape([1.0, 1.0, -2.0], 3, 1)
    misspecified_target_exposures = [1.0]
    return (
        covariance,
        reference_factor_exposures,
        reference_target_exposures,
        misspecified_factor_exposures,
        misspecified_target_exposures,
    )
end

@testset "allocation misspecification diagnostics" begin
    @test fieldnames(AllocationMisspecificationDiagnostics) == (
        :reference_weights,
        :misspecified_weights,
        :reference_target_exposures,
        :reference_achieved_exposures,
        :misspecified_reference_exposures,
        :misspecified_model_target_exposures,
        :misspecified_model_achieved_exposures,
        :reference_exposure_error,
        :weight_sign_reversals,
        :reference_variance,
        :misspecified_variance,
    )
    @test fieldtypes(AllocationMisspecificationDiagnostics) == (
        Vector{Float64},
        Vector{Float64},
        Vector{Float64},
        Vector{Float64},
        Vector{Float64},
        Vector{Float64},
        Vector{Float64},
        Vector{Float64},
        Vector{Bool},
        Float64,
        Float64,
    )

    @testset "independent KKT, exposure, sign, and variance oracles" begin
        inputs = _allocation_example_inputs()
        covariance,
        reference_exposures,
        reference_target,
        misspecified_exposures,
        misspecified_target = inputs
        result = allocation_misspecification_diagnostics(inputs...)

        reference_weights =
            _allocation_kkt_weights(covariance, reference_exposures, reference_target)
        misspecified_weights =
            _allocation_kkt_weights(covariance, misspecified_exposures, misspecified_target)
        reference_achieved = transpose(reference_exposures) * reference_weights
        misspecified_reference = transpose(reference_exposures) * misspecified_weights
        misspecified_model_achieved =
            transpose(misspecified_exposures) * misspecified_weights

        @test result isa AllocationMisspecificationDiagnostics
        @test isimmutable(result)
        @test result.reference_weights ≈ reference_weights atol = 2.0e-14 rtol = 0.0
        @test result.misspecified_weights ≈ misspecified_weights atol = 2.0e-14 rtol = 0.0
        @test result.reference_target_exposures == reference_target
        @test result.reference_achieved_exposures ≈ reference_achieved atol = 2.0e-14
        @test result.misspecified_reference_exposures ≈ misspecified_reference atol =
            2.0e-14
        @test result.misspecified_model_target_exposures == misspecified_target
        @test result.misspecified_model_achieved_exposures ≈ misspecified_model_achieved atol =
            2.0e-14
        @test result.reference_exposure_error ≈ misspecified_reference - reference_target atol =
            2.0e-14
        @test result.weight_sign_reversals == [true, false, true]
        @test result.reference_variance ≈
              dot(reference_weights, covariance, reference_weights) atol = 2.0e-14
        @test result.misspecified_variance ≈
              dot(misspecified_weights, covariance, misspecified_weights) atol = 2.0e-14

        reference_stationarity = 2.0 .* covariance * reference_weights
        reference_multiplier = -(reference_exposures \ reference_stationarity)
        @test reference_stationarity + reference_exposures * reference_multiplier ≈
              zeros(length(reference_weights)) atol = 5.0e-14
    end

    @testset "declared constraints and strict sign reversal" begin
        result = allocation_misspecification_diagnostics(_allocation_example_inputs()...)
        @test result.reference_achieved_exposures ≈ result.reference_target_exposures atol =
            2.0e-14
        @test result.misspecified_model_achieved_exposures ≈
              result.misspecified_model_target_exposures atol = 2.0e-14
        @test !isapprox(
            result.misspecified_reference_exposures,
            result.reference_target_exposures;
            atol = 1.0e-12,
        )
        @test result.reference_exposure_error ≈ [0.2, -1.0] atol = 2.0e-14

        identity_covariance = Matrix{Float64}(I, 2, 2)
        identity_exposures = Matrix{Float64}(I, 2, 2)
        strict = allocation_misspecification_diagnostics(
            identity_covariance,
            identity_exposures,
            [1.0, 0.0],
            identity_exposures,
            [-1.0, 0.0],
        )
        zero_boundary = allocation_misspecification_diagnostics(
            identity_covariance,
            identity_exposures,
            [0.0, 1.0],
            identity_exposures,
            [-1.0, 1.0],
        )
        @test strict.weight_sign_reversals == [true, false]
        @test zero_boundary.weight_sign_reversals == [false, false]

        negative_zero_boundary = allocation_misspecification_diagnostics(
            identity_covariance,
            identity_exposures,
            [-0.0, 1.0],
            identity_exposures,
            [1.0, 1.0],
        )
        @test negative_zero_boundary.weight_sign_reversals == [false, false]
    end

    @testset "covariance units and independent snapshots" begin
        inputs = _allocation_example_inputs()
        baseline = allocation_misspecification_diagnostics(inputs...)
        rescaled = allocation_misspecification_diagnostics(
            7.5 .* inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
        )
        @test rescaled.reference_weights ≈ baseline.reference_weights
        @test rescaled.misspecified_weights ≈ baseline.misspecified_weights
        @test rescaled.reference_variance ≈ 7.5 * baseline.reference_variance rtol = 2.0e-14
        @test rescaled.misspecified_variance ≈ 7.5 * baseline.misspecified_variance rtol =
            2.0e-14

        mutable_inputs = map(copy, inputs)
        result = allocation_misspecification_diagnostics(mutable_inputs...)
        snapshots = (
            copy(result.reference_weights),
            copy(result.misspecified_weights),
            copy(result.reference_target_exposures),
            copy(result.reference_achieved_exposures),
            copy(result.misspecified_reference_exposures),
            copy(result.misspecified_model_target_exposures),
            copy(result.misspecified_model_achieved_exposures),
            copy(result.reference_exposure_error),
            copy(result.weight_sign_reversals),
        )
        for input in mutable_inputs
            fill!(input, 99.0)
        end
        @test result.reference_weights == snapshots[1]
        @test result.misspecified_weights == snapshots[2]
        @test result.reference_target_exposures == snapshots[3]
        @test result.reference_achieved_exposures == snapshots[4]
        @test result.misspecified_reference_exposures == snapshots[5]
        @test result.misspecified_model_target_exposures == snapshots[6]
        @test result.misspecified_model_achieved_exposures == snapshots[7]
        @test result.reference_exposure_error == snapshots[8]
        @test result.weight_sign_reversals == snapshots[9]

        field_vectors = (
            result.reference_weights,
            result.misspecified_weights,
            result.reference_target_exposures,
            result.reference_achieved_exposures,
            result.misspecified_reference_exposures,
            result.misspecified_model_target_exposures,
            result.misspecified_model_achieved_exposures,
            result.reference_exposure_error,
            result.weight_sign_reversals,
        )
        @test all(
            field_vectors[left] !== field_vectors[right] for
            left in eachindex(field_vectors) for right = (left+1):length(field_vectors)
        )
    end

    @testset "seeded random KKT systems" begin
        rng = MersenneTwister(15485863)
        for _ = 1:40
            covariance_root = randn(rng, 5, 5)
            covariance =
                transpose(covariance_root) * covariance_root +
                0.5 .* Matrix{Float64}(I, 5, 5)
            reference_exposures = randn(rng, 5, 2)
            reference_target = randn(rng, 2)
            misspecified_exposures = randn(rng, 5, 3)
            misspecified_target = randn(rng, 3)

            result = allocation_misspecification_diagnostics(
                covariance,
                reference_exposures,
                reference_target,
                misspecified_exposures,
                misspecified_target,
            )
            reference_oracle =
                _allocation_kkt_weights(covariance, reference_exposures, reference_target)
            misspecified_oracle = _allocation_kkt_weights(
                covariance,
                misspecified_exposures,
                misspecified_target,
            )

            @test result.reference_weights ≈ reference_oracle rtol = 5.0e-11 atol = 5.0e-12
            @test result.misspecified_weights ≈ misspecified_oracle rtol = 5.0e-11 atol =
                5.0e-12
            @test result.reference_achieved_exposures ≈ reference_target rtol = 5.0e-11 atol =
                5.0e-12
            @test result.misspecified_model_achieved_exposures ≈ misspecified_target rtol =
                5.0e-11 atol = 5.0e-12
            @test result.reference_variance ≈
                  dot(reference_oracle, covariance, reference_oracle) rtol = 5.0e-11
            @test result.misspecified_variance ≈
                  dot(misspecified_oracle, covariance, misspecified_oracle) rtol = 5.0e-11
        end
    end

    @testset "public boundary validation" begin
        base_inputs = _allocation_example_inputs()
        invalid_replacements = (
            (1, Bool[true false; false true]),
            (1, [1.0 0.0 0.0; 0.0 NaN 0.0; 0.0 0.0 1.0]),
            (1, Diagonal([1.0, 2.0, 0.0]) |> Matrix),
            (2, ComplexF64[1.0 + 1.0im 0.0; 0.0 1.0; 1.0 1.0]),
            (2, ones(2, 2)),
            (3, reshape(ones(2), 2, 1)),
            (4, ones(3, 4)),
            (5, ones(2)),
        )
        for (replacement_index, replacement) in invalid_replacements
            inputs = Any[base_inputs...]
            inputs[replacement_index] = replacement
            @test_throws ArgumentError allocation_misspecification_diagnostics(inputs...)
        end

        covariance, reference, target, misspecified, misspecified_target = base_inputs
        redundant_reference = hcat(reference[:, 1], reference[:, 1])
        @test_throws ArgumentError allocation_misspecification_diagnostics(
            covariance,
            redundant_reference,
            target,
            misspecified,
            misspecified_target,
        )
        @test_throws ArgumentError allocation_misspecification_diagnostics(
            zeros(0, 0),
            zeros(0, 1),
            [1.0],
            zeros(0, 1),
            [1.0],
        )
        @test_throws ArgumentError allocation_misspecification_diagnostics(
            covariance,
            zeros(3, 0),
            Float64[],
            misspecified,
            misspecified_target,
        )
        @test_throws ArgumentError allocation_misspecification_diagnostics(
            reshape([1.0e308], 1, 1),
            reshape([1.0], 1, 1),
            [2.0],
            reshape([1.0], 1, 1),
            [2.0],
        )
    end

    @testset "direct evidence construction copies every vector" begin
        vectors = [Float64[index, -index] for index = 1:8]
        reversals = [true, false]
        result = AllocationMisspecificationDiagnostics(vectors..., reversals, 1.0, 2.0)
        expected = map(copy, vectors)
        expected_reversals = copy(reversals)
        for vector in vectors
            fill!(vector, 99.0)
        end
        fill!(reversals, false)

        @test result.reference_weights == expected[1]
        @test result.misspecified_weights == expected[2]
        @test result.reference_target_exposures == expected[3]
        @test result.reference_achieved_exposures == expected[4]
        @test result.misspecified_reference_exposures == expected[5]
        @test result.misspecified_model_target_exposures == expected[6]
        @test result.misspecified_model_achieved_exposures == expected[7]
        @test result.reference_exposure_error == expected[8]
        @test result.weight_sign_reversals == expected_reversals
    end

    @testset "inputs are not mutated" begin
        inputs = _allocation_example_inputs()
        snapshots = map(copy, inputs)
        allocation_misspecification_diagnostics(inputs...)
        @test all(observed == expected for (observed, expected) in zip(inputs, snapshots))
    end
end
