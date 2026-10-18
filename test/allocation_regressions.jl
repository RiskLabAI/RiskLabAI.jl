using Test
using TOML
using LinearAlgebra
using RiskLabAI

const AllocationContracts = RiskLabAI.CausalFactorAnalysis
const allocation_fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "allocation_python_320.toml"))
allocation_matrix(rows) = reduce(vcat, permutedims.(rows))

@testset "Factor equality allocations against Python 3.2.0" begin
    for row in allocation_fixtures["weights"]
        covariance, exposures, targets = allocation_matrix(row["covariance"]), allocation_matrix(row["exposures"]), copy(row["targets"])
        result = AllocationContracts.minimum_variance_factor_weights(covariance, exposures, targets)
        @test result ≈ row["expected"] rtol = 1e-10 atol = 1e-12
        scales = vec(maximum(abs.(exposures); dims = 1))
        @test all(abs.((exposures ./ scales')' * result - targets ./ scales) .<= [value == 0 ? 1e-12 : 1e-10abs(value) for value in targets ./ scales])
        @test covariance == allocation_matrix(row["covariance"]) && exposures == allocation_matrix(row["exposures"]) && targets == row["targets"]
        @test all(isfinite, result)
    end
    covariance = Matrix{Float64}(I, 3, 3)
    exposures = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    @test AllocationContracts.minimum_variance_factor_weights(covariance, exposures, zeros(2)) == zeros(3)
    @test AllocationContracts.minimum_variance_factor_weights([1.0;;], [1e-300;;], [1e-200]) ≈ [1e100]
    @test AllocationContracts.minimum_variance_factor_weights([1.0;;], [1e300;;], [1e200]) ≈ [1e-100] rtol = 1e-10 atol = 0.0
    for invalid in (zeros(0, 0), ones(2, 3), zeros(3, 3), [1.0 1.0; 1.0 1.0], [-1.0 0.0; 0.0 1.0], [1.0 0.1; 0.0 1.0], [1.0 NaN; NaN 1.0], [1im;;])
        @test_throws ArgumentError AllocationContracts.minimum_variance_factor_weights(invalid, exposures, [0.0, 1.0])
    end
    for invalid in (zeros(3, 0), zeros(3, 1), ones(3, 4), ones(3, 2), ones(2, 2), [1.0 NaN; 0.0 1.0; 1.0 1.0], complex.(exposures))
        @test_throws ArgumentError AllocationContracts.minimum_variance_factor_weights(covariance, invalid, [0.0, 1.0])
    end
    for invalid in (Float64[], [1.0], [1.0, NaN], ones(2, 1), [1im, 1im])
        @test_throws ArgumentError AllocationContracts.minimum_variance_factor_weights(covariance, exposures, invalid)
    end
    @test_throws ArgumentError AllocationContracts.minimum_variance_factor_weights([1.0;;], [1e300;;], [1e-300])
    @test_throws ArgumentError AllocationContracts.minimum_variance_factor_weights([1.0;;], [1e-300;;], [1e300])
end

@testset "Nearly dependent equality constraints" begin
    covariance = Matrix(Diagonal([1.0, 1.3, 1.8, 2.4]))
    base_one = [1.0, -1.0, 0.5, -0.5]
    base_two = [0.5, 0.2, -0.8, 0.1]
    separation = 1e-6
    exposures = hcat(ones(4), base_one, base_one + separation * base_two)
    targets = [1.0, 0.0, separation]
    weights = AllocationContracts.minimum_variance_factor_weights(covariance, exposures, targets)
    expected = [0.871437914635384, 0.500046715886808, -0.5571335140096719, 0.18564888348747993]
    @test weights ≈ expected rtol = 1e-10 atol = 1e-12
    scales = vec(maximum(abs.(exposures); dims = 1))
    scaled_exposures = exposures ./ scales'
    scaled_targets = targets ./ scales
    tolerances = [value == 0 ? 1e-12 : 1e-10 * abs(value) for value in scaled_targets]
    precise_feasibility = setprecision(BigFloat, 256) do
        residuals = abs.(BigFloat.(scaled_exposures)' * BigFloat.(weights) - BigFloat.(scaled_targets))
        all(residuals .<= BigFloat.(tolerances))
    end
    @test precise_feasibility
    unresolved_separation = 1e-8
    unresolved_exposures = hcat(ones(4), base_one, base_one + unresolved_separation * base_two)
    @test_throws ArgumentError AllocationContracts.minimum_variance_factor_weights(covariance, unresolved_exposures, [1.0, 0.0, unresolved_separation])
end

@testset "Allocation comparison evidence and copied inputs" begin
    for row in allocation_fixtures["diagnostics"]
        covariance, reference, misspecified = allocation_matrix(row["covariance"]), allocation_matrix(row["reference"]), allocation_matrix(row["misspecified"])
        reference_targets, misspecified_targets = copy(row["reference_targets"]), copy(row["misspecified_targets"])
        result = AllocationContracts.allocation_misspecification_diagnostics(covariance, reference, reference_targets, misspecified, misspecified_targets)
        @test result isa AllocationContracts.AllocationMisspecificationDiagnostics
        for name in propertynames(result)
            actual, expected = getproperty(result, name), row["expected"][string(name)]
            if name == :weight_sign_reversals
                @test collect(actual) == expected
            else
                @test actual ≈ expected rtol = 1e-10 atol = 1e-12
            end
        end
        @test all(isapprox.(reference' * result.reference_weights, reference_targets; rtol = 1e-10, atol = 1e-12))
        @test all(isapprox.(misspecified' * result.misspecified_weights, misspecified_targets; rtol = 1e-10, atol = 1e-12))
        saved = collect(result.reference_target_exposures)
        reference_targets .= 99
        @test collect(result.reference_target_exposures) == saved
        @test result.reference_weights isa Vector{Float64}
    end
    tiny = AllocationContracts.allocation_misspecification_diagnostics([1.0;;], [1.0;;], [1e-200], [1.0;;], [-1e-200])
    @test tiny.weight_sign_reversals == [true]
    @test tiny.reference_weights[1] * tiny.misspecified_weights[1] == -0.0
    zero = AllocationContracts.allocation_misspecification_diagnostics([1.0;;], [1.0;;], [0.0], [1.0;;], [-1.0])
    @test zero.weight_sign_reversals == [false]
    @test_throws ArgumentError AllocationContracts.allocation_misspecification_diagnostics([true;;], [1.0;;], [1.0], [1.0;;], [1.0])
    @test_throws ArgumentError AllocationContracts.allocation_misspecification_diagnostics([1.0;;], [true;;], [1.0], [1.0;;], [1.0])
    @test_throws ArgumentError AllocationContracts.allocation_misspecification_diagnostics([1.0;;], [1.0;;], [true], [1.0;;], [1.0])
end
