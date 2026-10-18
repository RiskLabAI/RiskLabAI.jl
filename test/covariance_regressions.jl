using Test
using TOML
using LinearAlgebra
using Statistics
using Random
using Distributions
using RiskLabAI

struct CovarianceFixtureRNG <: AbstractRNG
    samples::Matrix{Float64}
end

function Random.rand(rng::CovarianceFixtureRNG, distribution::MvNormal, count::Int)
    size(rng.samples) == (count, length(distribution)) || error("Unexpected fixture sample request")
    return permutedims(rng.samples)
end

covariance_fixture_matrix(rows) = reduce(vcat, permutedims.(rows))

@testset "Ledoit-Wolf simulation adapter against supplied Python 3.2.0 draws" begin
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "covariance_python_320.toml"))
    for row in fixtures["cases"]
        samples = covariance_fixture_matrix(row["data"])
        n, p = size(samples)
        expected = covariance_fixture_matrix(row["covariance"])
        mean_result, covariance = RiskLabAI.Data.simulates_cov_mu(zeros(p), Matrix{Float64}(I, p, p), n; shrink = true, rng = CovarianceFixtureRNG(samples))
        @test mean_result ≈ row["mean"] rtol = 1e-10 atol = 1e-12
        @test covariance ≈ expected rtol = 1e-10 atol = 1e-12
        @test covariance ≈ covariance' rtol = 1e-12 atol = 1e-14
        @test eigmin(Symmetric(covariance)) >= -1e-12 * max(1, opnorm(covariance, 2))
        _, unshrunk = RiskLabAI.Data.simulates_cov_mu(zeros(p), Matrix{Float64}(I, p, p), n; rng = CovarianceFixtureRNG(samples))
        @test unshrunk ≈ covariance_fixture_matrix(row["unshrunk"]) rtol = 1e-10 atol = 1e-12
        for scale in (1e-100, 1e100)
            scaled = RiskLabAI.Data._ledoit_wolf_covariance(samples * scale)
            @test scaled / scale^2 ≈ expected rtol = 1e-10 atol = 1e-12
        end
    end
    @test RiskLabAI.Data._ledoit_wolf_covariance([1.0 2.0 3.0]) == zeros(3, 3)
    @test RiskLabAI.Data._ledoit_wolf_covariance([1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]) == 0.5 * Matrix{Float64}(I, 2, 2)
    for samples in (zeros(0, 3), zeros(3, 0), [NaN 1.0; 2.0 3.0], [Inf 1.0; 2.0 3.0])
        @test_throws ArgumentError RiskLabAI.Data._ledoit_wolf_covariance(samples)
    end
    rng = MersenneTwister(419)
    samples = permutedims(rand(copy(rng), MvNormal(zeros(2), Matrix{Float64}(I, 2, 2)), 30))
    mean_result, covariance = RiskLabAI.Data.simulates_cov_mu(zeros(2), Matrix{Float64}(I, 2, 2), 30; shrink = true, rng)
    @test mean_result ≈ vec(mean(samples; dims = 1))
    @test covariance ≈ RiskLabAI.Data._ledoit_wolf_covariance(samples)
end
