using Test
using TOML
using LinearAlgebra
using RiskLabAI

mp_fit_matrix(rows) = reduce(vcat, [permutedims(Vector{Float64}(row)) for row in rows])

@testset "MP fitting: extrapolated line-search intervals" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "mp_line_search_book_cases.toml"))
    for case in fixture["fit"]
        values = Float64.(case["eigenvalues"])
        cutoff, variance = RiskLabAI.Data.find_max_eval(values, case["q"], case["bandwidth"])
        @test variance ≈ case["variance"] atol = 1e-5 rtol = 1e-6
        @test cutoff ≈ case["variance"]*(1+sqrt(1/case["q"]))^2 atol = 3e-5 rtol = 1e-6
        @test count(>(cutoff), values) == case["factors"]
    end
end

@testset "MP fitting: reference local optima and strict factor counts" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "mp_fit_python_320.toml"))
    for case in fixture["fit"]
        values = Vector{Float64}(case["eigenvalues"])
        cutoff, variance = D.find_max_eval(values, case["q"], case["bandwidth"])
        @test variance ≈ case["variance"] atol = 1e-5 rtol = 1e-6
        @test cutoff ≈ case["cutoff"] atol = 1e-5*(1+sqrt(1/case["q"]))^2 rtol = 1e-6
        @test count(>(cutoff), values) == case["factors"]
        @test D._mp_pdf_fit_error(case["variance"], case["q"], values, case["bandwidth"]) ≈ case["objective"] rtol = 2e-10 atol = 2e-11
        @test D.find_max_eval(Matrix(Diagonal(values)), case["q"], case["bandwidth"]) == (cutoff, variance)
    end
end

@testset "MP fitting: independent spectral reconstruction and portfolios" begin
    D = RiskLabAI.Data
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "mp_fit_python_320.toml"))
    for case in fixture["covariance"]
        covariance = mp_fit_matrix(case["covariance"])
        actual = D.denoise_cov(covariance, case["q"]; bandwidth = case["bandwidth"])
        @test actual ≈ mp_fit_matrix(case["output"]) rtol = 2e-10 atol = 2e-11
        @test diag(actual) ≈ diag(covariance) rtol = 2e-14 atol = 2e-14
        @test isposdef(Symmetric(actual))
        @test D.optimal_portfolio_denoised(covariance, case["q"]; mu = case["mu"], bandwidth = case["bandwidth"]) ≈ Vector{Float64}(case["portfolio"]) rtol = 2e-10 atol = 2e-11
        @test D.optimal_portfolio_denoised(covariance, case["q"]; bandwidth = case["bandwidth"]) ≈ Vector{Float64}(case["minimum"]) rtol = 2e-10 atol = 2e-11
        eigenvalues, _ = D.pca(D.cov_to_corr(covariance))
        cutoff, _ = D.find_max_eval(eigenvalues, case["q"], case["bandwidth"])
        @test count(>(cutoff), eigenvalues) == case["factors"]
    end
end

@testset "MP fitting: bounded local convergence and explicit failures" begin
    D = RiskLabAI.Data
    for (objective, expected) in ((value -> (value-.23)^2, .23), (value -> (value+2)^2, 1e-5), (value -> (value-2)^2, 1-1e-5), (value -> 3., .5))
        actual = D._mp_bounded_local_min(objective)
        @test actual.converged
        @test actual.variance ≈ expected atol = 2e-7
    end
    @test !D._mp_bounded_local_min(value -> (value-.23)^2; max_iter = 0).converged
    @test !D._mp_bounded_local_min(value -> (value-.23)^2; max_evals = 2).converged
    @test !D._mp_bounded_local_min(value -> NaN).converged
    @test_throws ArgumentError D.find_max_eval(Float64[], 2., .1)
    @test_throws ArgumentError D.find_max_eval([NaN], 2., .1)
    @test_throws ArgumentError D.find_max_eval([1.], 0., .1)
    @test_throws ArgumentError D.find_max_eval([1.], 2., 0.)
end
