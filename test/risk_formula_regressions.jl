using Test
using TOML
using Random
using RiskLabAI

riskformula_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isinf(right) ? left == right : isapprox(left, right; rtol = 1e-11, atol = 1e-13)
riskformula_equal(left::AbstractArray, right::AbstractArray) = size(left) == size(right) && all(riskformula_equal(a, b) for (a, b) in zip(left, right))

@testset "Shared family p-value corrections and extreme-tail haircuts" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "risk_formulas_python_320.toml"))
    for case in fixture["adjustment"]
        p = Float64[case["values"]...]
        original = copy(p)
        holm = holm_adjusted_p_values(p)
        bhy = benjamini_hochberg_yekutieli_adjusted_p_values(p)
        @test riskformula_equal(holm, Float64[case["holm"]...])
        @test riskformula_equal(bhy, Float64[case["bhy"]...])
        @test all(holm .>= p .- 1e-15)
        @test all(bhy .>= p .- 1e-15)
        @test issorted(holm[sortperm(p)])
        @test issorted(bhy[sortperm(p)])
        @test p == original
    end
    for case in fixture["haircut"]
        values = Float64[case["values"]...]
        actual = haircut_sharpe_ratios(values, case["count"]; method = case["method"], significance_level = case["level"])
        @test riskformula_equal(sharpe_ratio_p_values(values, case["count"]), Float64[case["output"]["p_values"]...])
        for (key, value) in pairs(actual)
            @test riskformula_equal(value, case["output"][string(key)])
        end
    end
    @test_throws ArgumentError haircut_sharpe_ratios([.1], 100; method = "unknown")
end

@testset "Shared expected maxima and test-set error formulas" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "risk_formulas_python_320.toml"))
    for case in fixture["maximum"]
        @test riskformula_equal(expected_max_sharpe_ratio(case["trials"], case["mean"], case["std"]), case["output"])
    end
    for case in fixture["error_rates"]
        @test riskformula_equal(strategy_type1_error_probability(case["z"]; k = case["count"]), case["alpha"])
        @test riskformula_equal(strategy_type2_error_probability(case["alpha"], case["count"], case["theta"]), case["beta"])
    end
    for case in fixture["z_statistics"]
        @test riskformula_equal(estimated_sharpe_ratio_z_statistics(case["observed"], case["count"]; true_sharpe_ratio = case["true"], skew = case["skew"], kurt = case["kurtosis"]), case["z"])
        @test riskformula_equal(theta_for_type2_error(case["observed"], case["count"], case["true"]; skew = case["skew"], kurt = case["kurtosis"]), case["theta"])
    end
end

@testset "Shared binary strategy moments and inverse requirements" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "risk_formulas_python_320.toml"))
    for case in fixture["variance"]
        @test riskformula_equal(target_sharpe_ratio_symbolic(case["probability"], case["up"], case["down"]), case["output"])
    end
    for case in fixture["binomial"]
        @test riskformula_equal(binomial_sharpe_ratio(case["loss"], case["profit"], case["frequency"], case["probability"]), case["output"])
    end
    for case in fixture["precision"]
        actual = implied_precision(case["loss"], case["profit"], case["frequency"], case["target"])
        @test riskformula_equal(actual, case["output"])
        @test 0 <= actual <= 1
        @test isapprox(binomial_sharpe_ratio(case["loss"], case["profit"], case["frequency"], actual), case["target"]; rtol = 1e-10, atol = 1e-12)
    end
    for case in fixture["frequency"]
        @test riskformula_equal(bin_frequency(case["loss"], case["profit"], case["probability"], case["target"]), case["output"])
    end
end

@testset "Shared Gaussian mixture primitives and permutations" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "risk_formulas_python_320.toml"))
    for case in fixture["mixture"]
        draws = [Float64[row...] for row in case["normal_draws"]]
        order = Int[case["permutation"]...]
        original = deepcopy((draws, order))
        actual = mix_gaussians(case["mu1"], case["mu2"], case["sigma1"], case["sigma2"], case["probability"], case["count"]; normal_draws = draws, permutation = order)
        @test riskformula_equal(actual, Float64[case["output"]...])
        @test length(actual) == case["count"]
        @test (draws, order) == original
    end
    @test mix_gaussians(.1, -.1, .2, .3, .4, 20; rng = MersenneTwister(4)) == mix_gaussians(.1, -.1, .2, .3, .4, 20; rng = MersenneTwister(4))
    @test_throws ArgumentError mix_gaussians(.1, -.1, .2, .3, 1.2, 10)
    @test_throws ArgumentError mix_gaussians(.1, -.1, -.2, .3, .4, 10)
    @test_throws DimensionMismatch mix_gaussians(.1, -.1, .2, .3, .4, 10; normal_draws = [ones(3), ones(7)])
    @test_throws ArgumentError mix_gaussians(.1, -.1, .2, .3, .4, 10; permutation = ones(Int, 10))
end
