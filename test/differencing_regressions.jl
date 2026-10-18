using Test
using TOML
using RiskLabAI

differencing_equal(actual::Number, expected::Number; rtol = 1e-10, atol = 1e-12) = isnan(expected) ? isnan(actual) : isapprox(actual, expected; rtol, atol)
differencing_equal(actual::AbstractArray, expected::AbstractArray; rtol = 1e-10, atol = 1e-12) = size(actual) == size(expected) && all(differencing_equal(left, right; rtol, atol) for (left, right) in zip(actual, expected))

@testset "Shared generalized-binomial differencing weights" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "differencing_python_320.toml"))
    for case in fixture["weights"]
        actual = case["kind"] == "standard" ? RiskLabAI.Data.calculate_weights_std(case["degree"], case["count"]) : RiskLabAI.Data.calculate_weights_ffd(case["degree"], case["threshold"])
        @test differencing_equal(actual, Float64.(case["output"]))
    end
end

@testset "Shared expanding and fixed filters preserve missing-value positions" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "differencing_python_320.toml"))
    for case in fixture["filters"]
        values = Float64[case["series"]...]
        original = copy(values)
        fixed = RiskLabAI.Data.fractional_difference_fixed(values, case["degree"]; threshold = case["threshold"])
        standard = RiskLabAI.Data.fractional_difference_std(values, case["degree"]; threshold = case["threshold"])
        @test differencing_equal(fixed, Float64.(case["fixed"]))
        @test differencing_equal(standard, Float64.(case["standard"]))
        @test isequal(values, original)
    end
end

@testset "Shared db2 and Haar reflection, octave powers and Hurst estimates" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "differencing_python_320.toml"))
    for case in fixture["hurst"]
        values = Float64[case["series"]...]
        original = copy(values)
        details = RiskLabAI.Data._differencing_wavelet_details(values, case["wavelet"])
        @test length(details) == length(case["details"])
        for (actual, expected) in zip(details, case["details"])
            @test differencing_equal(actual, Float64.(expected))
        end
        actual = RiskLabAI.Data.wavelet_variance_hurst(values; wavelet = case["wavelet"], skip_fine = case["skip_fine"])
        @test differencing_equal(actual, case["wavelet_hurst"])
        @test differencing_equal(RiskLabAI.Data.rescaled_range_hurst(values), case["rescaled_range"])
        if case["wavelet"] == "db2" && case["skip_fine"] == 1
            estimates = filter(isfinite, [case["wavelet_hurst"], case["rescaled_range"]])
            expected = isempty(estimates) ? NaN : clamp(sum(estimates) / length(estimates) - 0.5, 0.0, 0.99)
            @test differencing_equal(RiskLabAI.Data.adaptive_differencing_order(values), expected)
        end
        @test values == original
    end
end

@testset "Shared ADF probabilities and complete adaptive selection" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "differencing_python_320.toml"))
    for case in fixture["adf"]
        actual = RiskLabAI.Data._adf_pvalue(Float64[case["series"]...], case["lag"])
        @test differencing_equal(actual, case["pvalue"]; rtol = 1e-9, atol = 1e-11)
    end
    for case in fixture["adaptive"]
        values = Float64[case["series"]...]
        original = copy(values)
        actual = RiskLabAI.Data.adaptive_fractional_difference(values)
        for name in (:order, :d_hat, :memory_retained, :threshold)
            @test differencing_equal(getproperty(actual, name), case[string(name)]; rtol = 1e-9, atol = 1e-11)
        end
        @test differencing_equal(actual.adf_pvalue, case["adf_pvalue"]; rtol = 1e-9, atol = 1e-11)
        @test differencing_equal(actual.series, Float64.(case["differenced"]); rtol = 1e-9, atol = 1e-11)
        @test findall(!isnan, actual.series) == Int.(case["retained_indices"])
        @test isequal(values, original)
    end
end

@testset "Native wavelet domains and finite filter termination" begin
    D = RiskLabAI.Data
    @test_throws ArgumentError D.wavelet_variance_hurst(Float64[])
    @test_throws ArgumentError D.wavelet_variance_hurst(ones(32); wavelet = "unsupported")
    @test isnan(D.wavelet_variance_hurst(ones(32); skip_fine = 20))
    @test D.wavelet_variance_hurst(collect(1.:64.); wavelet = "db1") == D.wavelet_variance_hurst(collect(1.:64.); wavelet = "haar")
    for wavelet in ("db2", "haar")
        low, high = D._differencing_wavelet_filters(wavelet)
        @test sum(low) ≈ sqrt(2.)
        @test abs(sum(high)) < 1e-15
        @test sum(abs2, low) ≈ 1.
        @test sum(abs2, high) ≈ 1.
    end
    @test_throws ArgumentError D.calculate_weights_ffd(0.5, 0.)
    @test_throws ArgumentError D.calculate_weights_ffd(NaN, 0.001)
    @test_throws ArgumentError D.calculate_weights_ffd(-1., 0.001)
    @test D.calculate_weights_ffd(-1., 2.) == [1.]
end
