using Test
using TOML
using RiskLabAI

@testset "Shared string entropy and overlapping word probabilities" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "entropy_python_320.toml"))
    for case in fixture["messages"]
        @test isapprox(F.shannon_entropy(case["message"]), case["shannon"]; rtol = 1e-12, atol = 1e-13)
        @test F.lempel_ziv_entropy(case["message"]) == case["lempel"]
    end
    for case in fixture["words"]
        message, width = case["message"], case["length"]
        actual = F.probability_mass_function(message, width)
        @test Set(keys(actual)) == Set(keys(case["pmf"]))
        @test all(isapprox(actual[key], value; rtol = 1e-12, atol = 1e-13) for (key, value) in case["pmf"])
        @test isapprox(F.plug_in_entropy_estimator(message, width), case["plugin"]; rtol = 1e-12, atol = 1e-13)
        @test isapprox(F.miller_madow_entropy(message, width), case["miller"]; rtol = 1e-12, atol = 1e-13)
        @test isapprox(F.grassberger_entropy(message, width), case["grassberger"]; rtol = 1e-11, atol = 1e-12)
    end
end

@testset "Shared Unicode and overlapping Kontoyiannis match conventions" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "entropy_python_320.toml"))
    for case in fixture["matches"]
        @test F.longest_match_length(case["message"], case["position"], case["window"]) == (case["length"], case["word"])
    end
    for case in fixture["kontoyiannis"]
        window = case["window"] == -1 ? nothing : case["window"]
        @test isapprox(F.kontoyiannis_entropy(case["message"]; window = window), case["output"]; rtol = 1e-12, atol = 1e-13)
    end
    @test_throws DomainError F.kontoyiannis_entropy(""; window = 3)
end

@testset "Shared finite-grid NSB posterior entropy" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "entropy_python_320.toml"))
    for case in fixture["nsb"]
        actual = F.nsb_entropy(case["message"], case["length"], case["alphabet"]; n_points = case["points"])
        @test isapprox(actual, case["output"]; rtol = 1e-9, atol = 2e-10)
        effective = max(case["alphabet"]^case["length"], length(F.probability_mass_function(case["message"], case["length"])), 1)
        @test 0 <= actual <= log2(effective)/case["length"]+1e-12
    end
    @test F.nsb_entropy("aaaa") == 0.
    @test isnan(F.nsb_entropy("aabb", 1, 2; n_points = 1))
end
