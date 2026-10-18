using Test
using TOML
using RiskLabAI

@testset "Exact public bar state constant names and values" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "constants_python_320.toml"))
    @test length(fixture) == 31
    for (name, value) in fixture
        @test isascii(name)
        @test isdefined(RiskLabAI.Utils, Symbol(name))
        @test getproperty(RiskLabAI.Utils, Symbol(name)) == value
    end
end
