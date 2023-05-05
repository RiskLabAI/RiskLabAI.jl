"""
List of Collaborators, Developers, and Research Assistants (in alphabetical order)
Ahmad Zaferani
"""

using Test
include("../../src/Backtests/MultipleTestReporting.jl")


@testset "BacktestResultTemplate" begin
    @testset "test arguments" begin
        @test_throws AssertionError BacktestResultTemplate(-1)
        @test_throws AssertionError BacktestResultTemplate(1, -1)
        @test_throws AssertionError BacktestResultTemplate(1, 5)
        @test_throws AssertionError BacktestResultTemplate(1, 1, nothing)
        @test BacktestResultTemplate(1, 1, nothing, 1.0) === nothing
    end

    @testset "call dependencies" begin
        @test BacktestResultTemplate(1, 1, nothing, 1.0) === nothing
    end
end
