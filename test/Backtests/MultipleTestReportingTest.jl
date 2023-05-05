"""
List of Collaborators, Developers, and Research Assistants (in alphabetical order)
Ahmad Zaferani
"""

using Test
using Mocking
using DataFrames
include("../../src/Backtests/MultipleTestReporting.jl")

Mocking.activate()

function generateSuccessOutput(nTrials, familySize, familyWiseErrorRate, powerOfTest,
    familySizeNotProvided=false, calculatedFamilySize=0, familyWiseErrorRateNotProvided=false,
    calculatedFamilyWizeErrorRate=0, skipPowerOfTest=true, calculatedPowerOfTest=0)
    familySizeNotProvidedMsg = "familySize not provided; using clustering on correlation matrix of backtest returns..."
    calculatedFamilySizeMsg = "calculated familySize: $calculatedFamilySize"
    familyWiseErrorRateNotProvidedMsg = "familyWiseErrorRate not provided; using Sharpe Ratio type 1 error under multiple testing..."
    calculatedFamilyWizeErrorRateMsg = "calculated familyWiseErrorRate: $calculatedFamilyWizeErrorRate"
    skipPowerOfTestMsg = "skipping calculating optional parameter: powerOfTest ..."
    powerOfTestNotProvidedMsg = "powerOfTest not provided, using Sharpe Ratio type 2 error under multiple testing..."
    calculatedPowerOfTestMsg = "calculated powerOfTest: $calculatedPowerOfTest"

    buffer = IOBuffer()
    if familySizeNotProvided == true
        println(buffer, familySizeNotProvidedMsg)
        println(buffer, calculatedFamilySizeMsg)
        familySize = calculatedFamilySize
    end
    if familyWiseErrorRateNotProvided == true
        println(buffer, familyWiseErrorRateNotProvidedMsg)
        println(buffer, calculatedFamilyWizeErrorRateMsg)
        familyWiseErrorRate = calculatedFamilyWizeErrorRate
    end
    if skipPowerOfTest == true
        println(buffer, skipPowerOfTestMsg)
    else
        println(buffer, powerOfTestNotProvidedMsg)
        println(buffer, calculatedPowerOfTestMsg)
        powerOfTest = calculatedPowerOfTest
    end
    println(buffer, "Strategy Results are as follows:\n1. number of trials: $nTrials\n2. number of significantly different experiments: $familySize\n3. family-wise error rate: $familyWiseErrorRate\n4. power of the test: $powerOfTest")
    return String(take!(buffer))
end

@testset "backtestResultTemplate" begin
    @testset "test arguments" begin
        @test_throws AssertionError backtestResultTemplate(-1)
        @test_throws AssertionError backtestResultTemplate(1, -1)
        @test_throws AssertionError backtestResultTemplate(1, 5)
        @test_throws AssertionError backtestResultTemplate(1, 1, nothing)
        @test backtestResultTemplate(1, 1, nothing, 1.0) === generateSuccessOutput(1, 1, 1.0, nothing)
    end

    @testset "call dependencies" begin
        twentyCluster = @patch clusterKMeansTop(correlation, numberClusters=nothing, itetations=10) = nothing, 1:20, nothing
        apply(twentyCluster) do
            @test backtestResultTemplate(1, nothing, (DataFrame(), 1, 1), 1.0) == generateSuccessOutput(1, 10, 1.0, nothing, true, 20)
        end

        zeroPointSixError = @patch strategyType1ErrorProbability(z, k=1) = 0.6
        apply(zeroPointSixError) do
            @test backtestResultTemplate(20, 10, nothing, nothing, (0.1, 1)) == generateSuccessOutput(20, 10, 1.0, nothing, false, 0, true, 0.6)
        end
    end

    @testset "optional parameters" begin
        @test backtestResultTemplate(1, 1, nothing, 1.0, nothing, false, 0.44) == generateSuccessOutput(1, 1, 1.0, 0.44)
        @test_throws AssertionError backtestResultTemplate(1, 1, nothing, 1.0, nothing, true)
        @test_throws AssertionError backtestResultTemplate(1, 1, nothing, 1.0, nothing, true, 1.0)

        zeroPointFourtyFourError = @patch strategyType2ErrorProbability(α, k, θ) = 0.44
        apply(zeroPointFourtyFourError) do
            @test backtestResultTemplate(1, 1, nothing, 1.0, nothing, true, nothing, (1, 1.0)) == generateSuccessOutput(1, 1, 1.0, nothing, false, 0, false, 0, false, 0.44)
        end
    end
end
