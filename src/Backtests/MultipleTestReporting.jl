"""
List of Collaborators, Developers, and Research Assistants (in alphabetical order)
Ahmad Zaferani
"""

using DataFrames
using Mocking
include("../features/Clustering.jl")
include("TestSetOverfitting.jl")

#---------------------------------------------------
"""
function: a Template for Reporting Results of Backtest on Financial Strategies
refernce: Fabozzi, F, De Prado, M (2018) Being Honest in Backtest Reporting: A Template for Disclosing Multiple Tests
doi: https://doi.org/10.3905/jpm.2018.45.1.141
"""
function backtestResultTemplate(nTrials::Int64, # number of trials
    familySize::Union{Int64,Nothing}=nothing, # number of significantly different experiments
    clusteringArgs::Union{Tuple{DataFrame,Int,Int},Nothing}=nothing, # correlation dataframe, number of clusters, number of iterations
    familyWiseErrorRate::Union{Float64,Nothing}=nothing, # family-wise error rate
    typeOneErrorArgs::Union{Tuple{Float64,Int64},Nothing}=nothing, # z statistic for the estimated Sharpe Ratios, number of tests
    CalculatePowerOfTest::Bool=false, # should calculate power of the test
    powerOfTest::Union{Float64,Nothing}=nothing, # power of the test
    typeTwoErrorArgs::Union{Tuple{Int64,Float64},Nothing}=nothing # number of tests, calculated theta parameter
)::String
    buffer = IOBuffer()

    @assert nTrials > 0 "nTrials must be a positive integer"

    if familySize === nothing
        println(buffer, "familySize not provided; using clustering on correlation matrix of backtest returns...")
        @assert clusteringArgs !== nothing "clustering function arguments must be provided"
        correlationNew, clusters, silh = @mock clusterKMeansTop(clusteringArgs...)
        familySize = length(clusters)
        println(buffer, "calculated familySize: ", familySize)
    else
        @assert familySize > 0 "familySize must be a positive integer"
        @assert nTrials >= familySize "familySize must be less equal to nTrials"
    end

    if familyWiseErrorRate === nothing
        println(buffer, "familyWiseErrorRate not provided; using Sharpe Ratio type 1 error under multiple testing...")
        @assert typeOneErrorArgs !== nothing "Type 1 Error Probability function arguments must be provided"
        familyWiseErrorRate = @mock strategyType1ErrorProbability(typeOneErrorArgs...)
        println(buffer, "calculated familyWiseErrorRate: ", familyWiseErrorRate)
    end

    if CalculatePowerOfTest == false
        println(buffer, "skipping calculating optional parameter: powerOfTest ...")
    else
        @assert powerOfTest === nothing "powerOfTest must not be provided"
        println(buffer, "powerOfTest not provided, using Sharpe Ratio type 2 error under multiple testing...")
        @assert typeTwoErrorArgs !== nothing "Type 2 Error Probability function arguments must be provided"
        typeTwoErrorArgs = [typeTwoErrorArgs...]
        insert!(typeTwoErrorArgs, 1, familyWiseErrorRate)
        powerOfTest = @mock strategyType2ErrorProbability(typeTwoErrorArgs...)
        println(buffer, "calculated powerOfTest: ", powerOfTest)
    end

    println(buffer, "Strategy Results are as follows:\n1. number of trials: $nTrials\n2. number of significantly different experiments: $familySize\n3. family-wise error rate: $familyWiseErrorRate\n4. power of the test: $powerOfTest")
    return String(take!(buffer))
end
