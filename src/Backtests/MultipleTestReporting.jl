"""
List of Collaborators, Developers, and Research Assistants (in alphabetical order)
Ahmad Zaferani
"""

using DataFrames
include("../features/Clustering.jl")
include("TestSetOverfitting.jl")

#---------------------------------------------------
"""
function: a Template for Reporting Results of Backtest on Financial Strategies
refernce: Fabozzi, F, De Prado, M (2018) Being Honest in Backtest Reporting: A Template for Disclosing Multiple Tests
doi: https://doi.org/10.3905/jpm.2018.45.1.141
"""
function BacktestResultTemplate(nTrials::Int64, # number of trials
    familySize::Union{Int64,Nothing}=nothing, # number of significantly different experiments
    clusteringArgs::Union{Tuple{DataFrame,Int,Int},Nothing}=nothing, # correlation dataframe, number of clusters, number of iterations
    familyWiseErrorRate::Union{Float64,Nothing}=nothing, # family-wise error rate
    typeOneErrorArgs::Union{Tuple{Float64,Int64},Nothing}=nothing, # z statistic for the estimated Sharpe Ratios, number of tests
    CalculatePowerOfTest::Bool=false, # should calculate power of the test
    powerOfTest::Union{Float64,Nothing}=nothing, # power of the test
    typeTwoErrorArgs::Union{Tuple{Int64,Float64},Nothing}=nothing # number of tests, calculated theta parameter
)::Nothing
    @assert nTrials > 0 "nTrials must be a positive integer"

    if familySize === nothing
        println("familySize not provided; using clustering on correlation matrix of backtest returns...")
        @assert clusteringArgs !== nothing "clustering function arguments must be provided"
        correlationNew, clusters, silh = clusterKMeansTop(clusteringArgs...)
        familySize = length(clusters)
        println("calculated familySize: ", familySize)
    else
        @assert familySize > 0 "familySize must be a positive integer"
        @assert nTrials >= familySize "familySize must be less equal to nTrials"
    end

    if familyWiseErrorRate === nothing
        println("familyWiseErrorRate not provided; using Sharpe Ratio type 1 error under multiple testing...")
        @assert typeOneErrorArgs !== nothing "Type 1 Error Probability function arguments must be provided"
        familyWiseErrorRate = strategyType1ErrorProbability(typeOneErrorArgs...)
        println("calculated familyWiseErrorRate: ", familyWiseErrorRate)
    end

    if CalculatePowerOfTest == false
        println("skipping calculating optional parameter: powerOfTest ...")
    else
        @assert powerOfTest === nothing "powerOfTest must not be provided"
        println("powerOfTest not provided, using Sharpe Ratio type 2 error under multiple testing...")
        @assert typeTwoErrorArgs !== nothing "Type 2 Error Probability function arguments must be provided"
        insert!(typeTwoErrorArgs, 1, familyWiseErrorRate)
        powerOfTest = strategyType2ErrorProbability(typeTwoErrorArgs...)
        println("calculated powerOfTest: ", powerOfTest)
    end
end
