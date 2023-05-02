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
    familySize::Union{UInt64,Nothing}, #  number of significantly different experiments
    correlation::Union{DataFrame,Nothing}, # corr dataframe
    familyWiseErrorRate::Union{Float64,Nothing}, # family-wise error rate
    powerOfTest::Union{Float64,Nothing}, # power of the test
)::Nothing
    if familySize === nothing
        println("familySize not provided; using clustering on correlation matrix of backtest returns...")
        @assert correlation !== nothing "correlation must be provided"
        correlationNew, clusters, silh = clusterKMeansTop(correlation)
        familySize = length(clusters)
        println("calculated familySize: ", familySize)
    else
        @assert nTrials >= familySize "familySize must be less equal to nTrials"
    end

end
