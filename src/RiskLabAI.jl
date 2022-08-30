module RiskLabAI

using LinearAlgebra,DataFrames,TimeSeries,Random

export 
    #Backtest
    ProbabilityOfBacktestOverfitting,
    
    #Betsize 
    GenerateSignal,


    #
    sampleWeight

 include("Backtests/ProbabilityOfBacktestOverfitting.jl")
 include("BetSize/BetSizing.jl")
 include("Data/Weights/SampleWeight.jl")

end



