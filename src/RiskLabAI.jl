module RiskLabAI

using LinearAlgebra,DataFrames,TimeSeries,Random

export 
    #Backtest
    ProbabilityOfBacktestOverfitting,
    
    #Betsize 
    GenerateSignal,


    

 include("Backtests/ProbabilityOfBacktestOverfitting.jl")
 include("BetSize/BetSizing.jl")


end



