module RiskLabAI

using LinearAlgebra,DataFrames,TimeSeries,Random

export
    # Backtest
    probabilityOfBacktestOverfitting,

    # BetSize
    generateSignal

include("Backtests/ProbabilityOfBacktestOverfitting.jl")
 include("BetSize/BetSizing.jl")


end



