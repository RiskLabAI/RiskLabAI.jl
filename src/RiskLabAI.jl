module RiskLabAI

using LinearAlgebra, DataFrames, TimeSeries, Random

# --------------------------------------------------------------------------- #
# Submodules (Phase-2 reconstruction; mirrors the Python sub-package layout).
# Wired in one at a time, each green in CI before the next.
# `Data` depends on `Utils`, so order matters.
# --------------------------------------------------------------------------- #
include("Utils/Utils.jl")
using .Utils: ewma

include("Data/Data.jl")
using .Data: AbstractBars, StandardBars, TimeBars, ExpectedImbalanceBars,
    FixedImbalanceBars, ExpectedRunBars, FixedRunBars,
    Metric, Dollar, Volume, Tick, construct_bars_from_data

include("Backtest/Backtest.jl")
using .Backtest: sharpe_ratio, bet_timing, calculate_holding_period,
    calculate_hhi, calculate_hhi_concentration, compute_drawdowns_time_under_water

# --------------------------------------------------------------------------- #
# Top-level exports.
# --------------------------------------------------------------------------- #
export
    # Utils
    ewma,
    # Data.Structures (bars)
    StandardBars, TimeBars, ExpectedImbalanceBars, FixedImbalanceBars,
    ExpectedRunBars, FixedRunBars,
    Dollar, Volume, Tick, construct_bars_from_data,
    # Backtest.statistics
    sharpe_ratio, bet_timing, calculate_holding_period,
    calculate_hhi, calculate_hhi_concentration, compute_drawdowns_time_under_water,
    # Backtest (legacy)
    probabilityOfBacktestOverfitting,
    # BetSize
    generateSignal

# --------------------------------------------------------------------------- #
# Legacy top-level includes (still loading as before; migrated into submodules
# in later PRs).
# --------------------------------------------------------------------------- #
include("Backtests/ProbabilityOfBacktestOverfitting.jl")
include("BetSize/BetSizing.jl")

end
