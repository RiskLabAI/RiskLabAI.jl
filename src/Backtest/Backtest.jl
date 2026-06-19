"""
    RiskLabAI.Backtest

Backtesting submodule, mirroring the Python `RiskLabAI.backtest` sub-package.

This PR wires the **backtest statistics** (Sharpe ratio, bet timing, average
holding period, Herfindahl–Hirschman concentration, drawdown / time under
water). The remaining `backtest` modules — probabilistic / deflated Sharpe,
strategy risk, probability of backtest overfitting, synthetic backtesting and
bet sizing — are wired in subsequent PRs.
"""
module Backtest

using Dates, DataFrames
using Statistics: mean, std

# Backtest statistics (AFML Ch. 14).
include("BacktestStatistics.jl")

export
    # backtest statistics
    sharpe_ratio,
    bet_timing,
    calculate_holding_period,
    calculate_hhi,
    calculate_hhi_concentration,
    compute_drawdowns_time_under_water

end # module Backtest
