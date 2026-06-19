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

using Dates, DataFrames, Random
using Statistics: mean, std
using Distributions: Normal, cdf, quantile

# Backtest statistics (AFML Ch. 14).
include("BacktestStatistics.jl")

# Probabilistic / deflated Sharpe ratio and test-set overfitting (AFML Ch. 8, 14).
include("ProbabilisticSharpeRatio.jl")
include("TestSetOverfitting.jl")

export
    # backtest statistics
    sharpe_ratio,
    bet_timing,
    calculate_holding_period,
    calculate_hhi,
    calculate_hhi_concentration,
    compute_drawdowns_time_under_water,
    # probabilistic Sharpe ratio
    probabilistic_sharpe_ratio,
    benchmark_sharpe_ratio,
    # test-set overfitting
    expected_max_sharpe_ratio,
    generate_max_sharpe_ratios,
    mean_std_error,
    estimated_sharpe_ratio_z_statistics,
    strategy_type1_error_probability,
    theta_for_type2_error,
    strategy_type2_error_probability

end # module Backtest
