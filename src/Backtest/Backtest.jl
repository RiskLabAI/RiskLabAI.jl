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
using Combinatorics: combinations

# Backtest statistics (AFML Ch. 14).
include("BacktestStatistics.jl")

# Probabilistic / deflated Sharpe ratio and test-set overfitting (AFML Ch. 8, 14).
include("ProbabilisticSharpeRatio.jl")
include("TestSetOverfitting.jl")

# Strategy risk: binomial betting, implied precision, failure probability (AFML Ch. 15).
include("StrategyRisk.jl")

# Probability of backtest overfitting (CSCV) and synthetic backtesting (AFML Ch. 11-13).
include("ProbabilityOfBacktestOverfitting.jl")
include("BacktestSyntheticData.jl")

# Bet sizing: probability/meta-label sizing, signal averaging, sigmoid sizing (AFML Ch. 10).
include("BetSizing.jl")

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
    strategy_type2_error_probability,
    # strategy risk
    sharpe_ratio_trials,
    target_sharpe_ratio_symbolic,
    implied_precision,
    bin_frequency,
    binomial_sharpe_ratio,
    mix_gaussians,
    failure_probability,
    calculate_strategy_risk,
    # probability of backtest overfitting & synthetic backtesting
    performance_evaluation,
    probability_of_backtest_overfitting,
    synthetic_back_testing,
    # bet sizing
    probability_bet_size,
    average_bet_sizes,
    strategy_bet_sizing,
    mp_avg_active_signals,
    avg_active_signals,
    discrete_signal,
    generate_signal,
    bet_size_sigmoid,
    target_position,
    inverse_price,
    limit_price,
    compute_sigmoid_width

end # module Backtest
