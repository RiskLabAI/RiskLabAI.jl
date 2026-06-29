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

# Multiple-testing Sharpe haircuts: Holm (FWER) + BHY (FDR) (Harvey–Liu 2015).
include("MultipleTesting.jl")

# Closed-form optimal Ornstein–Uhlenbeck trading rules (Lipton–López de Prado 2020).
include("OUTradingRules.jl")

export
    # backtest statistics
    sharpe_ratio,
    bet_timing,
    calculate_holding_period,
    calculate_hhi,
    calculate_hhi_concentration,
    compute_drawdowns_time_under_water,
    conditional_expected_drawdown,
    sharpe_difference_test,
    # probabilistic Sharpe ratio
    probabilistic_sharpe_ratio,
    benchmark_sharpe_ratio,
    # LPLZ HAC Sharpe inference
    sharpe_ratio_influence_function,
    newey_west_long_run_variance,
    newey_west_automatic_lag,
    lplz_sharpe_inference,
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
    compute_sigmoid_width,
    # multiple-testing Sharpe haircuts
    sharpe_ratio_p_values,
    holm_adjusted_p_values,
    benjamini_hochberg_yekutieli_adjusted_p_values,
    haircut_sharpe_ratios,
    # closed-form OU trading rules
    theta_from_half_life,
    stationary_std,
    hit_upper_probability,
    mean_exit_time,
    ou_rule_metrics,
    optimal_ou_trading_rule,
    fit_ornstein_uhlenbeck

end # module Backtest
