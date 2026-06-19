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
    calculate_hhi, calculate_hhi_concentration, compute_drawdowns_time_under_water,
    probabilistic_sharpe_ratio, benchmark_sharpe_ratio,
    expected_max_sharpe_ratio, generate_max_sharpe_ratios, mean_std_error,
    estimated_sharpe_ratio_z_statistics, strategy_type1_error_probability,
    theta_for_type2_error, strategy_type2_error_probability,
    sharpe_ratio_trials, target_sharpe_ratio_symbolic, implied_precision,
    bin_frequency, binomial_sharpe_ratio, mix_gaussians, failure_probability,
    calculate_strategy_risk,
    performance_evaluation, probability_of_backtest_overfitting, synthetic_back_testing

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
    # Backtest — probabilistic Sharpe ratio & test-set overfitting
    probabilistic_sharpe_ratio, benchmark_sharpe_ratio,
    expected_max_sharpe_ratio, generate_max_sharpe_ratios, mean_std_error,
    estimated_sharpe_ratio_z_statistics, strategy_type1_error_probability,
    theta_for_type2_error, strategy_type2_error_probability,
    # Backtest — strategy risk
    sharpe_ratio_trials, target_sharpe_ratio_symbolic, implied_precision,
    bin_frequency, binomial_sharpe_ratio, mix_gaussians, failure_probability,
    calculate_strategy_risk,
    # Backtest — PBO & synthetic backtesting
    performance_evaluation, probability_of_backtest_overfitting, synthetic_back_testing,
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
