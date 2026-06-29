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
    Metric, Dollar, Volume, Tick, construct_bars_from_data,
    nercome_denoised_covariance

include("Features/Features.jl")
using .Features: shannon_entropy, probability_mass_function, plug_in_entropy_estimator,
    lempel_ziv_entropy, longest_match_length, kontoyiannis_entropy,
    miller_madow_entropy, grassberger_entropy, nsb_entropy,
    beta_estimates, gamma_estimates, alpha_estimates, corwin_schultz_estimator,
    sigma_estimates, bekker_parkinson_volatility_estimates, edge_estimator,
    lag_dataframe, prepare_data, compute_beta, get_expanding_window_adf,
    get_bsadf_statistic, psy_minimum_window, get_sadf_sequence, get_bsadf_sequence,
    get_gsadf_statistic, get_bubble_episodes, simulate_psy_critical_values,
    orthogonal_features, calculate_weighted_tau,
    feature_importance_mdi, feature_importance_mda, feature_importance_sfi,
    get_test_dataset, mdi_plus_importance, conditional_predictive_impact

include("Cluster/Cluster.jl")
using .Cluster: covariance_to_correlation, silhouette_samples, cluster_k_means_base,
    cluster_k_means_top, make_new_outputs, random_covariance_sub,
    random_block_covariance, random_block_correlation

# Optimization depends on Cluster (nco uses the k-means backend), so it loads after.
include("Optimization/Optimization.jl")
using .Optimization: inverse_variance_weights, cluster_variance, quasi_diagonal,
    recursive_bisection, distance_corr, hrp, pca_weights,
    get_optimal_portfolio_weights, get_optimal_portfolio_weights_nco

include("Backtest/Backtest.jl")
using .Backtest: sharpe_ratio, bet_timing, calculate_holding_period,
    calculate_hhi, calculate_hhi_concentration, compute_drawdowns_time_under_water,
    conditional_expected_drawdown, sharpe_difference_test,
    probabilistic_sharpe_ratio, benchmark_sharpe_ratio,
    expected_max_sharpe_ratio, generate_max_sharpe_ratios, mean_std_error,
    estimated_sharpe_ratio_z_statistics, strategy_type1_error_probability,
    theta_for_type2_error, strategy_type2_error_probability,
    sharpe_ratio_trials, target_sharpe_ratio_symbolic, implied_precision,
    bin_frequency, binomial_sharpe_ratio, mix_gaussians, failure_probability,
    calculate_strategy_risk,
    performance_evaluation, probability_of_backtest_overfitting, synthetic_back_testing,
    probability_bet_size, average_bet_sizes, strategy_bet_sizing, mp_avg_active_signals,
    avg_active_signals, discrete_signal, generate_signal, bet_size_sigmoid,
    target_position, inverse_price, limit_price, compute_sigmoid_width,
    sharpe_ratio_p_values, holm_adjusted_p_values,
    benjamini_hochberg_yekutieli_adjusted_p_values, haircut_sharpe_ratios,
    sharpe_ratio_influence_function, newey_west_long_run_variance,
    newey_west_automatic_lag, lplz_sharpe_inference,
    theta_from_half_life, hit_upper_probability, mean_exit_time, ou_rule_metrics,
    optimal_ou_trading_rule, fit_ornstein_uhlenbeck

include("Validation/Validation.jl")
using .Validation: KFoldCV, PurgedKFoldCV, CombinatorialPurgedCV, WalkForwardCV,
    cv_split, backtest_paths, get_n_splits, cross_val_score,
    grid_search_cv, random_search_cv, leakage_aware_hpo, deflated_sharpe_gate,
    moving_block_bootstrap_indices, bagged_probability_of_backtest_overfitting,
    estimate_volatility_regimes, adaptive_probability_of_backtest_overfitting

include("Ensemble/Ensemble.jl")
using .Ensemble: bagging_classifier_accuracy, fit_bagging, bagging_evaluate_schemes,
    calculate_bootstrap_accuracy

include("Pde/Pde.jl")
using .Pde: Equation, HJBLQ, BlackScholesBarenblatt, PricingDefaultRisk, PricingDiffRate,
    pde_sample, pde_driver, pde_hamiltonian, pde_terminal, pde_sigma, solve_deep_bsde

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
    # Data.Denoise (NERCOME sample-split denoiser)
    nercome_denoised_covariance,
    # Features — entropy
    shannon_entropy, probability_mass_function, plug_in_entropy_estimator,
    lempel_ziv_entropy, longest_match_length, kontoyiannis_entropy,
    # Features — microstructural
    beta_estimates, gamma_estimates, alpha_estimates, corwin_schultz_estimator,
    sigma_estimates, bekker_parkinson_volatility_estimates, edge_estimator,
    # Features — structural breaks
    lag_dataframe, prepare_data, compute_beta, get_expanding_window_adf,
    get_bsadf_statistic, psy_minimum_window, get_sadf_sequence, get_bsadf_sequence,
    get_gsadf_statistic, get_bubble_episodes, simulate_psy_critical_values,
    # Features — feature importance
    orthogonal_features, calculate_weighted_tau,
    feature_importance_mdi, feature_importance_mda, feature_importance_sfi,
    get_test_dataset, mdi_plus_importance, conditional_predictive_impact,
    # Optimization — HRP, hedging & NCO
    inverse_variance_weights, cluster_variance, quasi_diagonal, recursive_bisection,
    distance_corr, hrp, pca_weights,
    get_optimal_portfolio_weights, get_optimal_portfolio_weights_nco,
    # Cluster — ONC & silhouette
    covariance_to_correlation, silhouette_samples, cluster_k_means_base,
    cluster_k_means_top, make_new_outputs, random_covariance_sub,
    random_block_covariance, random_block_correlation,
    # Backtest.statistics
    sharpe_ratio, bet_timing, calculate_holding_period,
    calculate_hhi, calculate_hhi_concentration, compute_drawdowns_time_under_water,
    conditional_expected_drawdown, sharpe_difference_test,
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
    # Backtest — bet sizing
    probability_bet_size, average_bet_sizes, strategy_bet_sizing, mp_avg_active_signals,
    avg_active_signals, discrete_signal, generate_signal, bet_size_sigmoid,
    target_position, inverse_price, limit_price, compute_sigmoid_width,
    # Backtest — multiple-testing Sharpe haircuts & LPLZ HAC inference
    sharpe_ratio_p_values, holm_adjusted_p_values,
    benjamini_hochberg_yekutieli_adjusted_p_values, haircut_sharpe_ratios,
    sharpe_ratio_influence_function, newey_west_long_run_variance,
    newey_west_automatic_lag, lplz_sharpe_inference,
    # Backtest — closed-form OU trading rules
    theta_from_half_life, hit_upper_probability, mean_exit_time, ou_rule_metrics,
    optimal_ou_trading_rule, fit_ornstein_uhlenbeck,
    # Validation — cross-validators, scoring & tuning
    KFoldCV, PurgedKFoldCV, CombinatorialPurgedCV, WalkForwardCV,
    cv_split, backtest_paths, get_n_splits, cross_val_score,
    grid_search_cv, random_search_cv,
    # Validation — path-level Bagged / Adaptive CPCV
    moving_block_bootstrap_indices, bagged_probability_of_backtest_overfitting,
    estimate_volatility_regimes, adaptive_probability_of_backtest_overfitting,
    # Ensemble — bagging accuracy
    bagging_classifier_accuracy, fit_bagging, bagging_evaluate_schemes,
    calculate_bootstrap_accuracy,
    # Pde — equations & Deep-BSDE solver
    HJBLQ, BlackScholesBarenblatt, PricingDefaultRisk, PricingDiffRate,
    pde_sample, pde_driver, pde_hamiltonian, pde_terminal, pde_sigma, solve_deep_bsde,
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
