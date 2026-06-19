# RiskLabAI — Python ↔ Julia parity ledger

Tracks the public API correspondence between `RiskLabAI.py` and `RiskLabAI.jl`
as the Julia package is reconstructed (Phase 2). The Julia side adopts
Julia-convention `snake_case`; the table is the source of truth for matching
names. Built up one submodule per PR.

## Conventions

- Python: `snake_case` functions, `CapWords` classes, `UPPER_SNAKE` constants.
- Julia: `snake_case` functions, `CapWords` types, `UPPER_SNAKE` constants,
  submodules mirroring the Python sub-packages (`Utils`, `Data`, `Backtest`,
  `Features`, `Optimization`, `HPC`).
- "Deliberate divergence" rows note where the two intentionally differ (e.g.
  Python-only or Julia-only methods).

## Utils  (PR 1 — wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| EWMA | `utils.ewma` | `RiskLabAI.Utils.ewma` / `RiskLabAI.ewma` | identical denominator `1 + (1-α) + (1-α)² + …`; numeric parity asserted |
| Struct-field inheritance | — (n/a in Python) | `RiskLabAI.Utils.@field_inherit` | Julia-only; underpins the bar-type hierarchy |
| Column / stat constants | `utils.constants.*` | `RiskLabAI.Utils.*` | ASCII identifiers; `CUMULATIVE_θ`→`CUMULATIVE_THETA` |

Resolved duplicate: `Utils/EWMA_merge.jl` (`calculateEWMA`) used the wrong
weight exponent `(1-α)^i` (skips the first-order term); removed in favour of
the correct `Utils/ewma.jl`.

## Data.Structures (bars)  — PR 2 (standard bars wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Bar type taxonomy | `data.structures` class hierarchy | `Data` + `types.jl` parametric `Metric` dispatch (`Dollar`/`Volume`/`Tick`) | jl keeps the parametric design |
| Abstract base | `AbstractBars` | `RiskLabAI.Data.AbstractBars` | fields/methods → snake_case |
| Standard bars | `StandardBars` (`construct_bars_from_data`) | `RiskLabAI.Data.StandardBars{T}` (`construct_bars_from_data`) | **wired**; numeric parity asserted against `test_standard_bars.py` |
| `update_base_fields` / `tick_rule` / `construct_next_bar` / `bar_construction_condition` | snake_case | snake_case | renamed from camelCase |
| Time bars | `TimeBars` (`construct_bars_from_data`) | `RiskLabAI.Data.TimeBars` | **wired (PR 3)**; logic rewritten to match Python (emit at bucket boundary, then add the new tick); numeric parity asserted against `test_time_bars.py` |
| Imbalance bars | `ExpectedImbalanceBars`, `FixedImbalanceBars` | `RiskLabAI.Data.{Expected,Fixed}ImbalanceBars{T}` | **wired (PR 4)**; bar-count/tick parity asserted vs `test_imbalance_bars.py`. Fixed bugs: strict `>`→`≥`, `trunc`→`Int` EWMA window, post-bar `warm_up`, dropped `ProfileView`/`TimerOutputs` deps |
| Run bars | `ExpectedRunBars`, `FixedRunBars` | `RiskLabAI.Data.{Expected,Fixed}RunBars{T}` | **wired (PR 5)**; bar-count/tick parity asserted vs `test_run_bars.py` (Fixed→1 bar of 7 ticks). Threshold `E[T]·max(P_buy·E[b_buy], (1-P_buy)·E[b_sell])`; same bug fixes as imbalance |

**`Data.Structures` is now complete** — standard, time, imbalance, and run bars
are all wired with Python-parity tests.

## Data.Differentiation  — PR 6 (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Expanding-window weights | `calculate_weights_std` | `Data.calculate_weights_std` | exact parity (reversed, `w₀` last) |
| FFD weights | `calculate_weights_ffd` | `Data.calculate_weights_ffd` | exact parity |
| Standard frac-diff | `fractional_difference_std` | `Data.fractional_difference_std` | exact parity; `NaN` warm-up before `skip` |
| Fixed-width frac-diff | `fractional_difference_fixed` / `_single` | `Data.fractional_difference_fixed` | exact parity; `NaN` warm-up before `width` |
| Optimal `d` (ADF scan) | `find_optimal_ffd_simple` | `Data.find_optimal_ffd` | ADF via `HypothesisTests.ADFTest`; **behavioural** parity only (ADF impl ≠ statsmodels). Returns columnar `NamedTuple` |
| Stationary log-price | `fractionally_differentiated_log_price` | `Data.fractionally_differentiated_log_price` | same ADF caveat |

**Deliberate divergence:** the Julia frac-diff functions operate on a single
`AbstractVector` (callers `map` over columns) rather than on a `DataFrame`; and
Python's `plot_weights` (matplotlib) is omitted (no plotting dependency).
`HypothesisTests` added to `[deps]` for the ADF-based finders.

## Data.Weights  — PR 7 (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Label concurrency | `expand_label_for_meta_labeling` | `Data.expand_label_for_meta_labeling` | events as `(event_start, event_end)` vectors; returns `(index, concurrency)` |
| Average uniqueness | `calculate_average_uniqueness` | `Data.calculate_average_uniqueness` | T×N indicator matrix → per-event uniqueness; exact parity |
| Return-attribution weights | `sample_weight_absolute_return_meta_labeling` | `Data.sample_weight_absolute_return_meta_labeling` | `Σ|r_t|/c_t`, normalised to N; NaN first-return skipped (matches pandas `.sum`) |
| Time decay | `calculate_time_decay` | `Data.calculate_time_decay` | linear decay; oldest→`clf_last_weight ∈ [0,1]`; exact parity |

**Deliberate divergence:** Python carries event start times in the pandas
Series *index*; the Julia port passes `event_start`/`event_end` (and the price
index) as parallel sorted vectors. `calculate_time_decay` assumes the weights
are already in chronological order (Python sorts by index). All four reference
outputs are asserted against the Python implementation.

## Data.Denoise  — PR 9 (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Marcenko–Pastur PDF | `marcenko_pastur_pdf` | `Data.marcenko_pastur_pdf` | closed form; exact parity (returns `(grid, pdf)`) |
| PCA (desc) | `pca` | `Data.pca` | eigenvalues exact; eigenvectors up to sign |
| cov ↔ corr | `cov_to_corr` / `corr_to_cov` | `Data.cov_to_corr` / `Data.corr_to_cov` | exact parity |
| Denoised correlation | `denoised_corr` | `Data.denoised_corr` | exact parity (sign-invariant reconstruction) |
| Optimal portfolio | `optimal_portfolio` | `Data.optimal_portfolio` | GMV / mean-variance; exact parity |
| MP fit / denoise cov | `find_max_eval`, `denoise_cov`, `optimal_portfolio_denoised` | `Data.{find_max_eval,denoise_cov,optimal_portfolio_denoised}` | **behavioural** parity: Gaussian KDE + golden-section minimiser replace scikit-learn KDE + SciPy `minimize` |

**Deliberate divergence:** the KDE-fit step uses a hand-written Gaussian KDE and
a golden-section search instead of scikit-learn/SciPy, so `find_max_eval` /
`denoise_cov` are validated behaviourally (symmetric output, variances
preserved), not bit-for-bit. No new dependencies were added (the legacy file's
`KernelDensity`/`Optim`/`MultivariateStats`/`BlockArrays` usage is dropped).

## Data.Labeling  — PR 10 (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Symmetric CUSUM | `symmetric_cusum_filter` / `cusum_filter_events_dynamic_threshold` | `Data.symmetric_cusum_filter` / `Data.cusum_filter_events_dynamic_threshold` | exact parity |
| Daily volatility | `daily_volatility_with_log_returns` | `Data.daily_volatility_with_log_returns` | exact parity incl. pandas' debiased EWM std (first value `NaN`) |
| Vertical barrier | `vertical_barrier` | `Data.vertical_barrier` | exact parity; returns `(event, barrier)` |
| Triple barrier | `triple_barrier` | `Data.triple_barrier` | exact first-touch parity (vertical/PT/SL) |
| Meta-events | `meta_events` | `Data.meta_events` | exact parity; serial (Python's multiprocessing is an impl detail) |
| Meta-labeling | `meta_labeling` | `Data.meta_labeling` | exact parity (return + label) |
| OLS t-value | `calculate_t_value_linear_regression` | `Data.calculate_t_value_linear_regression` | closed-form OLS; matches scipy `linregress` |
| Trend scanning | `find_trend_using_trend_scanning` | `Data.find_trend_using_trend_scanning` | exact parity |

**Deliberate divergence:** the price series is passed as parallel
`(close_index, close)` vectors and event tables as `DataFrame`s (vs pandas
time-indexed Series); `target`/`vertical_barriers`/`side` are passed as
`event → value` dicts. No GLM/TimeSeries dependency (the legacy files used
`GLM` + `TimeSeries`); the t-value uses closed-form OLS.

## Data.Distance  — PR 11 (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Variation of information | `calculate_variation_of_information` / `_extended` | `Data.calculate_variation_of_information` / `_extended` | exact parity; histogram + MI/entropy replicated |
| Optimal bins | `calculate_number_of_bins` | `Data.calculate_number_of_bins` | closed form; exact |
| Mutual information | `calculate_mutual_information` | `Data.calculate_mutual_information` | exact parity (optional normalisation) |
| Angular distance | `calculate_distance` | `Data.calculate_distance` | `"angular"` / `"absolute_angular"`; exact |
| KL divergence / cross-entropy | `calculate_kullback_leibler_divergence` / `calculate_cross_entropy` | `Data.calculate_kullback_leibler_divergence` / `Data.calculate_cross_entropy` | exact parity |

**Deliberate divergence:** the 2-D histogram binning replicates
`numpy.histogram2d` and the MI/entropy formulas replicate scikit-learn's
`mutual_info_score` + SciPy's `entropy` (natural log) — so no `StatsBase`/sklearn
dependency is needed.

**The `Data` sub-package is now fully mirrored in Julia** — Structures (bars),
Differentiation, Weights, Denoise, Labeling, and Distance, each with
Python-parity tests. Next: `Backtest`, `Features`, `Optimization`, `HPC`.

Information-driven base: `ewma_expected_imbalance`, `imbalance_at_tick` (metric
dispatch), dynamic threshold `E[T]·|E[b]|` — shared by imbalance and run bars.

Bar row layout (both languages): `[date_time, idx, open, high, low, close,
volume, buy_volume, sell_volume, ticks, dollar, threshold]`.

## Backtest.statistics  — PR (wired)

First slice of the `backtest` sub-package: the backtest statistics.

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Sharpe ratio | `sharpe_ratio` | `Backtest.sharpe_ratio` | exact; population std (`numpy` `ddof=0`); `0.0` on zero dispersion |
| Bet timing | `bet_timing` | `Backtest.bet_timing` | exact; closes/flips + final timestamp |
| Holding period | `calculate_holding_period` | `Backtest.calculate_holding_period` | exact; average-entry-time pairing; weighted mean (`NaN` if no close) |
| HHI | `calculate_hhi` | `Backtest.calculate_hhi` | exact; `NaN` for ≤2 obs or zero sum |
| HHI concentration | `calculate_hhi_concentration` | `Backtest.calculate_hhi_concentration` | exact; positive / negative / monthly-count HHI |
| Drawdowns / TuW | `compute_drawdowns_time_under_water` | `Backtest.compute_drawdowns_time_under_water` | exact; `$`/fractional drawdown; TuW in 365.25-day years |

**Deliberate divergence:** series are passed as parallel `(index, values)`
vectors (timestamps `DateTime`) rather than pandas time-indexed Series; results
are `DataFrame`s / `NamedTuple`s. No `numba`/`TimeSeries`/`DayCounts` dependency
(the legacy `Backtests/` files used `TimeSeries` + `DayCounts`). Remaining
`backtest` modules (probabilistic/deflated Sharpe, strategy risk, PBO, synthetic
backtesting, bet sizing) follow in subsequent PRs.

## Backtest — PSR & test-set overfitting  — PR (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Probabilistic Sharpe ratio | `probabilistic_sharpe_ratio` | `Backtest.probabilistic_sharpe_ratio` | exact; `Φ(Z)` or `Z`; denom ≤ 0 → `0.0`/`-Inf` |
| Benchmark (E[max]) SR | `benchmark_sharpe_ratio` | `Backtest.benchmark_sharpe_ratio` | exact; full-precision `eulergamma`; population std |
| Expected max SR | `expected_max_sharpe_ratio` | `Backtest.expected_max_sharpe_ratio` | exact; truncated Euler constant `0.5772156649` (as in Python) |
| SR Z-statistic | `estimated_sharpe_ratio_z_statistics` | `Backtest.estimated_sharpe_ratio_z_statistics` | exact; `NaN` on non-positive denominator |
| Type-1 error | `strategy_type1_error_probability` | `Backtest.strategy_type1_error_probability` | exact; family-wise `1-(1-Φ(-z))^k` |
| θ for type-2 | `theta_for_type2_error` | `Backtest.theta_for_type2_error` | exact; `NaN` on non-positive denominator |
| Type-2 error | `strategy_type2_error_probability` | `Backtest.strategy_type2_error_probability` | exact |
| Max-SR Monte Carlo | `generate_max_sharpe_ratios` / `mean_std_error` | `Backtest.generate_max_sharpe_ratios` / `mean_std_error` | behavioural; stochastic, optional `rng` keyword for reproducibility |

**Deliberate divergence:** the normal CDF / quantile come from `Distributions`
(matching SciPy's `norm.cdf` / `norm.ppf`); no SciPy/`numba` dependency. The
Monte-Carlo helpers take an optional `rng` keyword (Julia idiom) in place of
NumPy's implicit default generator.

## Backtest — strategy risk  — PR (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Binomial SR trials | `sharpe_ratio_trials` | `Backtest.sharpe_ratio_trials` | behavioural; stochastic, optional `rng` |
| Binomial variance | `target_sharpe_ratio_symbolic` | `Backtest.target_sharpe_ratio_symbolic` | closed form `p(1-p)(d-u)²` of `(p,u,d)` (Python returns a SymPy expr) |
| Implied precision | `implied_precision` | `Backtest.implied_precision` | exact; `NaN` when discriminant < 0 |
| Implied frequency | `bin_frequency` | `Backtest.bin_frequency` | exact; `Inf` for degenerate precision / zero denom |
| Binomial Sharpe ratio | `binomial_sharpe_ratio` | `Backtest.binomial_sharpe_ratio` | exact; signed `Inf` on zero dispersion |
| Mixture of Gaussians | `mix_gaussians` | `Backtest.mix_gaussians` | behavioural; stochastic, optional `rng` |
| Failure probability | `failure_probability` | `Backtest.failure_probability` | exact; `0.0` if no winners/losers, `1.0` if target unachievable |
| Strategy risk | `calculate_strategy_risk` | `Backtest.calculate_strategy_risk` | behavioural; stochastic, optional `rng` |

**Deliberate divergence:** `target_sharpe_ratio_symbolic` returns the closed-form
variance value (no SymPy); the normal CDF comes from `Distributions` (no SciPy);
the Monte-Carlo helpers take an optional `rng` keyword in place of NumPy's
implicit generator.

_(further submodules appended as they are wired)_
