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
| Label concurrency | `expand_label_for_meta_labeling` | `Data.expand_label_for_meta_labeling` | counts all overlapping events, including those outside the molecule; inclusive endpoints; returns `(index, concurrency)` |
| Average uniqueness | `calculate_average_uniqueness` | `Data.calculate_average_uniqueness` | T×N indicator matrix → per-event uniqueness; exact parity |
| Sequential bootstrap | `sequential_bootstrap` | `Data.sequential_bootstrap` | samples with replacement using average added uniqueness; one-based Julia event positions; accepts a seed or `AbstractRNG` |
| Return-attribution weights | `sample_weight_absolute_return_meta_labeling` | `Data.sample_weight_absolute_return_meta_labeling` | `Σ|r_t|/c_t`, normalised to N; NaN first-return skipped (matches pandas `.sum`) |
| Time decay | `calculate_time_decay` | `Data.calculate_time_decay` | linear decay; oldest→`clf_last_weight ∈ [0,1]`; exact parity |

**Deliberate divergence:** Python carries event start times in the pandas
Series *index*; the Julia port passes `event_start`/`event_end` (and the price
index) as parallel sorted vectors. `calculate_time_decay` assumes the weights
are already in chronological order (Python sorts by index). Shared Python 3.2.0 fixtures cover 44 concurrency/return-weight cases, including outside-molecule overlaps and zero-duration events. Four sequential-bootstrap fixtures supply identical uniforms and convert Python's zero-based positions to Julia's one-based positions; equal seeds alone do not establish cross-language equivalence. The fixture generator checks the Python results against independent event enumeration and the added-uniqueness formula. Domain checks cover invalid indicators, inactive events, invalid draw counts/seeds and zero-draw RNG preservation.

## Data.Denoise  — PR 9 (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Marcenko–Pastur PDF | `marcenko_pastur_pdf` | `Data.marcenko_pastur_pdf` | closed form; exact parity (returns `(grid, pdf)`) |
| PCA (desc) | `pca` | `Data.pca` | eigenvalues exact; eigenvectors up to sign |
| cov ↔ corr | `cov_to_corr` / `corr_to_cov` | `Data.cov_to_corr` / `Data.corr_to_cov` | exact parity |
| Denoised correlation | `denoised_corr` | `Data.denoised_corr` | exact parity (sign-invariant reconstruction) |
| Optimal portfolio | `optimal_portfolio` | `Data.optimal_portfolio` | GMV / mean-variance; exact parity |
| MP fit / denoise cov | `find_max_eval`, `denoise_cov`, `optimal_portfolio_denoised` | `Data.{find_max_eval,denoise_cov,optimal_portfolio_denoised}` | Bounded shared-fixture parity for a native Gaussian KDE and scalar local fit; numerical convergence/fallback differences remain possible |

The native local fit uses the public initial variance, bounds, More–Thuente
interpolation and stopping settings. The public `find_max_eval` path supplies
the analytic derivative of the same MP-density loss, including its moving grid.
Forward finite differences remain the internal fallback when no derivative is
supplied. Extrapolated trial intervals are retained before applying the absolute
step bound. The 243 original fit fixtures and three additional line-search
regressions pass; twenty-four covariance/portfolio fixtures retain independent
spectral checks.

Near a stationary point, floating-point changes can alter convergence status.
The unchanged variance-one fallback can then change the signal count and the
resulting portfolio. Chapter 10 exposed such a case even when the two latent
optimizer variances agreed within `4e-12`. Its technical hold is closed following
the approved corrected normal-import and final-location validation in the tested
Windows environment. The original failure remains historical evidence; all
documented operating-system, numerical-backend, runtime, cross-platform,
historical-baseline and release limitations remain unchanged. Covariance
reconstruction conditional on a recorded cutoff is distinct from unconditional
optimizer parity; neither bitwise identity nor a global minimum is promised.

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

## Backtest — PBO & synthetic backtesting  — PR (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| CSCV performance eval | `performance_evaluation` | `Backtest.performance_evaluation` | exact; 1-based ordinal ranks (`argsort∘argsort`), logit of relative rank |
| Probability of backtest overfitting | `probability_of_backtest_overfitting` | `Backtest.probability_of_backtest_overfitting` | exact; `C(S,S/2)` splits, `numpy.array_split` row partitioning, default population-std Sharpe |
| Synthetic backtesting | `synthetic_back_testing` | `Backtest.synthetic_back_testing` | behavioural; OU paths + PT/SL grid; stochastic, optional `rng` |

**Deliberate divergence:** combinatorial splits use `Combinatorics.combinations`
(same lexicographic order as Python's `itertools.combinations`) and the CSCV runs
serially (Python's `joblib` parallelism is an implementation detail). The
synthetic OU noise uses Julia's `randn`/`rng` rather than Python's `random.gauss`,
so that helper is reproducible but not bit-identical across languages.

## Backtest — bet sizing  — PR (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Probability bet size | `probability_bet_size` | `Backtest.probability_bet_size` | exact; `side·(2Φ(p)-1)` |
| Concurrent average | `average_bet_sizes` | `Backtest.average_bet_sizes` | exact |
| Strategy bet sizing | `strategy_bet_sizing` | `Backtest.strategy_bet_sizing` | exact; aligned parallel-vector inputs |
| Active-signal averaging | `avg_active_signals` / `mp_avg_active_signals` | `Backtest.avg_active_signals` / `mp_avg_active_signals` | exact; prefix-sum + binary search; `missing` end never closes |
| Discretise signal | `discrete_signal` | `Backtest.discrete_signal` | exact; round-half-to-even, cap ±1 |
| Generate signal | `generate_signal` | `Backtest.generate_signal` | exact; OvR t-value → side·size → average → discretise |
| Sigmoid bet size | `bet_size_sigmoid` | `Backtest.bet_size_sigmoid` | exact |
| Target position | `target_position` | `Backtest.target_position` | exact; truncates toward zero |
| Inverse price | `inverse_price` | `Backtest.inverse_price` | exact |
| Limit price | `limit_price` | `Backtest.limit_price` | exact |
| Sigmoid width | `compute_sigmoid_width` | `Backtest.compute_sigmoid_width` | exact; `Inf` when `m ∈ {0, ±1}` |

**Deliberate divergence:** the Julia API uses the 2.0.0 snake_case canon directly
with **no deprecated camelCase aliases**. Series/DataFrame inputs become parallel
sorted vectors; the active-signal helpers return `(time_points, values)` and run
serially (Python's `mp_pandas_obj` parallelism is an implementation detail).

**The `backtest` core is now mirrored in Julia** — statistics, probabilistic /
expected-max Sharpe, test-set overfitting, strategy risk, probability of backtest
overfitting, synthetic backtesting, and bet sizing, each with Python-parity
tests. (The `backtest.validation` cross-validation subpackage and the
`backtest_overfitting_simulation` benchmark harness were scoped out of this
effort.)

## Features — entropy  — PR (wired)

First slice of the `features` sub-package: entropy estimators.

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Shannon entropy | `shannon_entropy` | `Features.shannon_entropy` | exact; bits (`log2`) |
| Probability mass function | `probability_mass_function` | `Features.probability_mass_function` | exact; n-gram PMF as a `Dict` |
| Plug-in entropy | `plug_in_entropy_estimator` | `Features.plug_in_entropy_estimator` | exact; PMF entropy / word length |
| Lempel–Ziv entropy | `lempel_ziv_entropy` | `Features.lempel_ziv_entropy` | exact; distinct-substring count / length |
| Longest match | `longest_match_length` | `Features.longest_match_length` | exact; `(length+1, substring)` |
| Kontoyiannis entropy | `kontoyiannis_entropy` | `Features.kontoyiannis_entropy` | exact; **averaged** `Σ log2(nᵢ)/Lᵢ` (de Prado's formula / current source) |

**Note:** the first three numeric `features` areas (entropy, microstructural,
structural breaks) are being mirrored; classifier-driven feature importance is
deferred pending a Julia ML-backend decision.

## Features — microstructural  — PR (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Corwin–Schultz β | `beta_estimates` | `Features.beta_estimates` | exact incl. rolling-2-then-window NaN warm-up |
| Corwin–Schultz γ | `gamma_estimates` | `Features.gamma_estimates` | exact; 2-day high-max / low-min |
| Corwin–Schultz α | `alpha_estimates` | `Features.alpha_estimates` | exact; floored at 0 |
| Corwin–Schultz spread | `corwin_schultz_estimator` | `Features.corwin_schultz_estimator` | exact; `2(eᵅ-1)/(1+eᵅ)` |
| Bekker–Parkinson σ | `sigma_estimates` | `Features.sigma_estimates` | exact |
| Bekker–Parkinson volatility | `bekker_parkinson_volatility_estimates` | `Features.bekker_parkinson_volatility_estimates` | exact |

**Deliberate divergence:** pandas Series become `Vector`s; a small `_rolling`
helper reproduces `rolling(window=w)` exactly (NaN for the first `w-1` points and
for any window containing a NaN), so the warm-up NaN pattern matches Python.

## Features — structural breaks  — PR (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Lagged frame | `lag_dataframe` | `Features.lag_dataframe` | exact; lag columns with NaN warm-up |
| ADF design | `prepare_data` | `Features.prepare_data` | exact diff/shift/dropna alignment; returns `(y, x, index)` |
| OLS β | `compute_beta` | `Features.compute_beta` | exact; NaN-filled on singular design |
| Expanding-window ADF | `get_expanding_window_adf` | `Features.get_expanding_window_adf` | exact t-statistic path |
| Backward Supremum ADF | `get_bsadf_statistic` | `Features.get_bsadf_statistic` | exact supremum-ADF (bubble origination) |

**Deliberate divergence:** pandas Series/DataFrames become `Vector`/`Matrix`; OLS
uses `LinearAlgebra` (LAPACK). With this slice the three numeric `features`
areas (entropy, microstructural, structural breaks) are mirrored; classifier-
driven `feature_importance` remains deferred pending a Julia ML-backend decision.

**CI:** the macOS test legs now run only on pushes to `main`/tags (PRs run
Julia 1.10 + 1 on Ubuntu and Windows), to avoid the slow macOS PR queue while
keeping full cross-platform coverage on merge.

## Optimization — HRP & hedging  — PR (wired)

First slice of the `optimization` sub-package: the pure-numeric pieces.

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Inverse-variance weights | `inverse_variance_weights` | `Optimization.inverse_variance_weights` | exact |
| Cluster variance | `cluster_variance` | `Optimization.cluster_variance` | exact; IVW-weighted `wᵀCw` |
| Quasi-diagonal order | `quasi_diagonal` | `Optimization.quasi_diagonal` | exact; SciPy-format linkage in, 1-based items out |
| Recursive bisection | `recursive_bisection` | `Optimization.recursive_bisection` | exact HRP weights given the sorted order |
| Correlation distance | `distance_corr` | `Optimization.distance_corr` | exact `√((1-ρ)/2)` |
| PCA hedging weights | `pca_weights` | `Optimization.pca_weights` | sign-free invariant `wᵀCw = risk_target²·Σρ` (eigenvectors are sign-ambiguous) |

**Deliberate divergence / deferred:** asset labels become 1-based integer
indices; matrices replace DataFrames. The top-level `hrp(cov, corr)` wrapper and
Nested Clustered Optimisation (`nco`) are now **wired** (see below); sklearn
`hyper_parameter_tuning` awaits the ML-backend decision.

### hrp() & NCO — wired (follow-up to the cluster port)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| HRP wrapper | `hrp` | `Optimization.hrp` | **behavioural**; single-linkage via `Clustering.hclust` (leaf order not bit-identical to SciPy), weights long-only and sum to one. Returns original-order weights |
| Markowitz GMV/MVO | `get_optimal_portfolio_weights` | `Optimization.get_optimal_portfolio_weights` | **exact** `w = C⁻¹μ / (1ᵀC⁻¹μ)`; `mu=nothing` → GMV |
| Nested Clustered Optimisation | `get_optimal_portfolio_weights_nco` | `Optimization.get_optimal_portfolio_weights_nco` | **behavioural**; clusters via `Cluster.cluster_k_means_base`, intra/inter optimisation recombined; weights sum to one |

## Cluster — ONC & silhouette  — PR (wired)

Port of the `cluster.clustering` sub-package: Optimized Nested Clustering and its
supporting pieces, built on `Clustering.jl` (new dependency) for k-means.

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Covariance → correlation | `covariance_to_correlation` | `Cluster.covariance_to_correlation` | exact; delegates to `Data.cov_to_corr` |
| Silhouette scores | `silhouette_samples` (sklearn) | `Cluster.silhouette_samples` | **exact**; native, matches sklearn `metric="precomputed"` (verified to 1e-9); singleton cluster → 0 |
| K-means base step | `cluster_k_means_base` | `Cluster.cluster_k_means_base` | **behavioural**; `Clustering.kmeans` vs sklearn KMeans. Best clustering by silhouette t-stat (population std). `random_state` reseeds the global RNG |
| ONC | `cluster_k_means_top` | `Cluster.cluster_k_means_top` | **behavioural**; recursion re-clusters below-average-t-stat clusters (cluster t-stat uses sample std, ddof 1) |
| Merge outputs | `make_new_outputs` | `Cluster.make_new_outputs` | recombines two cluster dicts and recomputes silhouette |
| Random sub-covariance | `random_covariance_sub` | `Cluster.random_covariance_sub` | **behavioural** (stochastic) |
| Random block covariance | `random_block_covariance` | `Cluster.random_block_covariance` | **behavioural**; SciPy `block_diag` → native block assembly |
| Random block correlation | `random_block_correlation` | `Cluster.random_block_correlation` | **behavioural** |

**Deliberate divergence:** pandas labels become 1-based integer item indices and
correlation/covariance are `Matrix`es; clusters are `Dict(label => Vector{Int})`.
K-means is stochastic and not bit-identical across backends, so those pieces are
**behavioural** ports — tests assert structural properties (valid partition over
all items, shapes, symmetry, unit diagonal) rather than exact values.
`silhouette_samples` is the deterministic anchor and is parity-tested exactly.

With the cluster port landed, the deferred `hrp()` top-level wrapper and Nested
Clustered Optimisation (`nco`) can follow.

## Validation — cross-validators  — PR (wired)

Port of the `backtest.validation` sub-package: train/test indices, chronological
paths and estimator-driven `backtest_predictions` through explicit caller
callbacks. Supplied-feature adaptive boundaries are included.

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Standard K-Fold | `KFold` | `Validation.KFoldCV` | contiguous folds, optional shuffle; `cv_split(cv, n)` |
| Purged K-Fold + embargo | `PurgedKFold` | `Validation.PurgedKFoldCV` | **exact** purge + embargo index logic (verified vs pandas reference); `cv_split(cv)` |
| Combinatorial Purged CV | `CombinatorialPurged` | `Validation.CombinatorialPurgedCV` | **exact**; `C(n,k)` splits, `backtest_paths` assembles the `k·C/n` paths |
| Adaptive combinatorial CV | `AdaptiveCombinatorialPurged` | `Validation.AdaptiveCombinatorialPurgedCV` | supplied feature, linear quantiles and public fine-split boundary adjustments |
| Walk-Forward | `WalkForward` | `Validation.WalkForwardCV` | **exact**; growing train window, `gap` / `max_train_size` |
| Number of splits | `get_n_splits` | `Validation.get_n_splits` | exact |

**Deliberate divergence:** Python nests this under `backtest.validation`; Julia
exposes the top-level `RiskLabAI.Validation`. The sklearn-style `CrossValidator`
ABC + factory + controller scaffolding is replaced by plain structs + dispatch.
`times` (a pandas Series start→end) becomes parallel `event_starts`/`event_ends`
vectors; all indices are 1-based.

## Data.SyntheticData  — PR (wired)

Port of the `data.synthetic_data` sub-package: covariance generators and the
Heston–Merton regime-switching price simulator.

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Block correlation | `form_block_matrix` | `Data.form_block_matrix` | **exact** |
| Drift-burst profile | `drift_volatility_burst` | `Data.drift_volatility_burst` | **exact** (midpoint clamp + NaN fill) |
| Heston–Merton step | `compute_log_returns` | `Data.compute_log_returns` | **exact** (numba → plain loop) |
| Parameter alignment | `align_params_length` | `Data.align_params_length` | **exact** |
| Random covariance | `random_cov` | `Data.random_cov` | **behavioural** (stochastic) |
| True mean/cov | `form_true_matrix` | `Data.form_true_matrix` | **behavioural** |
| Sample mean/cov | `simulates_cov_mu` | `Data.simulates_cov_mu` | `shrink=false` exact (`np.cov`); `shrink=true` Ledoit–Wolf deferred |
| Heston–Merton returns | `heston_merton_log_returns` | `Data.heston_merton_log_returns` | **behavioural** |
| Regime price path | `generate_prices_from_regimes` | `Data.generate_prices_from_regimes` | **behavioural**; native Markov simulate (replaces quantecon) |
| Many paths | `parallel_generate_prices` | `Data.parallel_generate_prices` | **behavioural**; serial (no joblib) |

**Deliberate divergence:** numba / joblib / quantecon are replaced by native
Julia. `simulates_cov_mu`'s `shrink=true` (Ledoit–Wolf) is deferred to the
covariance-estimation backend; the default exact `shrink=false` is wired. Price
generators return a plain price vector (Python: a business-day-indexed Series).

## Features.feature_importance — orthogonal & weighted-τ  — PR (wired)

Backend-independent pieces of the `features.feature_importance` sub-package.
Caller-estimator MDI/MDA/SFI and clustered variants complement the existing
DecisionTree.jl convenience methods.

| Concept | Python | Julia | Notes |
|---|---|---|---|
| PCA orthogonalisation | `orthogonal_features` | `Features.orthogonal_features` | shared spectra and invariant reconstruction after incomplete/constant-column removal; nullspace count can depend on floating-point rounding at 100% variance |
| Weighted Kendall-τ | `calculate_weighted_tau` | `Features.calculate_weighted_tau` | shared SciPy fixtures and independent exact pair sums; lexicographic ties and NaNs ordered lowest |

**Representation:** DataFrames become matrices. PCA eigenvector rows correspond
to usable columns in their original order. A dataset without a usable column
fails explicitly. Repeated eigenvalues permit different bases of the same
eigenspace; shared tests compare invariant quantities.

### Tree-based importances (DecisionTree.jl backend) — wired

| Concept | Python | Julia | Notes |
|---|---|---|---|
| MDI | `FeatureImportanceMDI` | `Features.feature_importance_mdi` | **behavioural**; per-tree normalised impurity over a bootstrap forest, 0→NaN, normalised |
| MDA | `FeatureImportanceMDA` | `Features.feature_importance_mda` | **behavioural**; shuffled-feature neg-log-loss drop over a shuffled K-Fold |
| SFI | `FeatureImportanceSFI` | `Features.feature_importance_sfi` | **behavioural**; CV score of a single-feature forest (`:log_loss`/`:accuracy`) |
| Test dataset | `get_test_dataset` | `Features.get_test_dataset` | **behavioural**; native generator (no `sklearn.make_classification`) |

**Deliberate divergence:** the tree backend is `DecisionTree.jl` (added
dependency), which is not bit-identical to scikit-learn — these importances are
**behavioural** and validated structurally (informative features rank above
noise). These existing two-argument convenience functions do not expose sample
weights. New caller-estimator overloads preserve training/scoring weights and
do not choose an alternative estimator. Clustered MDI/MDA are plain functions;
the Python controller/factory/strategy scaffolding is unnecessary for these
native calculation contracts.

## Ensemble & cross-validation scoring  — PR (wired)

Port of the `ensemble` sub-package plus the estimator-driven scoring half of
`backtest.validation`.

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Theoretical bagging accuracy | `bagging_classifier_accuracy` | `Ensemble.bagging_classifier_accuracy` | **exact** (binomial survival function) |
| Weighted-bagging evaluation | `BaggingClassifierAccuracy` | `Ensemble.fit_bagging` + `bagging_evaluate_schemes` | **behavioural**; uniform / cᵢ / 1−cᵢ² weighting on the DecisionTree.jl backend |
| Bootstrap accuracy | `calculate_bootstrap_accuracy` | `Ensemble.calculate_bootstrap_accuracy` | **behavioural** |
| Cross-validation score | `…validation.backtest_predictions` (scoring) | `Validation.cross_val_score` | **behavioural**; per-fold RF score (`:accuracy`/`:neg_log_loss`) over any cross-validator |

**Deliberate divergence:** the sklearn `BaggingClassifierAccuracy` class becomes a
functional API; `class_weight="balanced"` and the Matplotlib plotting helper are
dropped. `cross_val_score` retains its existing random-forest scoring contract.
Full path-level `backtest_predictions` now accepts caller-selected estimators
through explicit clone, weighted-fit and prediction callbacks. The random-forest
scorer is not a replacement for another specified estimator or scoring rule.

## Hyper-parameter tuning  — PR (wired)

Port of `optimization.hyper_parameter_tuning`: grid / randomised search over a
random-forest hyper-parameter grid, scored by a purged (or any) cross-validator.

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Grid search | `clf_hyper_fit` (`rnd_search_iter==0`) | `Validation.grid_search_cv` | **behavioural**; exhaustive grid, `cross_val_score`-scored, returns best params + refit forest |
| Randomised search | `clf_hyper_fit` (`rnd_search_iter>0`) | `Validation.random_search_cv` | **behavioural**; `n_iter` sampled configurations |

**Deliberate divergence:** scikit-learn `GridSearchCV`/`RandomizedSearchCV`,
`Pipeline`/`SampleWeightedPipeline`/`MyPipeline` and the `f1` scorer are replaced
by a functional API over the `DecisionTree.jl` random forest; the tunable grid is
`{:n_trees, :n_subfeatures, :max_depth}` and scoring is `:accuracy` /
`:neg_log_loss`.

## Pde — equations (Deep BSDE)  — PR (wired)

Port of `pde.equation`: the forward-SDE sampler and BSDE generator functions for
four financial PDEs (Han, Jentzen & E, 2018). The neural Deep-BSDE solver follows
with a deep-learning backend (Lux.jl).

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Equation base | `Equation` | `Pde.Equation` (abstract) | dispatch over the four concrete types |
| HJB-LQ | `HJBLQ` | `Pde.HJBLQ` | **exact** generators (`pde_driver`/`pde_hamiltonian`/`pde_terminal`) |
| Black–Scholes–Barenblatt | `BlackScholesBarenblatt` | `Pde.BlackScholesBarenblatt` | **exact** |
| Default-risk pricing | `PricingDefaultRisk` | `Pde.PricingDefaultRisk` | **exact** (piecewise-linear hazard) |
| Different-rate pricing | `PricingDiffRate` | `Pde.PricingDiffRate` | **exact** |
| Forward-SDE sample | `Equation.sample` | `Pde.pde_sample` | **behavioural** (Euler–Maruyama; stochastic) |

**Deliberate divergence:** torch tensors → `Matrix`es (`batch × dim`); the method
names are `pde_driver` (`r_u`), `pde_hamiltonian` (`h_z`), `pde_terminal`,
`pde_sigma`.

### Deep-BSDE solver (Lux.jl backend) — wired

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Deep-BSDE solver | `FBSDESolver(..., "DeepBSDE")` | `Pde.solve_deep_bsde` | **behavioural**; per-time-step MLP (`Lux.jl`), trainable `Y_0`/`Z_0`, terminal-MSE loss, Adam (`Optimisers.jl`) + `Zygote.jl` gradients; validated structurally (loss is finite and decreases) |

**Deliberate divergence:** PyTorch → Lux.jl + Zygote + Optimisers (added
dependencies). The bias-free MLP matches the Python `DeepBSDE` subnetwork (the
commented-out batch-norm is omitted). The set-transformer architectures
(`ISAB`/`MAB`/`SAB`/`PMA`/`DeepTimeSetTransformer`) and the `Monte-Carlo`/`DTNN`/
`FBSNN` solver variants are research scaffolding and are not ported.

## additional analytical methods — parity port

The additional analytical methods from `RiskLabAI.py` (the public method specification
rows), ported to parity. Deterministic estimators reproduce the Python reference
values within the recorded tolerance; stochastic / ADF-dependent pieces are
behavioural (validated structurally). Each docstring carries the documented usage guidance + citation + validation summary.

| Concept | Python | Julia | Notes |
|---|---|---|---|
| EDGE spread estimator | `microstructural_features.edge.edge_estimator` | `Features.edge_estimator` | **exact** (parity to full precision on a 12-bar OHLC fixture); NaN on degenerate/short input |
| GSADF / BSADF | `structural_breaks.get_gsadf_statistic` / `get_bsadf_sequence` / `get_sadf_sequence` / `get_bubble_episodes` / `psy_minimum_window` | `Features.get_gsadf_statistic` / `get_bsadf_sequence` / `get_sadf_sequence` / `get_bubble_episodes` / `psy_minimum_window` | **exact** sup-ADF sequences (~5e-10), date-stamping in 1-based positions; `simulate_psy_critical_values` behavioural (random-walk null, Julia RNG) |
| Bias-corrected entropy | `entropy_features.bias_corrected.{miller_madow,grassberger,nsb}_entropy` | `Features.{miller_madow,grassberger,nsb}_entropy` | **exact** (~1e-13; NSB quadrature via SpecialFunctions digamma/loggamma + bisection inversion) |
| MDI+ + CPI | `feature_importance.debiased_importance.{mdi_plus_importance,conditional_predictive_impact}` | `Features.{mdi_plus_importance,conditional_predictive_impact}` | **behavioural** (DecisionTree.jl backend, Julia-RNG knockoffs); informative features rank above noise, CPI assigns the smaller p-value |
| Holm + BHY Sharpe haircuts | `multiple_testing.{haircut_sharpe_ratios,holm_adjusted_p_values,benjamini_hochberg_yekutieli_adjusted_p_values,sharpe_ratio_p_values}` | `Backtest.*` (new `MultipleTesting.jl`) | **exact** (tie-handling gives equal adjusted values) |
| LPLZ HAC Sharpe inference | `sharpe_inference.{lplz_sharpe_inference,sharpe_ratio_influence_function,newey_west_long_run_variance,newey_west_automatic_lag}` | `Backtest.*` (`ProbabilisticSharpeRatio.jl`) | **exact** |
| CED + LW Sharpe-diff test | `robust_statistics.{conditional_expected_drawdown,sharpe_difference_test}` | `Backtest.*` (`BacktestStatistics.jl`) | CED + naive test + LW delta/se/stat **exact**; LW bootstrap p-value behavioural (Julia RNG) |
| Closed-form OU trading rules | `ou_trading_rules.{optimal_ou_trading_rule,ou_rule_metrics,hit_upper_probability,mean_exit_time,fit_ornstein_uhlenbeck,...}` | `Backtest.*` (new `OUTradingRules.jl`) | closed-form metrics + OU fit **exact** (erfi from SpecialFunctions); optimizer is a grid search (replaces SciPy L-BFGS-B), matches the maximized objective on the flat ridge |
| Path-level Bagged CPCV | `validation.path_bagged_cpcv.{bagged_probability_of_backtest_overfitting,moving_block_bootstrap_indices}` | `Validation.*` (new `PathBaggedCPCV.jl`) | **behavioural** (moving-block bootstrap, Julia RNG; builds on the parity-exact CSCV PBO) |
| Path-level Adaptive CPCV | `validation.path_adaptive_cpcv.{adaptive_probability_of_backtest_overfitting,estimate_volatility_regimes}` | `Validation.*` (new `PathAdaptiveCPCV.jl`) | **exact** (fully deterministic; regimes, weighted PBO and selected config match) |
| Leakage-aware HPO | `validation.leakage_aware_hpo.{leakage_aware_hpo,deflated_sharpe_gate}` | `Validation.*` (`HyperParameterTuning.jl`) | `deflated_sharpe_gate` **exact**; `leakage_aware_hpo` purged-CV random search (Optuna TPE/CMA-ES is an optional analogue, not bundled) |
| KSG MI + distance correlation | `distance.codependence.{ksg_mutual_information,distance_correlation}` | `Data.{ksg_mutual_information,distance_correlation}` | distance correlation **exact**; KSG **exact** via the jitter-invariant integer neighbour counts (brute-force O(N²) kNN, Chebyshev) |
| Adaptive Fractional Differencing | `differentiation.adaptive_differentiation.{adaptive_fractional_difference,rescaled_range_hurst,adaptive_differencing_order}` | `Data.*` (`Differentiation.jl`, `WaveletHurst.jl`) | Default db2 wavelet/R/S blend, missing-value alignment, thresholds, orders and ADF probabilities pass shared Python 3.2.0 fixtures; see the current bounded validation below |

**Deliberate divergences (this wave):** `SpecialFunctions` added as a direct
dependency (digamma/loggamma/erfi/erfc). Leakage-aware HPO's optional Optuna sampler
(`optuna`) uses random sampling in this historical implementation. The default
AFD wavelet component is now implemented natively without a new dependency.
The OU optimizer
uses a coarse-to-fine grid search in place of SciPy L-BFGS-B. CSCV-path order is the
lexicographic `Combinatorics.combinations` order (matches `itertools.combinations`).

## additional analytical-method completion — parity port

The three remaining additional analytical methods, closing the Julia parity gap. Each
docstring carries the documented usage guidance + citation + independent validation
summary.

| Concept | Python | Julia | Notes |
|---|---|---|---|
| NERCOME sample-split denoiser | `data.denoise.nercome.nercome_denoised_covariance` | `Data.nercome_denoised_covariance` | **behavioural** (seeded sample-splitting; Julia RNG ≠ NumPy PCG64). Validated structurally (symmetric, PD, reproducible under a seed) and on the documented mechanism (independent validation): lower relative-Frobenius covariance error than MP clipping on a no-gap / non-stationary spectrum, and lower min-variance OOS variance |
| Volatility-robust SADF | `features.structural_breaks.volatility_robust_sadf` | `Features.volatility_robust_sadf` | observed `sadf` / `gsadf` sup-ADF statistics **exact** (~1e-6, deterministic, share the `_psy_sadf_bsadf_sequences` kernel); wild-bootstrap p-values **behavioural** (Rademacher signs from Julia RNG). Reproduces the independent validation size result (plain GSADF empirical size ~0.48 vs vol-robust ~0.05 under a 4x variance break) |
| PELT change-points | `features.structural_breaks.pelt_change_points` (via BSD-2 `ruptures`) | `Features.pelt_change_points` | **exact** change-point indices — clean-room port of `ruptures.Pelt` + `CostNormal` (same candidate grid, admissible-set pruning and Gaussian segment cost), so the two agree on a shared series (mean-shift, pure-variance, and multi-break fixtures) |

**Deliberate divergences (this wave):** NERCOME and the vol-robust wild bootstrap are
seeded-stochastic — reproducible under a given Julia `rng` but not bit-identical to
the NumPy PCG64 reference, as the existing path-level modules already document. PELT
is deterministic and reproduces `ruptures` exactly; the GPL-3 `exuber` reference was
not used (the Python side uses the permissive BSD-2 `ruptures`, and the Julia side is
a clean-room reimplementation of the published Killick et al. 2012 algorithm).

## CausalFactorAnalysis: bounded estimators

`RiskLabAI.CausalFactorAnalysis` provides native treatment-effect arithmetic and the approved bounded estimators below. Inputs are already identified means/probabilities or caller-supplied observations and design probabilities. Numerical checks do not establish graph validity, exchangeability, parallel trends, transportability, exposure-mapping validity or nuisance-model training history.

| Concept | Python 3.2.0 | Julia | Validation |
|---|---|---|---|
| Mean effects and decomposition | `average_treatment_effect`, `treatment_effect_decomposition`, `randomized_mean_difference`, `difference_in_differences` | Same names under `CausalFactorAnalysis`; immutable decomposition types | Shared scalar/vector cases; decomposition identities; finite-domain checks |
| Adjustment arithmetic and IV ratio | `backdoor_adjusted_expectation`, `backdoor_adjusted_average_treatment_effect`, `frontdoor_adjusted_probability`, `linear_instrumental_variable_effect` | Same names under `CausalFactorAnalysis` | Shared probabilities/covariances; independent front-door sum; support and domain failures |
| Bounded outcomes and transport | `manski_ate_bounds`, `transported_ate` | Same names under `CausalFactorAnalysis` | Exhaustive missing-potential-outcome extrema; supported target standardization; zero-target and unsupported-stratum checks |
| Randomized exposure means | `exposure_mean_contrast` | Same name under `CausalFactorAnalysis` | Average over every independent binary randomized assignment in the finite fixture; positivity and omitted-category checks |
| One-step policy evaluation and selection | `evaluate_policy`, `select_policy` | Same names under `CausalFactorAnalysis` | Shared IPW/DR scores; independent finite-action sums; held-out reward perturbation; ties, invalid probabilities, zero weights and invalid chronological splits |

The 168 shared cases in `test/fixtures/causal_estimators_python_320.toml` cover 13 functions. Deterministic floating comparisons use `rtol=1e-10`, `atol=1e-12`. Julia returns named tuples for Python dictionary results. Actions, exposure columns, split rows and selected-policy indices are one-based in Julia. The two decomposition types retain the Python field names. Exposure contrasts do not report independent-unit standard errors, and policy scores use neither clipping nor self-normalization. Graph identification, protocol diagnostics and the separate OLS sensitivity supplier remain outside this completed subset.

## Constrained and ellipsoid-robust portfolios

`Optimization.constrained_minimum_variance` and `Optimization.robust_mean_variance`
match the approved Python 3.2.0 convex problems: fully invested, nonnegative,
capped weights with an optional nominal return floor. Covariance must be PSD
within the stated eigensolver tolerance and is never repaired. The robust mean
set is the caller-supplied ellipsoid `mu + L*u`, `norm(u) <= radius`; covariance
is fixed. JuMP/Clarabel status and primal feasibility are checked before returning
weights, variance, nominal/worst means and the objective in a named tuple.

The 21 shared fixtures independently minimize each two-asset problem by scalar
convex calculus. Objective comparisons retain `rtol=1e-8` and scale-aware
`atol=1e-10`; weight comparisons use componentwise `rtol=atol=1e-6`, matching
the Python fixture check. A conic solver's objective tolerance does not imply
equal relative accuracy for weights or each derived statistic: the Python
reference itself differs from the analytic weights by up to `8.63e-7` and from
the corresponding variance by `1.87e-8`. Cross-solver variance and mean bounds
therefore propagate the declared weight error through their quadratic, linear
and norm formulas. Returned diagnostics are also recomputed from returned weights
at `rtol=1e-12`. The optimal-objective and analytical-weight requirements are
unchanged. Nonunique optima are checked for objective and feasibility, without
requiring arbitrary equality of optimal weights. Additional cases cover singular
and zero covariance, very small/large scales, infeasibility and invalid domains.

Fourteen additional public-book cases use sixteen industries, fixed block
resamples, three risk-aversion values and four uncertainty radii, including a
zero loading and a nominal return floor. The conic representation lists the
budget, nonnegativity and cap constraints in the public reference order. The
second-order cone already implies a nonnegative norm epigraph, so no redundant
scalar bound is added. These equivalent representation choices preserve the
objective, feasible set, solver settings and tolerances. All fourteen cases
match the public weights and diagnostics at `rtol=1e-9`, `atol=1e-11`;
the maximum observed weight difference is `3.4e-11`.

## Clearing, execution, majorization and SPO+

The native `Optimization` functions `eisenberg_noe_clearing`,
`almgren_chriss_execution`, `is_majorized`, `is_doubly_stochastic`,
`majorization_matrix`, `birkhoff_von_neumann_decomposition` and `spo_plus_loss`
implement the preserved Python 3.2.0 contracts. `spo_plus` is the native Zygote
counterpart of `spo_plus_torch`, supporting vector and row-batched predictions
while treating truth, vertices and first-index oracle choices as constants.
Vertex and permutation indices are one-based. Dictionary results become named
tuples; no general optimization oracle or nonlinear feasible region is implied.

The shared fixtures contain 39 clearing networks/scales, 20 execution cases,
40 SPO+ cases, 15 majorization comparisons and nine stochastic matrices.
Independent checks use monotone descent to the greatest clearing fixed point,
the execution inventory first-order system, finite-hull support values and
subgradient inequalities, and stochastic-witness/permutation reconstruction.
PyTorch gradients and native Zygote gradients agree on the supplied SPO+ cases,
including ties and batches. Clearing residuals use the caller's scaled `1e-8`
default tolerance. Execution comparisons use `rtol=1e-10`, `atol=1e-12`.
Majorization witnesses and decompositions use absolute residual `1e-10` and
are compared by their defining constraints, since their representations are
not unique. Majorization partial sums use higher-precision accumulation.
The Birkhoff implementation uses native augmenting-path matching on positive
residual support; no equality of decomposition order is required.

Nineteen additional public-book clearing cases cover baseline networks, near-zero
unpaid balances, repeated shocks and monetary scales from `1e-12` to `1e12`.
The internal Clarabel accuracy cap is `min(1e-14, atol/10)`; the caller's
feasibility tolerance, linear objective and greatest-fixed-point contract are
unchanged. Rescaled payments are bounded again by zero and nominal liabilities
to remove last-bit rescaling overshoots. Across 2,089 public-book scenarios,
payments and unpaid balances match the preserved Python results at
`rtol=1e-9`, `atol=1e-11`, with identical default decisions and independent
monotone-iteration checks. The largest absolute gap occurs at monetary scale
`1e12` and is below `7e-15` after normalization by that scale.

## Ledoit--Wolf simulation covariance

`Data.simulates_cov_mu(...; shrink=true)` now uses `CovarianceEstimation` with
the centered maximum-likelihood covariance, Ledoit--Wolf shrinkage and a
scaled-identity target. Constant observations, one feature and exact target
degeneracy return the empirical covariance; they do not switch estimators.
Centering relative to the first observation and rescaling internally avoid
constant-data cancellation and fourth-moment overflow. The default unshrunk
path retains sample-covariance normalization.

Eight shared observation matrices run through the preserved Python simulation
adapter and the Julia adapter with only their random draws supplied explicitly.
They cover `p > n`, rank deficiency, one feature, zero and constant observations;
the supplier fixtures independently verify the outer-product shrinkage formula.
Comparisons use `rtol=1e-10`, `atol=1e-12`, with additional native scale-equivariance
checks at `1e-100` and `1e100`. Equal NumPy and Julia seeds are not treated as
equivalent streams. Julia returns a mean vector and covariance matrix, including
the one-feature case where NumPy's unshrunk covariance is scalar.

## Supplied-prior ambiguity and exponential Hawkes/VPIN

`Features.probability_ambiguity` computes the prior-weighted population
variance of exhaustive bin probabilities, weighted by their prior means.
`normal_prior_bin_probabilities` supplies stable normal-tail bin probabilities.
The scope is the supplied-prior measure; raw intraday probability estimation
and return-variance ambiguity proxies are not introduced. Eighteen ambiguity
cases and twelve binning cases are checked against pairwise population-variance
identities and an independent complementary-error-function calculation.

The six `Features` order-flow functions preserve the public empty-history,
univariate exponential Hawkes contract and VPIN's completed, equally sized,
already classified volume buckets. Intensity is predictable and excludes events
at the query time. Fifteen deterministic cases compare intensity, integrated
intensity, likelihood and the analytic score; quadrature and explicit event-pair
sums independently validate the reference. Six simulations share supplied
exponential and uniform draws, with no equal-seed cross-language assumption.
VPIN is checked against explicit rolling sums.

`fit_hawkes` uses Optim's bounded L-BFGS method, the analytic score, the same
three fixed starts and the same scaled parameter bounds as Python, or a single
caller-supplied start. Three reference fits and a time-unit transformation check
use an absolute likelihood gap of `1e-4`. Deterministic comparisons use
`rtol=1e-10`, `atol=1e-12`; finite-difference score checks use `rtol=1e-5`,
`atol=1e-7`. Fits are numerical local solutions, with no identification or global
optimality claim. Optimizer messages and convergence counts may differ, and
decay is unidentified in the zero-excitation limit.

## Accepted-DAG identification evidence and structural mechanisms

`CausalFactorAnalysis.CausalDAG` stores immutable canonical node/edge tuples and
explicit observed nodes. Latent nodes remain in all graphical calculations.
The native d-separation, ordinary back-door, front-door, minimal back-door set,
instrument and path-role functions reproduce the public Python criteria and
complete path evidence. Pearl's conditional instrument profile and the stricter
unconditional CFI profile remain distinct. Front-door eligibility requires a
nonempty observed mediator set and a directed causal path; positivity remains
an external premise. Search limits raise errors and never yield partial evidence.

Evidence records use named tuples. `classify_treatment_outcome_role` returns the
published role string instead of a Python enum; its eight permitted one-hop
signatures and rejection of reverse mediation are unchanged.
`evaluate_structural_causal_model` evaluates caller-supplied deterministic
mechanisms in lexical topological order with lexical parent columns. Inputs and
result values use immutable tuple-backed abstract arrays. A mechanism should
accept Julia abstract matrices/vectors; writes to snapshots are rejected.

Validation covers all 64 ordered four-node DAGs: 1,536 d-separation queries,
back-door/front-door subsets, minimal sets, instruments and path roles, plus
latent-confounding examples and all eight one-hop roles. NetworkX's independent
d-separator and directed reachability algorithms check the Python reference.
Julia compares complete normalized evidence fields exactly. Twelve structural
examples match a separately solved simultaneous linear system at `rtol=1e-10`,
`atol=1e-12`; additional checks verify immutability and domain/search boundaries.
These methods neither discover a graph nor establish its causal assumptions.

_(further submodules appended as they are wired)_

## Additive causal-factor completion

The released 57-name `CausalFactorAnalysis` set is preserved. The additive
release candidate exposes the same 30 new public names in Python and Julia,
producing an 87-name causal API:

- Evidence and value types: `AllocationMisspecificationDiagnostics`,
  `FDRComparisonEvidence`, `FDRNonIdentificationWitness`, `FactorControlRoles`,
  `GaussianSearchAdjustedFDR`, `GaussianTrialMixture`,
  `MaxSelectionFamilyErrors`, `SelectionLevelProbabilityEvidence`,
  `StructuralCausalModelResult`, `TreatmentOutcomeRole`, and
  `TreatmentOutcomeRoleEvidence`.
- Mirage and allocation functions:
  `generalized_confounder_undercontrolled_coefficient`,
  `generalized_collider_overcontrolled_coefficients`, and
  `allocation_misspecification_diagnostics`.
- Graph and structural-model functions: `factor_control_roles`,
  `classify_treatment_outcome_role`, and `evaluate_structural_causal_model`.
- False-discovery functions: `single_trial_false_discovery_rate`,
  `family_level_false_discovery_rate`, `max_selection_family_errors`,
  `compare_single_and_family_fdr`, `maximum_mixture_cdf`,
  `conditional_upper_tail_probability`, `gaussian_trial_mixture_cdf`,
  `gaussian_max_selection_cdf`, `gaussian_max_selection_log_density`,
  `gaussian_max_selection_log_likelihood`,
  `gaussian_search_adjusted_false_discovery_rate`,
  `fdr_nonidentification_witness`, and `max_selection_null_probability`.

The formulas, evidence fields, validation intent, deterministic ordering, and
estimands match. Language-native differences are deliberate: Python uses
frozen records, protected NumPy snapshots, `Enum`, and SciPy quadrature; Julia
uses immutable structs with independent arrays, a validated symbol-backed role,
multiple dispatch, and directly declared QuadGK. Quadrature error estimates are
backend-specific. The shared numerical fixture supplements independent native
mathematical and graph oracles; neither implementation is treated as the sole
correctness authority.

## Fitted density, targeted shrinkage and tail risk

`Data.fit_kde` returns an immutable native `KernelDensityEstimate` and
`Data.score_samples` evaluates its log density on a vector or one-column
matrix. The supplied positive numeric bandwidth and all six public kernels
are supported. Compact kernels exclude points exactly one bandwidth away;
log-sum-exp evaluation preserves far-tail Gaussian log densities. The public
targeted eigenvector shrinkage construction `denoised_corr2` retains the
supplied signal/noise partition, the noise diagonal and the `1e-12` absolute
diagonality tolerance, without additional correlation normalization.

`Backtest.expected_shortfall` preserves the Python signed lower-tail return
mean, type-7 quantile interpolation and inclusive ties. Its risk-free argument
is unused. `generalized_pareto_tail_risk` instead reports upper-tail loss
quantiles and expectations for a supplied threshold/exceedance rate. It uses
the exponential limit at shape zero and infinite shortfall for shape at least
one. `quantile_exception_tests` preserves the Kupiec and Christoffersen
likelihood conventions: coverage uses all observations, transitions condition
on the first, and absent previous-state rows yield `nothing` for unidentified
independence and conditional coverage results. P-values remain asymptotic.

`fit_pot_tail_risk` fits a zero-location GPD to strictly positive excesses above
the caller's threshold. Native Optim Nelder–Mead rescales excesses by their
mean, starts at shape/normalized scale `(1,1)`, allows unrestricted shape,
and checks numerical convergence, support and finite likelihood. This is
local numerical likelihood fitting, with no global-optimality or finite-MLE
existence guarantee. Threshold selection and held-out separation remain the
caller's responsibility.

The shared Python 3.2.0 fixtures contain 25 density cases, 63 shrinkage cases,
20 empirical shortfall cases, 40 Pareto risk cases, 387 exception-test cases
and ten fits. Independent evidence includes normalized kernel integrals,
outer-product reconstruction, 70-digit Pareto formulas, every binary sequence
through length six, direct transition likelihoods and chi-square survival
identities, and tightly converged likelihood/finite-difference score checks.
Deterministic comparisons use `rtol=1e-10`, `atol=1e-12`; numerical fit parameter
comparisons use `rtol=atol=5e-4` against the default SciPy fit, with a stricter
`1e-5` comparison against the tightly converged reference and objective error
below `1e-7` per excess. Native tests also check density snapshot immutability,
invalid domains and Pareto unit scaling by `1e-50` and `1e50`.

## Public factor, specification and allocation diagnostics

`CausalFactorAnalysis` implements the released `factor_mirage` coefficients,
conditional factor returns, integrated forecast returns and collider model
diagnostics, with immutable `StrategyPerformance`, `ColliderCoefficients`
and `ColliderDiagnostics` records. Exact rational intermediates preserve the
strict comparison rules and avoid intermediate overflow/cancellation; square
root signals use 320-bit arithmetic before checked Float64 conversion. The
optional realized confounder link follows the public regime-change contract.
Collider factor returns consume the supplied pre-outcome proxy. Diagnostic
t-statistic fields remain expected signals rather than p-values.

The three public `specification_diagnostics` systems have native population
and experiment functions and four immutable result types. Experiments retain
the fixed structural equations, independent standard-normal disturbances,
classical OLS variances, degrees of freedom, coefficient labels and sample-size
ceiling. Native seeds select a local Julia RNG; the optional `rng` permits
identical primitive draws. Shared tests compare both specifications against
the preserved Python execution and an independent pivoted-QR regression.
Tuples preserve coefficient/result snapshot immutability.

`minimum_variance_factor_weights` and `allocation_misspecification_diagnostics`
implement the supplied positive-definite covariance/full-rank equality
constraint problem with unrestricted weights. Cholesky whitening and reduced
QR preserve the public unit normalization, rank and feasibility checks;
square systems use the direct exposure solve. No covariance repair or
additional constraints are applied. The two exposure systems can have different
factor counts. Diagnostic vectors are independent immutable snapshots and
sign reversals use nonzero opposite signs, avoiding product underflow.

The factor/specification fixtures cover 50 coefficient/return cases, 150
diagnostic cases, five extreme-scale cases, three exact population systems and
12 shared-draw experiments. Independent covariance normal equations verify
coefficients, forecast products, variances and strict comparison rules. The
allocation fixtures add 61 weight cases and 15 comparisons checked by full
block KKT systems, an exact three-asset solution and covariance/exposure-unit
scaling. Deterministic numerical comparisons use `rtol=1e-10`, `atol=1e-12`,
with zero absolute tolerance for tiny extreme-scale coefficient results;
labels, counts and comparison flags are exact. These methods expose the
released synthetic/model calculations without new empirical conclusions or
paper-specific collider allocation appendices.

## Search-adjusted false-discovery evidence

`CausalFactorAnalysis` exposes the 13 released search-adjustment functions and
six immutable evidence/model types. Supplied trial errors, mixture parameters
and independent search counts retain their public domains. Log-domain
evaluation preserves posterior boundaries and Gaussian tail calculations.
Family FDR conditions on the event that every searched candidate is null;
`max_selection_null_probability` concerns the selected winner's null status.
The equal-component nonidentification witness remains limited to the
unrestricted model class. No empirical parameter-fitting procedure is added.

The native selected-winner integrator uses adaptive 15/7-point Gauss–Kronrod
quadrature after transforming the infinite interval. It validates callback
values, positive selection probability, convergence tolerances, subdivision
limits and the final probability bound. Its reported absolute error estimate
is implementation dependent and is not expected to equal SciPy's estimate.
The mathematical rule constants were generated locally with the already
pinned QuadGK supplier and checked against 37 polynomial moments; QuadGK is
not added as a runtime dependency.

Shared fixtures cover 54 single-trial posteriors, 216 family posteriors, 192
family-error cases, 40 comparisons, 81 mixture maxima, 12 conditional-tail
ratios, 18 Gaussian distributions/likelihoods, 12 aggregate threshold records,
16 nonidentification witnesses and 27 winner-probability integrals. Independent
checks use exact rational arithmetic, 100-digit probabilities and Gaussian
tails, 2048-point Gauss–Legendre integration in null-probability coordinates,
and analytic equal-component/one-trial identities. Deterministic and shared
probability comparisons use `rtol=1e-10`, `atol=1e-12`; the independent fixed-rule
integrals use `rtol=1e-8`, `atol=1e-10`. Native tests also verify immutable
threshold snapshots and rejection of undefined posteriors, inconsistent
callbacks and exhausted integration limits.

## Kelly sizing and probability calibration

The four public `advanced_bet_sizing` interfaces preserve binary Kelly payout
conventions, long-only clipping, the pessimistic standard-error probability
box and right-closed calibration-error bins. Native scalar and array sizing
retain their shapes. `PlattCalibrator` uses `fit!` and `transform`; an unfit
model or one-class calibration block is the identity. Fitted predictions use
the public row-major flattening order. Calibration-block separation remains
the caller's responsibility.

The Platt objective retains the clipped logit, fixed `C=1e6` slope penalty and
unpenalized intercept. Native L-BFGS uses tighter numerical stopping criteria
than the default Python solver. Eight shared fits are checked against an
independent Newton solution refined by the analytic score equations, using
`rtol=atol=1e-5` for parameters and predictions and an explicit objective check.
The default Python prediction differs by up to 0.028725 on the separable
synthetic fixture; differences are bounded using its residual score, Hessian
curvature and the sigmoid Lipschitz constant. This is a stopping-accuracy
difference for the same objective, with no calibration-efficacy claim.

Nine Kelly cases, 108 robust-sizing cases and four calibration-bin cases also
use shared Python 3.2.0 outputs and independent growth/bin-membership checks.
Native tests cover invalid inputs, floating-point bin edges, one-class refits
and matrix prediction order.

## Seven-stage causal-factor protocol records

The 11 public immutable protocol record types and
`validate_causal_factor_protocol` preserve the released structural evidence
contract. Constructors accept positional or keyword fields, freeze valid
vector collections into tuples and retain malformed values for aggregate
validation. The report exposes the fixed source metadata and published stage
order. Julia raises `ArgumentError` with the same unique stage-specific
diagnostic set; diagnostic order is not contractual because Python iterates
sets. The protocol's numbered stages do not change disclosure classification
or approval state.

Validation retains half-open event intervals, complete training membership,
prediction-before-refit evidence, purging and post-test embargoes, complete
and distinct CPCV paths, resolved graph roles and identification premises,
purpose-compatible separate performance evidence, implemented portfolio
considerations, and complete trial families with dependence-aware correction
inputs. Successful validation checks structural consistency of the supplied
record. It neither runs the recorded algorithms nor proves causal premises
or the truth of empirical evidence.

The shared synthetic fixtures contain 777 records: 77 accepted and 700
rejected, covering 332 distinct diagnostics. Independent NetworkX graph
criteria, pairwise interval arithmetic, fold/path multiplicities and trial
partition counts supplement Python 3.2.0 execution. Native tests also check
tuple snapshot immutability, constructor defaults, fixed report properties,
and touching versus overlapping half-open intervals. The initial native
empty-conditioning set error and its corrected validation run are retained
in the local program evidence.

## Predict-before-update adaptation

`prequential_adaptation` preserves the caller's estimator through explicit
native `clone_estimator`, `fit_estimator` and `predict_estimator` callbacks.
Each fit starts from an independent unfit clone; the caller supplies model
parameters and random state. Warm-up forecasts/losses are NaN. Alarm and refit
indices are one-based, and the alarm row enters its rolling training window
only after that row has been predicted and scored. Callback input slices are
independent copies.

The internal ADWIN2 implementation keeps chronological exponential summaries,
with the fixed public settings of five summaries per size, checks every 32
losses, minimum subwindow size five and grace period ten. It preserves the
installed River 0.26.1 detector's next-observation reset and cut arithmetic.
Losses are finite values in `[0,1]`. The adaptive detector window and the
caller-selected rolling estimator window remain separate.

Shared fixtures validate 24 detector streams with 36,864 updates and 16
mean/ridge adaptation scenarios. Detector alarms and widths match exactly;
totals, centered sums of squares, predictions and losses use `rtol=1e-10`,
`atol=1e-12`. Independent retained-window moments, augmented least-squares
regression, recorded fit intervals and 64 prefix comparisons validate the
reference. Native tests additionally check summary reconstruction, isolation
from callback input mutation, invalid domains and reset timing. No predictive
improvement or general adaptation guarantee is asserted.

## Single-asset entropic deep hedging

The six public deep-hedging interfaces use native arrays, Zygote and the
already pinned Optimisers Adam. `DeepHedge` retains the shared three-layer
dense/tanh architecture, remaining-time/current-log-price/previous-holding
inputs and unconstrained positions. Explicit parameter arrays are copied on
construction; seeded initialization and GBM simulation use local Julia RNGs.
Supplied primitive normal draws and supplied network weights provide direct
cross-language comparisons without equating language-specific seed streams.

`hedging_pnl` preserves the self-financing short-call cash flow, premium and
proportional entry/rebalancing/liquidation costs. `entropic_risk` preserves the
detached minimum anchor and exponential-weight gradient. Call deltas use
zero rates, and GBM drift remains a supplied simulation-measure parameter.
Training mutates only model parameters using supplied training paths,
full-batch entropic loss and Adam with the public decays and epsilon.
Each recorded loss precedes its parameter update; independent evaluation
paths enter only subsequent evaluation.

The 76 shared fixtures cover 16 GBM cases, 24 Float64/Float32 cash-flow cases,
16 entropic risks, 12 delta grids and eight shared-weight training cases.
Independent checks include 80-digit cash ledgers and risks, analytic holding
and risk gradients, a NumPy recurrent network, directional finite differences
and explicit Adam moment/bias-correction replay. Float64 deterministic and
network checks use `rtol=1e-10`, `atol=1e-12`. Float32 cash-flow roundoff is
bounded by `32*eps` times the absolute operation scale; tiny-aversion risk
uses `16*eps*max(1,abs(anchor),log(n)/aversion)`. Native tests verify global RNG
preservation, immutable input parameter snapshots, adapted positions and
unchanged weights during held-out evaluation. The initial overly strict
decimal-equality assertion is retained in the local test evidence. No
optimality, pricing-measure or performance-superiority claim is added.

## Temporal-only TimeGAN

`chronological_timegan_data`, `TimeGAN`, `fit_timegan` and
`timegan_quality_report` retain the public fixed-window temporal model.
Data split before windowing; scaling uses training minima/ranges, constant
training features have scale one and validation values are never clipped.
Arrays have `(windows,time,features)` order. The Float64 native GRUs use
reset/update/new gates with reset applied after the hidden projection.
Embedding and recovery are causal, the generator is autoregressive, and the
discriminator is bidirectional. `latent_sequence`, `timegan_losses` and
`sample_timegan` provide the native counterparts of Python instance methods.

Training preserves reconstruction, teacher-forced supervision including the
initial latent, and the minimax fake-data logarithm. The three fixed phases
use separate Adam states with the public decays and epsilon. Joint training
updates E/R, then G, then D, recomputing losses between updates. Histories
contain losses before each step's final group update. Optional shared weights,
one-based minibatch indices and normal draws separate numerical comparisons
from language-specific RNG streams. Seeded operations preserve the global RNG.

Twenty shared cases cover four chronological splits, four shared-weight
models with all training phases, and twelve held-out quality scenarios.
Independent NumPy GRU recurrences, stable loss formulas, 112 finite-difference
parameter directions and explicit Adam moment/bias-correction replay check
the reference. Quality checks use direct lag-pair enumeration, augmented
least-squares ridge with alpha one and Newton score/Hessian checks for
balanced logistic regression with C one and an unpenalized intercept.
The native logistic optimizer solves the same objective more tightly than
the default Python stopping rule. Fixture predictions agree; default Python
parameter distances from the independently refined solution reach 0.000713.
Deterministic outputs, gradients and trained parameters use `rtol=1e-10`,
`atol=1e-12`; finite-difference checks use `rtol=2e-5`, `atol=2e-7`.

Native validation also checks causal prefixes, teacher timing, constructor
snapshots, invalid domains and fresh-noise evaluation after training. Matrix
sequence accumulation avoids a lazy tuple-gradient incompatibility in the
pinned differentiation packages; the failed runs remain in local evidence.
No separate supervisor or moment-matching network is added. Synthetic windows
are not an unlimited market path. Descriptive quality scores neither select
parameters nor certify realistic dynamics or uncertainty for overlapping
windows; chronological real validation and fresh synthetic noise remain
caller-supplied requirements.

## Fixed-coordinate constrained neural SDEs

The ten public neural-SDE interfaces preserve the supplied finite-grid and
price-basis contract. Call-grid inequalities retain intrinsic/upper bounds,
vertical-spread slopes, strike convexity including the known zero-strike
price, and calendar ordering. A separate static probability witness rejects
positive flat tails and verifies total mass, mean one and quote reconstruction.
Scalar affine state bounds reject unbounded, empty and singleton intervals.
The HJM diagnostic solves only the supplied finite-grid linear equation by
minimum-norm SVD with dimension-scaled machine-epsilon rank tolerance.

`IntervalNeuralSDE` and `PolytopeNeuralSDE` use copied explicit Float64 weights
or local seeded dense/tanh initialization. They preserve inward drift
corrections and diffusion that vanishes linearly at a boundary. The polytope
uses the corrected fixed-coordinate transformation `Q*diag(g(distance))*Q'*L`,
with two-pass orthogonalization of stably sorted nearest independent normals.
This retains the public correction to the paper's one-sided square-root
transformation. Approved JuMP/Clarabel checks boundedness after normal-row
normalization. Negative distances within relative `1e-12` roundoff tolerance
are evaluated as zero; positive distances are not snapped to a face.

`negative_log_likelihood` is the native instance-method counterpart and
preserves the approximate Gaussian Euler objective. Boundary starts and
singular transition covariance fail without variance jitter. Fitting uses
fixed iterations of Adam on supplied training transitions; histories record
pre-update losses. Simulation uses caller-supplied Brownian increments under
the fitted measure. Any Euler exit invalidates the entire run and identifies
its one-based step. No clipping, projection, reflection or path removal occurs.

Thirty-seven shared cases cover four call grids, eight affine intervals,
twelve HJM systems, eight scalar models and five polytope models including
rotated boxes and a simplex. Independent slope inequalities, payoff sums,
rank-revealing QR, dense networks, sums of rank-one diffusion shrinkers,
Gaussian likelihood formulas, parameter/state finite differences, explicit
Adam replay and Euler updates check the reference. Deterministic values,
gradients and trained parameters use `rtol=1e-10`, `atol=1e-12`; directional
finite differences use `rtol=2e-5`, `atol=2e-7`. Native checks include inward
face drift, zero normal diffusion, row rescaling/permutation, positive
near-boundary diffusion, held-out evaluation and explicit exit failures.
The corrected standalone zero-tangent assertion is retained in local evidence.

Static witnesses, boundary conditions and even a zero HJM residual do not
certify a dynamic martingale measure, off-grid prices, global calibration,
parameter recovery or invariant discrete Euler steps.

## PDE network components

Twelve native network types and the dense initialization callback preserve
the public linear, sine, bias-free ReLU, time-modulated and set-attention
architectures. Float64 parameter trees accept copied explicit weights;
local seeded initialization leaves the global random stream unchanged.
Inactive Python constructor modules are omitted. Attention retains the
public square-root-of-total-value-dimension scaling and population layer
normalization with epsilon `1e-5`. Set permutation identities are checked.

The time-dependent sine network retains active batch normalization with
epsilon `1e-6`, momentum `0.01`, population training variance and unbiased
running variance. `model.training=false` uses fixed running moments. The
Monte Carlo network requires the caller's `y_mc` input and implements the
public control formula without inferring a gradient estimator.
`initialize_weights` mutates one dense parameter leaf using supplied or
local normal draws with standard deviation `0.1` and zero bias.

Sixteen network cases and three initializer cases compare shared forward
values, complete input gradients, parameter gradient norms, projections and
selected coordinates against preserved Python 3.2.0. Independent NumPy
equations and 278 finite-difference directions check the reference.
Deterministic values and gradients use `rtol=1e-10`, `atol=1e-12`;
finite differences use `rtol=2e-5`, `atol=2e-7`. Sequential layer evaluation
and a distinct residual-branch input copy preserve gradients in the pinned
differentiation runtime. Failed gradient runs remain in local evidence.

This coverage validates network components. Full PDE solver training parity
and convergence are separate requirements. Reproduced Python solver
validation-mode and missing-Monte-Carlo-input errors remain recorded locally.

## PDE equations and solver training

Twelve shared equation cases validate all four public rates, Hamiltonians,
volatilities, terminal functions and supplied-normal forward paths against
independent cumulative-sum, product and exact geometric-step calculations.
`Equation` remains a native abstract interface. Dimensions and step counts
must be positive; horizons must be finite and nonnegative. The public
`PricingDiffRate` multi-sample terminal helper returns only the strike-120
call, while its ordinary terminal returns the strike-120/150 call spread.
Both contracts remain distinct and are explicitly checked.

`FBSDESolver`, `FBSNNSolver`, `compute_pde_loss` and `solve_pde` provide native
loss evaluation and fixed-step training. The deprecated Python misspelling
`FBSNNolver` maps to the canonical native `FBSNNSolver`. DeepBSDE retains the
terminal MSE, bias-free ReLU networks, network Adam learning rate and separate
initial-value/gradient learning rates `0.1` and `0.001`. All three use betas
`(0.9,0.99)` and epsilon `1e-8`. The unused first Python DeepBSDE network is
omitted. The existing `solve_deep_bsde` convenience wrapper now delegates to
this contract, replacing its earlier single-optimizer approximation.

FBSNN retains the sum of one-step residual MSEs plus terminal MSE and Adam
betas `(0.99,0.999)`. Analytic sine-network state derivatives remain
differentiable in the parameters and available during validation. Backward
solvers other than DeepBSDE keep the supplied initial value during training
and report the public coefficient-based estimate after each update. DTNN
retains training-mode batch moments for validation. Network optimizer state
persists across repeated training calls; DeepBSDE initial parameters restart.

Three unmodified Python execution failures are preserved in local evidence:
FBSNN disables derivatives during validation, the Monte Carlo call omits its
required gradient input, and the set-network call omits a required array
axis. Native Monte Carlo training requires caller-supplied gradient arrays;
the set route inserts a singleton set axis within the public stated
one-dimensional equation scope. No estimator or higher-dimensional set
control is inferred. Explicit batch inputs are validated and copied before
training; ordinary sampling uses a local RNG, one fixed validation batch of
128 and fresh training batches.

Sixteen shared solver cases validate losses, coefficient recurrences,
complete path/increment/initial-parameter gradients, parameter-gradient
summaries, trained parameter summaries, normalization state and two-step
optimizer replay. Independent NumPy recurrences and analytic derivatives
check values; 286 parameter directions check gradients; direct moment and
bias-correction calculations check every Python Adam update. Deterministic
comparisons use `rtol=1e-10`, `atol=1e-12`. Finite differences of the larger
PDE objectives use `rtol=1e-4`, `atol=5e-5`. Shared comparisons use Float64
inputs, weights and temporary reference buffer defaults. They do not assert
equal random streams, PDE convergence, approximation accuracy or an error
rate. The corrected continuation-test naming error remains in local evidence.

## Fractional filters and default-wavelet adaptive differencing

The default adaptive algorithm now includes the native db2 wavelet-variance
component and its equal-weight blend with R/S Hurst. Closed-form db2 and Haar
filters use half-sample symmetric reflection, the public level rule, mean
squared detail coefficients and the same octave exclusion. `db1` aliases Haar;
other named filters are explicitly unsupported. The forced three-level rule
on short inputs is preserved; such boundary-dominated estimates are not given
an additional statistical interpretation.

Fixed and expanding filters forward-fill internal missing observations while
retaining leading missing values and original output positions. Adaptive
memory estimation removes missing levels before forming increments, as in
Python, then filters the original aligned series. Returned Julia vectors
retain their NaN warm-up and original length. Threshold ranking and upward
order selection retain the public ADF significance and lag settings. The
existing ADF supplier uses the same response-surface probability convention;
shared probabilities and selection outcomes are checked directly.

There are 295 shared cases: 49 weight cases, 35 filters, 180 wavelet/R/S
combinations, 15 ADF checks and 16 complete adaptive selections. Independent
high-precision generalized binomial weights, direct sums, explicit reflection,
block R/S ranges, pivoted-QR ADF regressions and candidate enumeration check
the reference. Default numerical tolerances are `rtol=1e-10`, `atol=1e-12`;
ADF probabilities and complete aligned adaptive outputs use `rtol=1e-9`,
`atol=1e-11`. A finite positive FFD threshold is required; integration weights
that never cross the supplied threshold fail explicitly. The former R/S-only
regression anchors are updated to the preserved wavelet-enabled reference.
This validation makes no new empirical accuracy or predictive-lift claim.

## Estimator-driven validation paths and adaptive boundaries

Native K-fold, purged K-fold, combinatorial and walk-forward validators now
provide complete paths through `backtest_predictions`. The caller supplies
`clone_estimator`, `fit_estimator`, `predict` and, when requested,
`predict_proba`. Each fit receives copied training rows and the aligned
observation weights; omitted weights become ones. The fitting callback returns
the fitted model. Failures propagate rather than triggering the Python wrapper's
unweighted retry. An unweighted estimator can be used through an explicit
caller-written callback when that is the intended contract.

Combinatorial validators fit once per combination and reuse each fitted model
across its path segments. Integer path keys replace `"Path N"` strings. Shuffled
K-fold restores observation order and now sorts training positions as the
reference does. Supplied one-based permutations provide exact shared-draw
comparisons; Julia and NumPy seeds do not imply the same random stream. Standard
lexicographic CPCV paths retain their existing order. The path-location helper
now also returns group order when combinations arrive in a different order.
Walk-forward folds without training observations retain their NaN predictions.
Keyed dictionaries of validators and estimator/data inputs support separate
datasets without inferring a shared model.

`AdaptiveCombinatorialPurgedCV` copies the supplied external feature and uses
the public full-vector linear quantiles, fine contiguous subsplits and signed
boundary adjustments. No feature or regime estimator is inferred. The default
three fine subsplits preserve contiguous complete group coverage. Small
subsplit counts can yield empty or overlapping groups under the source rule;
the native implementation preserves these outputs explicitly. Inputs require
finite features, aligned increasing event starts, ordered quantiles and at
least one observation per fine subsplit. Public continuous-range purging uses
the last test event's end, while disjoint ranges retain inclusive overlap and
observation-count embargo rules.

There are 1,718 shared cases: 1,536 exhaustive purge/embargo cases, 104 complete
combinatorial/adaptive geometries, 54 basic fold designs and 24 prediction
scenarios. Independent interval intersections, group occurrence lists, linear
quantiles, split arithmetic and deterministic test-estimator formulas check
the reference. Row and weight traces verify clone/reset behavior and one fit
per combination. Index comparisons are exact; probability comparisons use
`rtol=1e-12`, `atol=1e-13`. The diagnostic estimator is used only in tests and
does not stand in for a book experiment's estimator. Bagged estimator fitting,
scientific model selection and empirical comparative findings are separate
contracts.

## Bagged combinatorial validation

`BaggedCombinatorialPurgedCV` fits one bagged caller estimator per combinatorial
split and assembles the same chronological paths. Its validated supplier
contract is scikit-learn 1.9.1: observation weights determine sampling
probabilities, floating `max_samples` multiplies the sum of training weights,
and the base estimator receives sampled-row multiplicities when it supports
weights. With explicit `fit_sample_weights=false`, it instead receives sampled
rows and no weight argument. Integer sizes remain counts. Feature draws follow
the requested feature fraction/count and replacement rule.

Classifier targets use the wrapper's zero-based class codes internally;
returned labels retain their original sorted classes. Member probability
columns are aligned before averaging. Without a probability callback, the
ensemble averages hard votes. Regressors average predictions and reject
probability requests. All fits are serial and failures propagate. Native
random streams are local when `random_seed` is set; supplied training-relative
row positions, feature positions and estimator seeds enable exact replay
without a cross-language seed-equivalence assumption.

Thirty-six shared cases cover probability classifiers, hard-vote classifiers
and regressors, weighted and explicitly unweighted members, omitted and unequal
observation weights, and three row/feature sampling designs. Each case has six
combinations and three members per combination. Independent arithmetic verifies
all fitted-input traces, weighted sample counts, class alignment and averaged
path outputs at `rtol=1e-12`, `atol=1e-13`. Test estimators remain diagnostic
fixtures; the public book's selected estimator and scientific scoring rule
must still be supplied by its caller.

## Weighted caller-estimator feature importance

MDI, MDA and SFI now have overloads taking an explicit estimator plus fit and
prediction callbacks. Clustered MDI/MDA have matching native functions.
MDI can also aggregate a supplied tree-by-feature matrix. These interfaces
preserve the caller's model and observation weights; the existing two-argument
DecisionTree.jl convenience methods retain their separately documented behavior.

MDI replaces zero tree importances with NaN, skips these entries in feature
means and sample deviations, scales deviations by the square root of the total
tree count, and normalizes both columns by the sum of means. Clustered MDI first
sums each cluster within each tree, including zero for an empty/all-missing
cluster. Overlapping clusters retain their supplied memberships.

MDA uses separate training and scoring weights, the public negative-log-loss
difference, and joint row permutations within each supplied cluster. Empty
clusters have zero decrease. SFI uses weighted accuracy or negative log loss
and advances in feature-major, fold-minor order. Both methods retain sequential
estimator fitting state, including a caller's warm-start behavior. Returned
results include the final fitted estimator. Supplied folds and permutations
permit exact replay; native random streams do not imply NumPy seed equivalence.
Log loss clips at the prediction floating type's machine epsilon and preserves
the supplier's weighted-average convention.

PCA now removes constant or incomplete columns before normalization, matching
the public column filter. Weighted Kendall tau now uses both decreasing
lexicographic rankings to resolve ties and orders NaNs below other scores.
An empty or completely tied comparison remains undefined.

The 141 shared cases include 24 PCA configurations, 91 weighted-rank cases,
eight impurity matrices, sixteen weighted importance scenarios and two loss
boundaries. Independent SVD, exact rational pair sums, NaN-aware aggregation,
weighted probability/loss arithmetic and fitted-input traces check the
reference. Importance tolerances are `rtol=1e-11`, `atol=1e-13`; PCA uses
`atol=1e-10` for spectral and invariant comparisons.

At a 100% variance threshold on one centered five-by-eight fixture, Python
retains a fifth eigenvalue of `1.81e-15`, while Julia retains four components.
Independent SVD identifies four nonzero directions; transformed observation
Gram matrices differ by at most `2.39e-15`. The threshold algorithm is unchanged.
Tests require the same positive spectrum and reconstruction, while allowing
numerically zero component-count differences and arbitrary repeated-eigenspace
bases. These checks validate calculation contracts and do not establish a
feature's predictive or causal relevance.

## Weighted native pipeline and classifier search

`SampleWeightedPipeline` composes caller-selected native estimators through
explicit clone, fit, transform, parameter-setting, prediction and class-label
callbacks. Global fitting weights go only to the final estimator, overriding
an explicitly supplied final-step weight. Intermediate transformers receive
weights only when their own `step__sample_weight` option is supplied. Fitting
arrays are sliced with each training fold. Callers provide a fresh-estimator
clone operation; fitted state is never reused between candidates or folds.

`clf_hyper_fit` preserves the public scoring decision: labels exactly `{0,1}`
use F1 with positive class 1; other labels use negative log loss. The validated
scikit-learn 1.9.1 supplier also routes top-level `sample_weight` to scoring.
Step-prefixed weights affect fitting only. Shared fixtures expose a maximum
weighted-versus-unweighted score difference of `0.3919975566370453`, so these
routes are tested separately. Fold scores use an arithmetic mean, first ties
win, and the selected estimator is fitted again on all observations. Default
individual fit/scoring failures become NaN; all failed fits raise an error.

Finite grids use sorted keys with the final key varying fastest. Finite random
search samples without replacement; distribution-valued grids use supplied
draws or native distribution sampling with replacement. One-based supplied
candidate positions, explicit parameter draws and bagging row/feature/seed
plans permit replay across languages. Native seeds alone do not reproduce
NumPy streams. Execution is serial. Optional post-search bagging preserves
the validated supplier's weight-proportional sampling and multiplicity fits,
with original class labels restored in predictions. As in the supplier's
default configuration, step-prefixed fit options are rejected at the outer
bagging fit when metadata routing is unavailable.

The 44 shared fixtures cover six direct weight-routing cases, 24 purged
grid/random searches, eight post-search bags, five failure/tie cases and one
continuous-distribution search. Independent standardization, class-probability
arithmetic, F1/log-loss sums, interval purging, candidate enumeration and every
estimator's fitted-input trace check the reference at `rtol=1e-11`,
`atol=1e-13`. Diagnostic classifiers in the fixtures test these contracts only;
they do not replace the books' estimators or establish predictive relevance.
The existing random-forest search convenience functions retain their earlier
behavioral scope.

## Backtest statistics and Sharpe inference

Shared fixtures check fourteen public statistics and inference functions across
1,204 synthetic cases. Exhaustive four-position tracks are checked by an
independent pro-rata entry-lot ledger. Calendar concentration includes every
month between the first and last observation, including empty months, and
counts nonmissing observations. HHI and drawdown grouping preserve the public
missing-value behavior. Date and DateTime holding durations use the same day
units; drawdown durations retain the public 365.25-day year convention.

The preserved Python concentration function uses pandas' removed `M` frequency
alias and fails on pandas 3.0.6. Its failure is recorded. A bounded in-memory
replay maps only `M` to the equivalent month-end alias `ME`; separate calendar
enumeration checks the resulting arithmetic. The preserved source is unchanged.

Conditional drawdown fixtures independently integrate fractional empirical
tail mass. Sharpe-difference tests accept supplied one-based circular-block
resamples, checking the full studentized bootstrap probability against direct
moment arithmetic and dense Bartlett covariance kernels. Default native random
seeds remain language-specific. The public 5% rejection rule is unchanged.

Probabilistic Sharpe, benchmark maxima, influence functions and HAC intervals
retain their existing population/sample variance conventions. Oversized HAC
lags contribute zero beyond the sample length while preserving the requested
Bartlett denominator. Standard-library normal quantiles and complementary error
functions independently check probabilities and intervals. Comparisons use
`rtol=1e-11`, `atol=1e-13`, including explicit NaN/infinity outcomes where the
public contract is undefined or unbounded. These checks establish arithmetic
and replay consistency; they do not establish empirical test size, coverage or
predictive performance.

## Multiple-testing, test-set and binary strategy formulas

Another 634 shared synthetic cases cover fourteen existing functions: Holm and
BHY p-value corrections, Sharpe haircuts, expected maximum Sharpe, test-set
error formulas, binary-outcome risk and Gaussian mixtures. Exact rational rank
envelopes check all 125 length-three combinations of five fixed p-values.
Independent normal probabilities and inverse probabilities use sufficient
precision to resolve the public `1e-300` haircut floor. The public truncated
Euler constant in expected maxima is preserved.

Binary moments are checked by enumerating the two outcomes. Implied precision
also matches 70-digit quadratic roots and defining residuals. Boundary values,
including the public zero-dispersion and infinite-frequency conventions, remain
explicit. Gaussian mixtures now accept a pair of supplied standard-normal
vectors and a one-based output permutation. Component counts, zero deviations,
empty/singleton samples and input preservation are tested without equating
native and NumPy seeds.

Comparisons use `rtol=1e-11`, `atol=1e-13`; inverse-relationship checks account for
floating-point propagation. The significance thresholds and statistical targets
are unchanged. This arithmetic validation does not establish empirical
error-control or strategy performance.

## Probability and sigmoid bet sizing

The eleven existing public bet-sizing functions are checked with 347 shared
cases and independent interval enumeration. Inclusive bet intervals retain
their closing observation; active signals use half-open intervals and missing
end times remain active indefinitely. Zero-duration intervals, reordered starts,
signed meta-labels, class-count transforms, ties-to-even discretization and
signal saturation are covered. Native inputs remain already-aligned vectors;
pandas index alignment is a caller responsibility under the documented adapter.

Sigmoid positions use truncation toward zero. Inverse-price and width boundary
conventions, and the public limit-price range over absolute position indices,
are preserved exactly, including empty position ranges. Checks use
`rtol=1e-11`, `atol=1e-13` and separately verify valid sigmoid inverse identities.
No sizing rule or profitability interpretation is changed.

## OU first-passage metrics and fitted persistence

Earlier validation checked seven public interfaces through 250 shared cases.
Adaptive scale integrals independently verified hit probabilities; scalar
outcome enumeration checked gain moments. Eighteen differential boundary
problems checked the earlier 400-node first-exit quadrature, with maximum
relative error `6.12e-6`. The native AR(1) fit uses minimum-norm least squares
for rank-deficient inputs and centers residuals when computing the released
two-degree-of-freedom variance. Finite-row filtering, persistence clipping and
pre-clipping residuals are retained.

The approved Chapter 22.02 correction splits the mean-exit integral at entry and
uses 64-point Gauss–Legendre quadrature on each side. The existing quadrature
fallback is retained. The search also checks edge endpoints and refines
stationary maxima on all four box edges. Its approved boundary-count assertion
requires four contacts. Normal-import and final-location validation passed in
the recorded tested Windows environment; no validation beyond that environment
is claimed, and all documented limitations remain unchanged.

The existing coarse-to-fine Julia optimizer remains distinct from SciPy L-BFGS-B.
Six earlier shared objective-and-bound checks gave maximum objective difference
below `7e-12` and threshold difference below `3.5e-6`; these checks do not certify
identical optimizer iterates or universal global optimality. Metrics use
`rtol=1e-9`, `atol=2e-11`; fitted values use `rtol=1e-8`, `atol=2e-9`.

## Discrete-message entropy

Eight public entropy interfaces and the longest-match helper are checked with
3,854 shared cases. Exhaustive binary strings through seven symbols and Unicode
code points test overlapping word counts, Shannon and bias-corrected entropy,
Lempel-Ziv parsing and Kontoyiannis windows. Independent trie parsing and brute
substring matching retain the public overlapping-match convention.

The NSB check uses 45-digit concentration inversion and rising-factorial
evidence independently of the released floating log-gamma computation. It
validates the same finite trapezoidal grid, with `rtol=1e-9`, `atol=2e-10`;
other entropy checks use `rtol` between `1e-12` and `1e-11`. No estimator or
alphabet convention is changed.

## OHLC spread and volatility calculations

Seven public interfaces are checked through 197 shared cases. Native rolling
calculations now treat infinity as missing, matching pandas alongside NaN
warm-up and window behavior. Scalar sums, extrema and a hyperbolic-tangent
identity independently check spread calculations. The released
Bekker-Parkinson formula is preserved.

Row-wise EDGE indicators and moment calculations cover short and constant
series, missing OHLC components, sparse trading and seven negative signed
estimates. Inputs remain already-aligned vectors; values and scale invariance
are checked at `rtol=1e-9`, `atol=2e-12` without an empirical spread-accuracy claim.

## Information distances and KSG jitter

Seven public interfaces are checked with 301 shared cases. Histogram edges now
match NumPy's floating edge construction, right inclusion and equality of
signed zeros. Automatic bin counts preserve NumPy's near-one tolerance and
avoid native integer overflow at large observation counts. Explicit empty
variation-of-information input is rejected, as in the public reference.

KSG accepts supplied standard-normal jitter for each input, then applies the
public population-deviation scale internally. Exhaustive Chebyshev neighbor
counts and harmonic-number identities verify tied, constant and short samples
without claiming equal native and NumPy seeds. Distance correlation also
matches an independent uncentered-moment identity. Shared values use
`rtol=1e-10`, `atol=1e-12`; histogram cells and neighbor counts are exact.

## Trend windows and missing sample weights

Four public interfaces are checked with 866 shared cases. Sixty-digit OLS
calculations independently verify the slope t-values; installed SciPy 1.18.1
behavior is retained for short and degenerate regressions. Trend scanning keeps
the first maximum after the public nonfinite-to-zero ranking transform, reports
the maximum-window end, and now removes undefined trend rows.

Exhaustive binary concurrency matrices and nonnegative weighted examples check
average uniqueness. Missing entries are excluded from row concurrency, matching
pandas. Time-decay accumulation resumes after missing weights while retaining
NaN at the missing positions. Native vectors must already be chronological;
reference fixtures explicitly check the sorting adapter. Numerical comparisons
use `rtol=2e-8`, `atol=2e-9`, including regression error propagation.

## CUSUM and barrier labels

Six public labeling interfaces are checked through 480 shared cases. CUSUM
now skips missing differences without resetting accumulated moves. A pairwise
weighted-variance identity verifies pandas' debiased EWM with absolute-time
decay and missing observations. Daily lookbacks, vertical dates, strict
horizontal-barrier comparisons, and first-touch times are checked separately.

Triple-barrier inputs may omit the side column, retaining the public default
of one. Meta-labeling uses the next available price for between-observation
timestamps and produces NaN when no later price exists. Missing side mappings
in the native event builder remain missing rather than becoming long positions.
Times, row selection and label signs are exact; floating values use
`rtol=1e-10`, `atol=1e-12`. Caller-aligned vector representations and the
existing labeling decision rules are preserved.

## ADF, PSY and structural-break simulation

Eleven public references are checked through 1,077 shared cases. The ADF design
now preserves pandas' order of operations around missing observations: discard
missing first differences, shift that compressed difference sequence, join it
to the original lagged levels, then remove incomplete rows. Independent QR
regressions and explicit enumeration check forward/backward windows and original
observation positions. Exhaustive short sequences check episode duration and
the first collapse timestamp.

PSY critical values accept supplied standard-normal innovations. The volatility
robust routine accepts supplied Rademacher signs. Independent path construction,
window enumeration and quantiles check the resulting statistics, p-values and
unchanged strict 5% decisions without equating runtime seeds. Statistical values
use `rtol=5e-8`, `atol=5e-9`; shared p-values, decisions and observation indices
are exact.

The existing normal/l2 PELT interface now computes centered segment variances,
avoiding cancellation for large price levels while preserving ruptures 1.1.10
pruning and returned boundaries. **Reference compatibility does not establish
global optimality:** seven of the 102 shared PELT cases differ from unpruned
dynamic programming on the same candidate grid. Exhaustive partition enumeration
confirms the first counterexample. For its 15 observations, normal cost,
`min_size=2`, `jump=1`, and penalty multiplier `0.5`, the reference returns
`[2,4,6,8,10,12]` at objective `-21.2006061218`; `[2,4,6,8,10]` attains
`-21.8979243821`. Both outputs, objectives and full synthetic inputs are retained
in the fixture. The maximum observed objective gap is `0.8161520392`.
The reference pruning rule has not been replaced with a new decision procedure.

## Synthetic covariances and regime paths

Seven public references are checked through 156 cases with scalar Gram sums,
block membership, drift profiles, parameter padding and independently computed
variance/return ledgers. Covariance generators accept supplied normal, uniform
and permutation draws. The one-point drift profile now starts at time zero.

Heston–Merton simulations accept already-correlated Gaussian outcomes and
Poisson counts. Native sampling supports the positive-semidefinite boundaries
`rho = ±1` and zero jump dispersion. Exact mean/covariance checks cover the
native Gaussian factor. The public jump compensator, variance reset and
nonnegative diffusion variance are unchanged; equal NumPy and Julia seeds do
not imply equal streams.

Regime simulations accept an explicit regime order, initial state and
transition uniforms. These preserve Python dictionary insertion order and
right-sided categorical CDF boundaries. Regime-specific parameter lengths
expand the selected states using the public rule. Missing log returns are
forward-filled before accumulation; leading missing prices remain missing.
The native price vector omits Python's business-day index as documented.
Shared indices, names and counts are exact; floating outputs use `rtol=1e-11`,
`atol=1e-12`. No empirical simulation-fidelity claim is inferred from parity.

## Covariance algebra, NERCOME and hierarchical allocation

Shared cases check covariance normalization, Marcenko–Pastur grids, spectral
reconstruction and noise eigenvalue averaging, normalized covariance weights,
and PCA risk allocation. One-point MP grids now return the lower support point.
PCA checks compare eigenspace risk and total variance, allowing arbitrary signs
and bases within repeated-eigenvalue subspaces. They do not equate raw signed
hedging weights across eigensolvers.

NERCOME accepts supplied row permutations. Thirty-six cases compare held-out
projected variances with independent centered Gram calculations, including
constant columns, zero splits and the public positive-definite eigenvalue floor.
These checks do not validate the historical empirical performance claims in
the existing documentation. Native seeds and degenerate eigenvector bases can
differ across implementations.

The native HRP wrapper now preserves the reference single-linkage traversal,
stable sorting of equal-height edges and ascending child identifiers. The
previous supplier order changed weights in six of sixty cases, by up to
`0.0496904218`. All sixty complete portfolios now agree. Sixty-four exhaustive
four-node binary-distance patterns check tie ordering; independent graph
connected components check every tree-cut threshold. Another 120 permutations
check recursive allocation. NCO's intra/inter-cluster arithmetic is checked
through eighteen supplied partitions; its default stochastic k-means output
is not equated with scikit-learn.

The previous native golden search and reference local optimizer chose different
factor counts in two of eighteen diagnostic cases. Those historical results are
preserved. The native fit now uses projected scalar BFGS with the reference
initial point, bounds, More–Thuente settings and outer stopping rules. The
public fit supplies the analytic derivative of the same objective; forward
finite differences remain the internal fallback when no derivative is supplied.
All 243 current shared fits select the same factor counts;
variance differences are below `5.6e-6` under the declared `1e-5` absolute
tolerance. Twenty-four covariance and portfolio cases independently check the
downstream spectral arithmetic. This is a local fit, not a global-minimum
guarantee; eigenvalues near a cutoff remain sensitive to roundoff.

All 31 canonical bar-state constants are checked exactly. The three cumulative
imbalance values now use the public Greek theta character while their Julia
identifiers remain ASCII.

### Clustering selection and streamed bars

Clustering selection now computes silhouette scores from Euclidean distances
between rows of the correlation-distance embedding, matching the reference.
A supplied `label_provider(features, cluster_count, rng)` replays candidate
labels through base selection and recursive refinement. Thirty-nine shared
base/top cases include nine recursive calls, eight accepted refinements and one
rejection. Native seeded calls preserve the caller's global RNG state. Default
native and sklearn k-means partitions remain distinct stochastic outputs;
within-cluster matrix ordering is stable in Julia and may differ from NumPy's
unspecified argsort tie order. Thirty-six recorded block partitions and Gaussian
primitive cases independently check random block correlations.

The six bar constructors are checked against 224 reference cases with scalar
accounting, explicit EWMA weight sums and batch replay. Standard, imbalance and
run bars now retain global tick counts and closing prices across batches;
imbalance histories persist. Information-driven bars expose the public
pre-emission threshold snapshots, including shared history-list references.
Expected-count windows of zero or `nothing` use the reference unbounded mean
fallback; fixed bars ignore that window. The run buy-proportion update uses the
imbalance window when the expected-count window is absent or zero. Twenty time
cases check exact boundaries, skipped empty periods and negative Unix epochs.
These contracts use finite prices, nonnegative volumes and chronological
millisecond timestamps; callers provide the required timezone conversion.

### Caller-estimator debiased importance and ensemble accuracy

The three-argument MDI+ overload accepts a caller-fitted ensemble through
`fit_estimator` and `tree_records`. Records supply one-based split features,
thresholds and bootstrap rows. Twenty-four cases independently check augmented
QR ridge fits, unique in-bag observations, out-of-bag population variances and
the public tree-use denominator. This is the released simplified stump-basis
variant; the full paper algorithm and existing empirical claims are outside
this validation.

The caller-estimator CPI overload accepts independent clone, fit, probability
and class callbacks plus supplied folds and Gaussian draws. Thirty-two cases
check every fitted and predicted array, conditional ridge knockoffs and pooled
clipped loss differences. The near-zero gate now uses population standard
deviation, matching the public implementation; the paired Student statistic
still uses sample deviation. Five direct cases include a counterexample at that
population/sample boundary. Native and sklearn forest training outputs are not
declared equivalent by these conditional arithmetic checks.

The caller-estimator bagging evaluator recomputes fresh training accuracies and
three member-weight schemes, preserving direct comparison of member outputs to
the external labels and assigning zero votes to the first class. Twelve fitted
reference cases and three exact rational cases validate this aggregation. The
released encoded-label mismatch for nonstandard labels and its feature-subset
prediction failure are retained explicitly. The default native tree convenience
does not replace the public entropy/balanced estimator in this contract, and
mutable Python cache reuse is not emulated by the fresh functional evaluator.

Fitted-predictor bootstrap accepts supplied one-based row plans, including
expanded plans for duplicated pandas index labels. Twenty-four cases cover
zero/one/multiple replicates and sample deviation. Forty-five theoretical
majority-vote cases agree with high-precision finite binomial sums within the
declared floating tolerance; the released `1-cdf` cancellation is preserved.

### PBO ranks, regimes and circular resampling

The five PBO interfaces are checked against 378 shared cases with explicit
partition enumeration, first-maximizing strategy selection, scalar dispersion,
linear quantiles, modal path weights and supplied circular block starts. First
maxima now preserve the public signed-zero behavior. Regime labels preserve the
observed NumPy behavior for missing values, including all-missing cutoffs.

Ordinal ranks accept `rank_order(scores)` for exact reference replay. Default
Julia stable sorting and NumPy's default sorting can rank tied scores differently;
one ten-strategy counterexample reverses the overfit decision. The default native
tie rule is preserved and this difference remains explicit. No tie-invariant
decision claim is made. Bagged PBO accepts one-based `bootstrap_rows` plans;
native and NumPy random streams are not declared seed-equivalent.

The volatility feature uses trailing observations, while the released regime
quantile cutoffs use the entire supplied sample. These checks do not certify
prefix-invariant regime labels or historical empirical selection claims.

### Leakage-aware caller-estimator search

The four-argument HPO overload accepts a caller sampler through `ask_trial` and
`tell_trial!`, the public suggestion and estimator-factory functions, and fit,
prediction or custom-score callbacks. It preserves all three public fold-skip
conditions, fresh unweighted fits, first completed maxima, failed NaN trials,
completed infinite scores, and finite-trial population moments. Reported best
parameters contain sampled settings; fixed factory settings remain separate.

Shared checks cover 144 objective cases, twelve recorded actual Optuna 4.9.0 TPE
studies with up to eighteen trials, six trial-state boundary cases and 32 DSR
gates. Proposal replay validates orchestration. It does not supply fresh native
TPE/CMA-ES proposals, so callers requesting those families must provide the
corresponding sampler. The earlier random-forest/random-search convenience
remains a separate interface. No search-efficiency or empirical performance
claim follows from these checks.

### Native component discovery and work ordering

`Components` provides native factories, canonical names, aliases, metadata and
lazy loader callbacks. Explicit accepted-keyword lists replace Python signature
introspection. The six public family catalogues retain their canonical names;
labeling, bet-sizing and portfolio registries start empty. `AbstractLabeler` and
`label` provide multiple-dispatch extension points. Registered feature-importance
components configure the already validated functional interfaces.

`ParallelWork` preserves the scalar partition equations and submitted-order
parallel mapping. General jobs may be collected in completion order; explicit
`ordered=true` restores submission order. Indexed table work requires an index
column and sorts the combined table. Concurrency is bounded by the Julia thread
count, and job definitions are preserved. Completion order and speedups are
properties of the current schedule and machine, not numerical guarantees.

The bar controller exposes all sixteen public routes with the public defaults
and the twelve-column schema. It processes caller-loaded tables and returns
completion status, fully processed row count and an error alongside the bars.
Thus empty or partial output does not certify successful input completion. CSV
loading and output writing remain explicit caller operations. Shared checks
cover 119 partition, mapping, registry, labeler and synthetic controller cases,
including all sixteen routes under two batch sizes. These native interfaces
preserve teaching contracts without recreating Python class/import scaffolding.
