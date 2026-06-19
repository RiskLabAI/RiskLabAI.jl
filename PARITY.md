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

Port of the `backtest.validation` sub-package: the train/test index-generating
logic. Estimator-driven `backtest_predictions` is deferred to the
cross-validation-scoring slice (needs the ML backend).

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Standard K-Fold | `KFold` | `Validation.KFoldCV` | contiguous folds, optional shuffle; `cv_split(cv, n)` |
| Purged K-Fold + embargo | `PurgedKFold` | `Validation.PurgedKFoldCV` | **exact** purge + embargo index logic (verified vs pandas reference); `cv_split(cv)` |
| Combinatorial Purged CV | `CombinatorialPurged` | `Validation.CombinatorialPurgedCV` | **exact**; `C(n,k)` splits, `backtest_paths` assembles the `k·C/n` paths |
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

_(further submodules appended as they are wired)_
