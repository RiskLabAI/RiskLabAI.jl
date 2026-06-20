# Changelog

All notable changes to `RiskLabAI.jl` are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0: minor
versions may include breaking changes).

## [0.6.1] — 2026-06-20

### Changed

- Added `[compat]` entries for the standard-library dependencies (`Dates`,
  `LinearAlgebra`, `Random`, `Statistics`) so the package passes Julia General
  registry auto-merge. No source or API changes.

## [0.6.0] — 2026-06-20

Milestone release: `RiskLabAI.jl` is parity-complete with `RiskLabAI.py`, and the
companion [`Notebooks.jl`](https://github.com/RiskLabAI/Notebooks.jl) now ships a
complete set of seven flagship tutorials (portfolio construction, financial data
structures, fractional differentiation, triple-barrier labeling, cross-validation
& PBO, feature importance, and the Deep-BSDE PDE solver), each runnable
top-to-bottom against this package.

No changes to the package's public API or behaviour since 0.5.1; the minor bump
marks the completed parity + tutorial milestone.

## [0.5.1] — 2026-06-19

### Fixed

- `Optimization.hrp` now symmetrises the correlation-distance matrix before
  single-linkage clustering, so it accepts correlation matrices with the tiny
  floating-point asymmetry that real sample covariances always have (previously
  `Clustering.hclust` threw `ArgumentError: Distance matrix should be symmetric`).
  Added a regression test covering an asymmetric correlation input.

## [0.5.0] — 2026-06-19

First substantive release. The package was reconstructed from the ground up to
mirror the `RiskLabAI.py` public API in idiomatic Julia (`snake_case` functions,
`CapWords` types, sub-modules mirroring the Python sub-packages). Every
deterministic function is verified against a Python reference; stochastic and
machine-learning pieces are validated structurally. The parity ledger lives in
`PARITY.md`.

### Added

- **Data**
  - `Data.Structures` — standard, time, imbalance (expected/fixed) and run
    (expected/fixed) bars.
  - `Data.Differentiation` — fractional differentiation (standard + fixed-width).
  - `Data.Weights` — sample uniqueness and return/time-decay weighting.
  - `Data.Denoise` — Marčenko–Pastur denoising, `cov_to_corr`/`corr_to_cov`,
    optimal-portfolio helpers.
  - `Data.Labeling` — CUSUM filter, triple-barrier, meta-labeling, trend scanning.
  - `Data.Distance` — variation of information, mutual information, KL/cross-entropy.
  - `Data.SyntheticData` — `form_block_matrix`, `drift_volatility_burst`,
    `compute_log_returns`, `align_params_length` (exact) and the Heston–Merton
    regime-switching price simulators (behavioural).
- **Features** — Shannon/plug-in/Lempel–Ziv/Kontoyiannis entropy; Corwin–Schultz
  and Bekker–Parkinson microstructural estimators; ADF / (B)SADF structural
  breaks; feature importance: `orthogonal_features` + `calculate_weighted_tau`
  (exact) and `feature_importance_mdi`/`mda`/`sfi` + `get_test_dataset`
  (DecisionTree.jl backend).
- **Backtest** — backtest statistics, probabilistic Sharpe ratio, test-set
  overfitting, strategy risk, probability of backtest overfitting, synthetic
  backtesting, and bet sizing.
- **Optimization** — Hierarchical Risk Parity building blocks + `hrp()`, PCA
  hedging, and Nested Clustered Optimisation (`get_optimal_portfolio_weights`,
  `get_optimal_portfolio_weights_nco`).
- **Cluster** — Optimized Nested Clustering (`cluster_k_means_base`/`top`), an
  exact `silhouette_samples`, and random block-correlation generators.
- **Validation** — purged K-Fold (with embargo), Combinatorial Purged CV,
  Walk-Forward and standard K-Fold splitters; `cross_val_score`; grid/random
  hyper-parameter search (`grid_search_cv`/`random_search_cv`).
- **Ensemble** — theoretical bagging accuracy (exact) plus an empirical
  weighted-bagging evaluator and bootstrap accuracy.
- **Pde** — the four Deep-BSDE equations (HJB-LQ, Black–Scholes–Barenblatt,
  default-risk and different-rate pricing) with exact generators, and a neural
  Deep-BSDE solver (`solve_deep_bsde`) on the Lux.jl backend.

### Dependencies

- Added `Clustering`, `DecisionTree`, `Lux`, `Zygote` and `Optimisers`.

### Notes

- The Python `controller`, `core`/registry and `hpc` infrastructure layers are
  intentionally **not** ported; Julia's multiple dispatch and native
  `Distributed` replace them (see `PARITY.md` for the full list of deliberate
  divergences). Sample weights are unavailable in the DecisionTree.jl-backed
  importances, and the Deep-BSDE set-transformer/Monte-Carlo/FBSNN variants are
  research scaffolding that was not ported.

## [0.0.1]

- Initial package skeleton.

[0.5.1]: https://github.com/RiskLabAI/RiskLabAI.jl/releases/tag/v0.5.1
[0.5.0]: https://github.com/RiskLabAI/RiskLabAI.jl/releases/tag/v0.5.0
[0.0.1]: https://github.com/RiskLabAI/RiskLabAI.jl/releases/tag/v0.0.1
