"""
    RiskLabAI.Validation

Cross-validation submodule, mirroring the Python `RiskLabAI.backtest.validation`
sub-package (López de Prado, AFML Ch. 7): standard K-Fold, Purged K-Fold with
embargo, Combinatorial Purged Cross-Validation (CPCV), and Walk-Forward.

The index-generating interfaces (`cv_split`, `backtest_paths`, `get_n_splits`)
include supplied-feature adaptive boundaries. `backtest_predictions` accepts
explicit estimator callbacks and preserves aligned observation weights.

Deliberate divergence: Python nests this under `backtest.validation`; the Julia
package exposes it as the top-level `RiskLabAI.Validation` submodule. The
sklearn-style `CrossValidator` ABC / factory / controller scaffolding is replaced
by plain structs + multiple dispatch.
"""
module Validation

include("CrossValidators.jl")
include("AdaptiveCombinatorialCV.jl")
include("BacktestPredictions.jl")
include("BaggedCombinatorialCV.jl")

# Estimator-driven scoring over the cross-validators (DecisionTree.jl backend).
include("CrossValScore.jl")

# Grid / randomised hyper-parameter search over the cross-validators.
include("HyperParameterTuning.jl")
include("LeakageAwareSearch.jl")
include("SampleWeightedPipeline.jl")
include("PipelineHyperparameterSearch.jl")

# Path-level Bagged / Adaptive CPCV PBO (Arian–Norouzi–Seco 2024). Loaded after the
# Backtest submodule (these call its CSCV PBO primitives via `..Backtest`).
include("PathBaggedCPCV.jl")
include("PathAdaptiveCPCV.jl")

export
    KFoldCV,
    PurgedKFoldCV,
    CombinatorialPurgedCV,
    AdaptiveCombinatorialPurgedCV,
    BaggedCombinatorialPurgedCV,
    WalkForwardCV,
    cv_split,
    get_n_splits,
    backtest_paths,
    backtest_predictions,
    filtered_training_indices_with_embargo,
    cross_val_score,
    grid_search_cv,
    random_search_cv,
    SampleWeightedPipeline,
    clone_pipeline,
    configure_pipeline!,
    fit_pipeline!,
    predict_pipeline,
    predict_pipeline_proba,
    pipeline_classes,
    pipeline_parameter_grid,
    clf_hyper_fit,
    leakage_aware_hpo,
    deflated_sharpe_gate,
    # path-level Bagged / Adaptive CPCV
    moving_block_bootstrap_indices,
    bagged_probability_of_backtest_overfitting,
    estimate_volatility_regimes,
    adaptive_probability_of_backtest_overfitting

end # module Validation
