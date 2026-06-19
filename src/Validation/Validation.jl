"""
    RiskLabAI.Validation

Cross-validation submodule, mirroring the Python `RiskLabAI.backtest.validation`
sub-package (López de Prado, AFML Ch. 7): standard K-Fold, Purged K-Fold with
embargo, Combinatorial Purged Cross-Validation (CPCV), and Walk-Forward.

This slice wires the **index-generating** logic (`cv_split`, `backtest_paths`,
`get_n_splits`). The estimator-driven `backtest_predictions` lands with the
cross-validation-scoring slice once the ML backend is wired.

Deliberate divergence: Python nests this under `backtest.validation`; the Julia
package exposes it as the top-level `RiskLabAI.Validation` submodule. The
sklearn-style `CrossValidator` ABC / factory / controller scaffolding is replaced
by plain structs + multiple dispatch.
"""
module Validation

include("CrossValidators.jl")

export
    KFoldCV,
    PurgedKFoldCV,
    CombinatorialPurgedCV,
    WalkForwardCV,
    cv_split,
    get_n_splits,
    backtest_paths

end # module Validation
