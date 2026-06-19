"""
    RiskLabAI.Ensemble

Ensemble-methods submodule, mirroring the Python `RiskLabAI.ensemble`
sub-package (López de Prado, AFML Ch. 6): the theoretical accuracy of a
majority-vote bagging classifier and an empirical weighted-bagging evaluator
(on the `DecisionTree.jl` backend).
"""
module Ensemble

include("BaggingAccuracy.jl")

export
    bagging_classifier_accuracy,
    fit_bagging,
    bagging_evaluate_schemes,
    calculate_bootstrap_accuracy

end # module Ensemble
