using DataFrames
using PyCall
using Statistics
using PlotlyJS

@pyimport sklearn.datasets as Datasets
@pyimport sklearn.ensemble as Ensemble

"""
    Calculate feature importance using Mean-Decrease Impurity (MDI) method.

    Parameters:
    - classifier: Classifier for fit and prediction.
    - featureNames::Vector{String}: Vector of feature names.

    Returns:
    - DataFrame: DataFrame with feature importance scores.
"""
function feature_importance_mdi(
    classifier, featureNames::Vector{String}
)::DataFrame
    # Feature importance based on IS mean impurity reduction
    featureImportanceDataFrame = DataFrame([[] for _ in featureNames], featureNames)

    for tree in classifier.estimators_
        push!(featureImportanceDataFrame, tree.feature_importances_)
    end

    featureImportanceDataFrame .= ifelse.(featureImportanceDataFrame .== 0, missing, featureImportanceDataFrame)

    means = mean.(skipmissing.(eachcol(featureImportanceDataFrame)))
    stds = std.(skipmissing.(eachcol(featureImportanceDataFrame))) .* nrow(featureImportanceDataFrame) ^ -0.5

    featureImportances = means ./ sum(means)

    result = DataFrame(FeatureName=featureNames, FeatureImportance=featureImportances, Std=stds)

    return result
end
