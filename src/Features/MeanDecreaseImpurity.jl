using DataFrames
using PyCall
using Statistics
using Missings

@pyimport sklearn.datasets as Datasets
@pyimport sklearn.ensemble as Ensemble

"""
    featureImportanceMDI

Calculate feature importance using Mean-Decrease Impurity (MDI) method.

Parameters:
- `classifier`: Classifier for fit and prediction.
- `featureNames::Vector{String}`: Vector of feature names.

Returns:
- `DataFrame`: DataFrame with feature importance scores.

The feature importance is calculated based on the mean impurity reduction,
which is represented by the formula:

.. math:: \text{Importance}(f) = \frac{1}{N} \sum_{i=1}^{N} I(f, t_i)

where:
- :math:`f` is the feature,
- :math:`N` is the number of trees in the forest,
- :math:`t_i` is the i-th tree in the forest,
- :math:`I(f, t_i)` is the impurity reduction of feature `f` in tree `t_i`.
"""
function featureImportanceMDI(
    classifier,
    featureNames::Vector{String}
)::DataFrame
    featureImportanceDataFrame = DataFrame([[] for _ in featureNames], featureNames)

    for tree in classifier.estimators_
        push!(featureImportanceDataFrame, tree.feature_importances_)
    end

    featureImportanceDataFrame .= ifelse.(featureImportanceDataFrame .== 0, missing, featureImportanceDataFrame)

    means = mean.(skipmissing.(eachcol(featureImportanceDataFrame)))
    stds = std.(skipmissing.(eachcol(featureImportanceDataFrame))) .* nrow(featureImportanceDataFrame) ^ -0.5

    featureImportances = means ./ sum(means)

    return DataFrame(FeatureName=featureNames, FeatureImportance=featureImportances, Std=stds)
end
