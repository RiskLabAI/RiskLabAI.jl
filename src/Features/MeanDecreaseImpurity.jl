using DataFrames
using PyCall
using Statistics
using PlotlyJS

@pyimport sklearn.datasets as Datasets
@pyimport sklearn.ensemble as Ensemble

"""
    function: Implementation of an ensemble MDI method
    reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
    methodology: page 79 Mean-Decrease Impurity section snippet 6.2 (snippet 8.2 2018)
"""
function featureImportanceMDI(
    classifier, # classifier for fit and prediction
    featureNames::Vector{String} # feature names
)::DataFrame
    # feature importance based on IS mean impurity reduction
    featureImportanceDataFrame = DataFrame([[] for _ = featureNames] , featureNames)

    for tree ∈ classifier.estimators_
        push!(featureImportanceDataFrame, tree.feature_importances_)
    end

    featureImportanceDataFrame .= ifelse.(featureImportanceDataFrame .== 0, missing, featureImportanceDataFrame)

    means = mean.(skipmissing.(eachcol(featureImportanceDataFrame)))
    stds = std.(skipmissing.(eachcol(featureImportanceDataFrame))) .* nrow(featureImportanceDataFrame) ^ -0.5
    
    featureImportances = means ./ sum(means)

    result = DataFrame(FeatureName=featureNames, FeatureImportance=featureImportances, Std=stds)

    return result
end 

