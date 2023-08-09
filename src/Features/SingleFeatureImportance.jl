using DataFrames
using DataFramesMeta

using PyCall
using Statistics
using PlotlyJS
using TimeSeries
using Random

@pyimport sklearn.metrics as Metrics
@pyimport sklearn.ensemble as Ensemble
@pyimport sklearn.datasets as Datasets
@pyimport sklearn.metrics as Metrics
@pyimport sklearn.model_selection as ModelSelection

"""
    Implementation of SFI method.

    Parameters:
    - classifier: Classifier for fit and prediction.
    - X::DataFrame: Features matrix.
    - y::DataFrame: Labels vector.
    - nSplits::Int64: Cross-validation n folds.
    - scoreSampleWeights::Union{Vector, Nothing}: Sample weights for score step.
    - trainSampleWeights::Union{Vector, Nothing}: Sample weights for train step.
    - scoring::String: Classification prediction and true values scoring type.

    Returns:
    - DataFrame: DataFrame containing feature importances.
"""
function featureImportanceSFI(
    classifier,
    X::DataFrame,
    y::DataFrame,
    nSplits::Int64;
    scoreSampleWeights::Union{Vector, Nothing}=nothing,
    trainSampleWeights::Union{Vector, Nothing}=nothing,
    scoring::String="log_loss"
)::DataFrame

    trainSampleWeights = isnothing(trainSampleWeights) ? ones(size(X)[1]) : trainSampleWeights
    scoreSampleWeights = isnothing(scoreSampleWeights) ? ones(size(X)[1]) : scoreSampleWeights

    cvGenerator = ModelSelection.KFold(n_splits=nSplits)

    featureNames = names(X)
    importances = DataFrame([name => [] for name in ["FeatureName", "Mean", "StandardDeviation"]])
    for featureName in featureNames
        
        scores = []
        for (i, (train, test)) in enumerate(cvGenerator.split(X |> Matrix))
    
            train .+= 1 # Python indexing starts at 0
            test .+= 1 # Python indexing starts at 0
    
            X0, y0, sampleWeights0 = X[train, [featureName]], y[train, :], trainSampleWeights[train]
            X1, y1, sampleWeights1 = X[test, [featureName]], y[test, :], scoreSampleWeights[test]
            
            fit = classifier.fit(X0 |> Matrix, y0 |> Matrix |> vec, sample_weight=sampleWeights0)

            if scoring == "log_loss"
                predictionProbability = fit.predict_proba(X1 |> Matrix)
                score_ = -Metrics.log_loss(y1 |> Matrix, predictionProbability, sample_weight=sampleWeights1, labels=classifier.classes_)        
            
            elseif scoring == "accuracy"
                prediction = fit.predict(X1 |> Matrix)
                score_ = Metrics.accuracy_score(y1 |> Matrix, prediction, sample_weight=sampleWeights1)
            
            else
                throw("'$scoring' method not defined.")
            end
            append!(scores, score_)
        end

        push!(importances, [featureName, mean(scores), std(scores) * size(scores)[1] ^ -0.5])        
    end

    return importances
end
