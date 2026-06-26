using DataFrames
using DataFramesMeta
using PyCall
using Statistics
using TimeSeries
using Random

@pyimport sklearn.metrics as Metrics
@pyimport sklearn.ensemble as Ensemble
@pyimport sklearn.datasets as Datasets
@pyimport sklearn.model_selection as ModelSelection

"""
    featureImportanceSingleFeature

Compute the importance of features using Single Feature Importance (SFI) method.

Parameters:
- `classifier`: Classifier for fit and prediction.
- `X::DataFrame`: Features matrix.
- `y::DataFrame`: Labels vector.
- `nSplits::Int64`: Cross-validation n folds.
- `scoreSampleWeights::Union{Vector, Nothing}`: Sample weights for score step.
- `trainSampleWeights::Union{Vector, Nothing}`: Sample weights for train step.
- `scoring::String`: Classification prediction and true values scoring type.

Returns:
- `DataFrame`: DataFrame containing feature importances with their mean and standard deviation.
"""
function featureImportanceSingleFeature(
    classifier,
    X::DataFrame,
    y::DataFrame,
    nSplits::Int64;
    scoreSampleWeights::Union{Vector, Nothing}=nothing,
    trainSampleWeights::Union{Vector, Nothing}=nothing,
    scoring::String="log_loss"
)::DataFrame

    trainSampleWeights = isnothing(trainSampleWeights) ? ones(nrow(X)) : trainSampleWeights
    scoreSampleWeights = isnothing(scoreSampleWeights) ? ones(nrow(X)) : scoreSampleWeights

    cvGenerator = ModelSelection.KFold(n_splits=nSplits)

    featureNames = names(X)
    importances = DataFrame(FeatureName=String[], Mean=Float64[], StandardDeviation=Float64[])
    
    for featureName in featureNames
        scores = []
        for (trainIndices, testIndices) in cvGenerator.split(Matrix(X))
            # Adjusting for 1-based Julia indexing
            trainIndices .+= 1
            testIndices .+= 1

            XTrainSubset, yTrainSubset, trainWeightsSubset = X[trainIndices, [featureName]], y[trainIndices, :], trainSampleWeights[trainIndices]
            XTestSubset, yTestSubset, scoreWeightsSubset = X[testIndices, [featureName]], y[testIndices, :], scoreSampleWeights[testIndices]
            
            fit = classifier.fit(Matrix(XTrainSubset), vec(Matrix(yTrainSubset)), sample_weight=trainWeightsSubset)

            if scoring == "log_loss"
                predictionProbability = fit.predict_proba(Matrix(XTestSubset))
                score = -Metrics.log_loss(Matrix(yTestSubset), predictionProbability, sample_weight=scoreWeightsSubset, labels=classifier.classes_)        
            elseif scoring == "accuracy"
                prediction = fit.predict(Matrix(XTestSubset))
                score = Metrics.accuracy_score(Matrix(yTestSubset), prediction, sample_weight=scoreWeightsSubset)
            else
                throw(ArgumentError("The scoring method '$scoring' is not defined."))
            end
            
            push!(scores, score)
        end

        push!(importances, (FeatureName=featureName, Mean=mean(scores), StandardDeviation=std(scores) / sqrt(length(scores))))        
    end

    return importances
end
