using DataFrames
using DataFramesMeta
using PyCall
using Statistics
using PlotlyJS
using TimeSeries
using Random

@pyimport sklearn.metrics as Metrics
@pyimport sklearn.ensemble as Ensemble
@pyimport sklearn.tree as Tree
@pyimport sklearn.datasets as Datasets
@pyimport sklearn.model_selection as ModelSelection

"""
    Calculate feature importance using Mean-Decrease Accuracy (MDA) method.

    Parameters:
    - classifier: Classifier for fit and prediction.
    - X::DataFrame: Features matrix.
    - y::DataFrame: Labels vector.
    - nSplits::Int64: Number of cross-validation folds.
    - scoreSampleWeights::Union{Vector, Nothing}: Sample weights for score step.
    - trainSampleWeights::Union{Vector, Nothing}: Sample weights for train step.

    Returns:
    - DataFrame: DataFrame with feature importance scores.
"""
function feature_importance_mda(
    classifier, X::DataFrame, y::DataFrame, nSplits::Int64;
    scoreSampleWeights::Union{Vector, Nothing}=nothing,
    trainSampleWeights::Union{Vector, Nothing}=nothing
)::DataFrame

    trainSampleWeights = isnothing(trainSampleWeights) ? ones(size(X, 1)) : trainSampleWeights
    scoreSampleWeights = isnothing(scoreSampleWeights) ? ones(size(X, 1)) : scoreSampleWeights

    cvGenerator = ModelSelection.KFold(n_splits=nSplits)    
    score0, score1 = DataFrame("value" => zeros(nSplits)), DataFrame([name => zeros(nSplits) for name in names(X)])

    for (i, (train, test)) in enumerate(cvGenerator.split(Matrix(X)))
        println("fold $(i) start ...")

        train .+= 1  # Python indexing starts at 0
        test .+= 1   # Python indexing starts at 0

        X0, y0, sampleWeights0 = X[train, :], y[train, :], trainSampleWeights[train]
        X1, y1, sampleWeights1 = X[test, :], y[test, :], scoreSampleWeights[test]

        fit = classifier.fit(Matrix(X0), vec(Matrix(y0)), sample_weight=sampleWeights0)

        predictionProbability = fit.predict_proba(Matrix(X1))

        score0[i, 1] = -Metrics.log_loss(Matrix(y1), predictionProbability, sample_weight=sampleWeights1, labels=classifier.classes_)

        for j in names(X)
            X1_ = deepcopy(X1)
            X1_[!, j] = shuffle(X1_[!, j])

            predictionProbability = fit.predict_proba(Matrix(X1_))
            log_loss = Metrics.log_loss(Matrix(y1), predictionProbability, labels=classifier.classes_)
            score1[i, j] = -log_loss
        end
    end

    importances = DataFrame(-Matrix(score1) .+ Matrix(score0), names(score1))
    importances = importances ./ (-score1)

    means = mean.(eachcol(importances))
    stds = std.(eachcol(importances)) .* (size(importances, 1)^(-0.5))

    return DataFrame("FeatureName" => names(X), "Mean" => means, "StandardDeviation" => stds)
end
