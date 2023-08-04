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
    function: Implementation of MDA
    reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
    methodology: page 82 Mean-Decrease Accuracy section snippet 6.3 (snippet 8.3 2018)
"""
function featureImportanceMDA(
    classifier, # classifier for fit and prediction
    X::DataFrame, # features matrix
    y::DataFrame, # labels vector
    nSplits::Int64; # cross-validation n folds
    scoreSampleWeights::Union{Vector, Nothing}=nothing, # sample weights for score step
    trainSampleWeights::Union{Vector, Nothing}=nothing, # sample weights for train step 
)::DataFrame

    trainSampleWeights = isnothing(trainSampleWeights) ? ones(size(X)[1]) : trainSampleWeights
    scoreSampleWeights = isnothing(scoreSampleWeights) ? ones(size(X)[1]) : scoreSampleWeights

    cvGenerator = ModelSelection.KFold(n_splits=nSplits)    
    score0, score1 = DataFrame("value" => zeros(nSplits)), DataFrame([name => zeros(nSplits) for name in names(X)])

    for (i, (train, test)) ∈ enumerate(cvGenerator.split(X |> Matrix))
        println("fold $(i) start ...")

        train .+= 1 # Python indexing starts at 0
        test .+= 1 # Python indexing starts at 0

        X0, y0, sampleWeights0 = X[train, :], y[train, :], trainSampleWeights[train]
        X1, y1, sampleWeights1 = X[test, :], y[test, :], scoreSampleWeights[test]
            
        fit = classifier.fit(X0 |> Matrix, y0 |> Matrix |> vec, sample_weight=sampleWeights0)
        
        predictionProbability = fit.predict_proba(X1 |> Matrix)

        score0[i, 1] = -Metrics.log_loss(y1 |> Matrix, predictionProbability, sample_weight=sampleWeights1, labels=classifier.classes_)        
        
        for j ∈ names(X)
            X1_ = deepcopy(X1) 
            X1_[!,  j] = shuffle(X1_[!, j])

            predictionProbability = fit.predict_proba(X1_ |> Matrix)
            log_loss = Metrics.log_loss(y1 |> Matrix, predictionProbability, labels=classifier.classes_)
            score1[i, j] = -log_loss
        end
    end

    importances = DataFrame(.-Matrix(score1) .+ Matrix(score0), names(score1))
    importances = importances ./ .-score1

    means = mean.(eachcol(importances))
    stds = std.(eachcol(importances)) .* (nrow(importances)^(-0.5)) 

    return DataFrame("FeatureName"=> names(X),"Mean"=>means, "SandardDeviation"=>stds)
end 
