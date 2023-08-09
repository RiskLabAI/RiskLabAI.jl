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
Clustered feature importance using Clustered MDA.

This function calculates the clustered feature importance using the Clustered MDA method based on the
methodology presented in De Prado (2020), Machine Learning for Asset Managers, page 87, Clustered MDA section.

Parameters:
- classifier: Classifier for fitting and prediction.
- X (DataFrame): Features matrix.
- y (DataFrame): Labels vector.
- clusters: Dictionary of feature clusters.
- nSplits (Int64): Number of cross-validation folds.

Returns:
- DataFrame: DataFrame containing clustered feature importances.
"""
function clusteredFeatureImportanceMDA(
    classifier, 
    X::DataFrame, 
    y::DataFrame, 
    clusters, 
    nSplits::Int64
)::DataFrame

    cvGenerator = ModelSelection.KFold(n_splits=nSplits)    
    score0, score1 = DataFrame("value" => zeros(nSplits)), DataFrame([name => zeros(nSplits) for name in names(X)])
    for (i, (train, test)) ∈ enumerate(cvGenerator.split(X |> Matrix))
        println("fold $(i) start ...")

        train .+= 1 # Python indexing starts at 0
        test .+= 1 # Python indexing starts at 0

        X0, y0 = X[train, :], y[train, :] 
        X1, y1 = X[test, :], y[test, :]

        fit = classifier.fit(X0 |> Matrix, y0 |> Matrix |> vec)

        predictionProbability = fit.predict_proba(X1 |> Matrix)
        score0[i, 1] = -Metrics.log_loss(y1 |> Matrix, predictionProbability, labels=classifier.classes_)        
        for j ∈ names(X)
            X1_ = deepcopy(X1) 
            for k ∈ clusters[j]
                X1_[:, k] = shuffle(X1_[:, k])
            end
            predictionProbability = fit.predict_proba(X1_ |> Matrix)
            logLoss = Metrics.log_loss(y1 |> Matrix, predictionProbability, labels=classifier.classes_)
            score1[i, j] = -logLoss
        end
    end

    importances = .-score1 .+ score0
    importances = importances ./ .-score1

    importances[!, :importancesMean] = mean.(eachcol(importances))
    importances[!, :importancesStd] = std.(eachcol(importances))

    return importances
end 
