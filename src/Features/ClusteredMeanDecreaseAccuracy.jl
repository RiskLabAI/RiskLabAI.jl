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
function: Clustered feature importance MDA
reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
methodology: page 87 Clustered MDA section
"""
function clusteredFeatureImportanceMDA(
    classifier, # classifier for fit and prediction
    X::DataFrame, # features matrix
    y::DataFrame, # labels vector
    clusters, # clusters
    nSplits::Int64 # cross-validation n folds
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
            log_loss = Metrics.log_loss(y1 |> Matrix, predictionProbability, labels=classifier.classes_)
            score1[i, j] = -log_loss
        end
    end

    importances = .-score1 .+ score0
    importances = importances ./ .-score1

    importances[!, :importancesMean] = mean.(eachcol(importances))
    importances[!, :importancesStd] = std.(eachcol(importances))

    return importances
end 
