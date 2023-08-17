using DataFrames
using DataFramesMeta
using ScikitLearn
using Statistics
using PyCall

@pyimport sklearn.metrics as Metrics
@pyimport sklearn.ensemble as Ensemble
@pyimport sklearn.tree as Tree
@pyimport sklearn.datasets as Datasets
@pyimport sklearn.model_selection as ModelSelection

"""
    clusteredFeatureImportanceMDA(
        classifier,
        X::DataFrame,
        y::DataFrame,
        clusters::Dict{Symbol, Vector{Symbol}},
        nSplits::Int
    )::DataFrame

Clustered feature importance using Clustered MDA.

This function calculates the clustered feature importance using the Clustered MDA method based on the
methodology presented in De Prado (2020), Machine Learning for Asset Managers, page 87, Clustered MDA section.

# Parameters
- `classifier`: Classifier for fitting and prediction.
- `X::DataFrame`: Features matrix.
- `y::DataFrame`: Labels vector.
- `clusters::Dict{Symbol, Vector{Symbol}}`: Dictionary of feature clusters.
- `nSplits::Int`: Number of cross-validation folds.

# Returns
- `DataFrame`: DataFrame containing clustered feature importances.

# References
- De Prado, M. (2020), Machine Learning for Asset Managers, page 87, Clustered MDA section.
"""
function clusteredFeatureImportanceMDA(
        classifier,
        X::DataFrame,
        y::DataFrame,
        clusters::Dict{Symbol, Vector{Symbol}},
        nSplits::Int
    )::DataFrame

    cvGenerator = ModelSelection.KFold(n_splits=nSplits)
    score0 = DataFrame("value" => zeros(nSplits))
    score1 = DataFrame([name => zeros(nSplits) for name in names(X)])

    for (i, (train, test)) in enumerate(cvGenerator.split(X |> Matrix))
        println("Fold $(i) start ...")

        X0, y0 = X[train .+ 1, :], y[train .+ 1, :]
        X1, y1 = X[test .+ 1, :], y[test .+ 1, :]

        fit = classifier.fit(X0 |> Matrix, y0 |> Matrix |> vec)

        predictionProbability = fit.predict_proba(X1 |> Matrix)
        score0[i, 1] = -Metrics.log_loss(y1 |> Matrix, predictionProbability, labels=classifier.classes_)

        for j in names(X)
            X1_ = deepcopy(X1)
            for k in clusters[j]
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
