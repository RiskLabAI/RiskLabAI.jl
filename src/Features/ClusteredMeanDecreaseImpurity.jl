using DataFrames
using Statistics
using Random

"""
    skipNaN(array::Vector)::Vector

Remove NaN values from an input array.

# Parameters
- `array::Vector`: Input array containing NaN values.

# Returns
- `Vector`: Array with NaN values removed.
"""
function skipNaN(
    array::Vector
)::Vector
    return filter(!isnan, array)
end

"""
    groupMeanStd(dataframe::DataFrame, clusters::Dict{Int, Vector{Symbol}})::DataFrame

Calculate group mean and standard deviation.

This function calculates the mean and standard deviation for each cluster of features.

# Parameters
- `dataframe::DataFrame`: Input dataframe.
- `clusters::Dict{Int, Vector{Symbol}}`: Clusters of features.

# Returns
- `DataFrame`: DataFrame containing cluster index, mean, and standard deviation for each cluster.
"""
function groupMeanStd(
    dataframe::DataFrame,
    clusters::Dict{Int, Vector{Symbol}}
)::DataFrame
    output = DataFrame(ClusterIndex=String[], Mean=Float64[], StandardDeviation=Float64[])
    for (clusterIndex, features) in clusters
        values = vcat(skipNaN.(eachcol(dataframe[:, features]))...)
        push!(output, ["Cluster $clusterIndex", mean(values), std(values)])
    end
    return output
end

"""
    clusteredFeatureImportanceMDI(
        classifier,
        featureNames::Vector{Symbol},
        clusters::Dict{Int, Vector{Symbol}}
    )::DataFrame

Clustered feature importance using MDI (Mean Decrease in Impurity).

This function calculates the clustered feature importance using the MDI method based on
the methodology presented in De Prado (2020), Machine Learning for Asset Managers, page 86, Clustered MDI section.

# Parameters
- `classifier`: Classifier for MDI calculation.
- `featureNames::Vector{Symbol}`: Names of features.
- `clusters::Dict{Int, Vector{Symbol}}`: Clusters of features.

# Returns
- `DataFrame`: DataFrame containing clustered feature importances.

# References
- De Prado, M. (2020), Machine Learning for Asset Managers, page 86, Clustered MDI section.
"""
function clusteredFeatureImportanceMDI(
        classifier,
        featureNames::Vector{Symbol},
        clusters::Dict{Int, Vector{Symbol}}
    )::DataFrame

    featureImportances = Dict(
        i => tree.feature_importances_ for (i, tree) in enumerate(classifier.estimators_)
    )
    dataframe = DataFrame(;[Symbol(k) => v for (k, v) in featureImportances]...)
    DataFrames.rename!(dataframe, featureNames)

    for columnIndex in 1:ncol(dataframe)
        dataframe[:, columnIndex] = replace(dataframe[:, columnIndex], 0.0 => NaN)
    end

    importances = groupMeanStd(dataframe, clusters)
    importances[!, [:Mean, :StandardDeviation]] .= importances[!, [:Mean, :StandardDeviation]] ./ sum(importances[!, :Mean])

    return importances
end
