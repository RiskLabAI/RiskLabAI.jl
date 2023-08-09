using DataFrames
using Statistics
using PlotlyJS
using Random

"""
Skip NaN values in an input array.

This function removes NaN values from the input array.

Parameters:
- array (Vector): Input array containing NaN values.

Returns:
- Vector: Array with NaN values removed.
"""
function skip_nan(
    array::Vector
)::Vector
    return filter(!isnan, array)
end

"""
Calculate Group Mean and Standard Deviation.

This function calculates the mean and standard deviation for each cluster of features.

Parameters:
- dataframe0 (DataFrame): Input dataframe.
- clusters: Clusters of features.

Returns:
- DataFrame: DataFrame containing cluster index, mean, and standard deviation for each cluster.
"""
function group_mean_std(
    dataframe0,
    clusters
)::DataFrame
    output = DataFrame([name => [] for name in ["ClusterIndex", "Mean", "StandardDeviation"]])
    for (clusterIndex, j) ∈ clusters
        dataframe1 = sum.(skip_nan.(eachcol(dataframe0[:, j])))
        push!(output, ["Cluster $clusterIndex", mean(dataframe1), std(dataframe1)])
    end
    return output
end

"""
Clustered feature importance using MDI.

This function calculates the clustered feature importance using the MDI (Mean Decrease in Impurity) method based on
the methodology presented in De Prado (2020), Machine Learning for Asset Managers, page 86, Clustered MDI section.

Parameters:
- classifier: Classifier for MDI calculation.
- featureNames: Names of features.
- clusters: Clusters of features.

Returns:
- DataFrame: DataFrame containing clustered feature importances.
"""
function clustered_feature_importance_MDI(
    classifier,
    featureNames,
    clusters
)::DataFrame
    dict0 = Dict(
        i => tree.feature_importances_ for (i, tree) ∈ enumerate(classifier.estimators_)
    )
    dataframe0 = DataFrame(;[Symbol(k)=>v for (k,v) in dict0]...) |> Matrix |> transpose |> DataFrame
    DataFrames.rename!(dataframe0, featureNames)
    
    for column_index ∈ 1:ncol(dataframe0)
        dataframe0[:, column_index] = replace(dataframe0[:, column_index], 0.0 => NaN)
    end

    importances = group_mean_std(dataframe0, clusters)
    importances[["Mean", "StandardDeviation"]] = importances[["Mean", "StandardDeviation"]] ./ sum(importances["Mean"])

    return importances
end
