using DataFrames
using Statistics
using PlotlyJS
using Random

"""
    function: skip nan values of input array
    reference: n/a
    methodology: n/a
"""
function skipnan(
    array::Vector # array to skip nan
)
    filter(!isnan, array)
end


"""
    function: Group Mean and Standard Deviation
    reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
    methodology: page 86 Clustered MDI section
"""
function groupMeanStd(
    dataframe0, # input dataframe
    clusters # clusters
)::DataFrame

    output = DataFrame([name => [] for name ∈ ["ClusterIndex", "Mean", "StandardDeviation"]])
    for (clusterIndex, j) ∈ clusters
        dataframe1 = sum.(skipnan.(eachcol(dataframe0[:, j])))
        push!(output, ["Cluster $clusterIndex", mean(dataframe1, ), std(dataframe1, )])
    end

    return output
end


"""
    function: Clustered feature importance MDI
    reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
    methodology: page 86 Clustered MDI section
"""
function clusteredFeatureImportanceMDI(
    classifier, # classifier for mdi
    featureNames, # feature names
    clusters # clusters
)::DataFrame
    dict0 = Dict(
        i => tree.feature_importances_ for (i, tree) ∈ enumerate(classifier.estimators_)
    )
    dataframe0 = DataFrame(;[Symbol(k)=>v for (k,v) in dict0]...) |> Matrix |> transpose |> DataFrame
    DataFrames.rename!(dataframe0, featureNames)
    
    for column_index ∈ 1:ncol(dataframe0)
        dataframe0[:, column_index] = replace(dataframe0[:, column_index], 0.0 => NaN)
    end

    importances = groupMeanStd(dataframe0, clusters)
    importances[["Mean", "StandardDeviation"]] = importances[["Mean", "StandardDeviation"]] ./ sum(importances["Mean"])

    importances
end

