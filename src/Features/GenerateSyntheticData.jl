using DataFrames
using DataFramesMeta
using PyCall
using Statistics
using PlotlyJS
using TimeSeries
using Random
using Distributions

@pyimport sklearn.datasets as Datasets

"""
    Generate a dataset with informed, redundant, and explanatory variables

    Parameters:
    - nFeatures::Int: Total number of features.
    - nInformative::Int: Number of informative features.
    - nRedundant::Int: Number of redundant features.
    - nSamples::Int: Number of samples to generate.
    - randomState::Int: Random state.
    - sigmaStd::Float64: Standard deviation of generation.

    Returns:
    - X::DataFrame: Features dataframe.
    - y::DataFrame: Target variable dataframe.
"""
function get_test_data(
    nFeatures::Int=100,
    nInformative::Int=25,
    nRedundant::Int=25,
    nSamples::Int=10000,
    randomState::Int=1,
    sigmaStd::Float64=0.0
)

    # Generate a random dataset for a classification problem
    Random.seed!(randomState)

    X, y = Datasets.make_classification(
        n_samples=nSamples,
        n_features=nFeatures - nRedundant,
        n_informative=nInformative,
        n_redundant=0,
        shuffle=false,
        random_state=randomState
    )

    columns = ["I_$i" for i ∈ 1:nInformative]
    append!(columns, ["N_$i" for i ∈ 1:(nFeatures - nInformative - nRedundant)])
    
    X = DataFrame(X, columns)
    y = DataFrame([y], :auto)
    i = Random.rand(1:nInformative, nRedundant)

    distribution = Normal(0.0, 0.1)
    for (k, j) ∈ enumerate(i)
        X[!, "R_$k"] = X[!, "I_$j"] .+ Random.rand(distribution, size(X)[1]) .* sigmaStd
    end

    return X, y
end
