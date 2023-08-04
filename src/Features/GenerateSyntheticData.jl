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
    function: Generating a set of informed, redundant and explanatory variables
    reference: De Prado, M. (2020) MACHINE LEARNING FOR ASSET MANAGERS
    methodology: page 77 A Few Caveats of p-Values section snippet 6.1 (snippet 8.7 2018)
"""
function getTestData(;
    nFeatures::Int=100, # total number of features
    nInformative::Int=25, # number of informative features
    nRedundant::Int=25, # number of redundant features
    nSamples::Int=10000, # number of sample to generate
    randomState::Int=1, # random state
    sigmaStd::Float64=0.0, # standard deviation of generation
)

    # generate a random dataset for a classiﬁcation problem
    Random.seed!(randomState);

    X, y = Datasets.make_classification(
        n_samples=nSamples,
        n_features=nFeatures-nRedundant,
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
