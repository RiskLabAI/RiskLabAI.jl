using DataFrames
using TimeSeries
using MLJ
using MLJBase

"""
    crossValidationScore(
        classifier::Supervised,
        assetsData::Dict{Symbol, TimeArray},
        assetsLabels::Dict{Symbol, TimeArray},
        sampleWeights::Dict{Symbol, Vector{Float64}},
        scoring::String = "Log Loss",
        times::Dict{Symbol, TimeArray} = nothing,
        crossValidationGenerator::PurgedKFoldStacked = nothing,
        numSplits::Int = nothing,
        percentEmbargo::Float64 = 0.0
    ) -> Dict{Symbol, Vector{Float64}}

Compute cross-validation scores using either Log Loss or Accuracy for multiple financial assets.

:param classifier: A classifier model from the MLJ package.
:type classifier: Supervised
:param assetsData: Dictionary of samples for different assets.
:type assetsData: Dict{Symbol, TimeArray}
:param assetsLabels: Dictionary of labels corresponding to samples for different assets.
:type assetsLabels: Dict{Symbol, TimeArray}
:param sampleWeights: Dictionary of sample weights for the classifier.
:type sampleWeights: Dict{Symbol, Vector{Float64}}
:param scoring: Scoring type (either "Log Loss" or "Accuracy").
:type scoring: String
:param times: Dictionary of entire observation times for multiple assets.
:type times: Dict{Symbol, TimeArray}
:param crossValidationGenerator: The `PurgedKFoldStacked` struct containing observation times and split information.
:type crossValidationGenerator: PurgedKFoldStacked
:param numSplits: The number of KFold splits.
:type numSplits: Int
:param percentEmbargo: Embargo size percentage divided by 100.
:type percentEmbargo: Float64

:return: A dictionary where keys are asset names and values are vectors of cross-validation scores.
:rtype: Dict{Symbol, Vector{Float64}}
"""

function crossValidationScore(
        classifier::Supervised,
        assetsData::Dict{Symbol, TimeArray},
        assetsLabels::Dict{Symbol, TimeArray},
        sampleWeights::Dict{Symbol, Vector{Float64}},
        scoring::String = "Log Loss",
        times::Dict{Symbol, TimeArray} = nothing,
        crossValidationGenerator::PurgedKFoldStacked = nothing,
        numSplits::Int = nothing,
        percentEmbargo::Float64 = 0.0
    )::Dict{Symbol, Vector{Float64}}

    if scoring ∉ ["Log Loss", "Accuracy"]
        error("Invalid scoring method.")
    end

    if isnothing(times)
        times = Dict()
        for asset in keys(assetsData)
            times[asset] = TimeArray(
                DataFrame(
                    startTime=timestamp(assetsData[asset]),
                    endTime=append!(timestamp(assetsData[asset])[2:end], [last(timestamp(assetsData[asset])) + diff(timestamp(assetsData[asset])[1:2])]),
                    timestamp=:startTime
                )
            )
        end
    end

    if isnothing(crossValidationGenerator)
        crossValidationGenerator = PurgedKFoldStacked(numSplits, times, percentEmbargo)
    end

    assetScores = Dict{Symbol, Vector{Float64}}()

    for asset in keys(assetsData)
        labelName = first(names(assetsLabels[asset]))

        data, labels = DataFrame(assetsData[asset]), DataFrame(assetsLabels[asset])
        select!(data, Not(:timestamp))
        select!(labels, Not(:timestamp))

        sample = hcat(data, labels)
        for column in names(sample)
            sample = coerce(sample, column => eltype(sample[!, column]) <: Number ? Continuous : Multiclass)
        end

        labels, data = unpack(sample, ==(labelName), colname -> true)
        machine = machine(classifier, data, labels)
        scores = Vector{Float64}()

        for (train, test) in purgedKFoldSplit(crossValidationGenerator, times[asset])
            fit!(machine, rows=train)

            if scoring == "Log Loss"
                predictions = predict(machine, data[test, :])
                score = mean(log_loss(predictions, labels[test]))
            else
                predictions = predict_mode(machine, data[test, :])
                score = mean(accuracy(predictions, labels[test], sampleWeights[test]))
            end

            push!(scores, score)
        end

        assetScores[asset] = scores
    end

    return assetScores
end
