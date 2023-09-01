using DataFrames

@field_inherit StandardBars{T<:Metric} StandardBarsType{T} AbstractBars begin
    threshold::Float64 # threshold to sample
end

function StandardBars{T}(;
    barType::String,
    threshold::Float64
) where {T<:Metric}
    """
    StandardBars constructor function
    :param barType: type of bar. e.g. dollar_standard_bars, tick_standard_bars etc.
    :param threshold: threshold that used to sampling process
    """

    abstractBars = AbstractBars(
        barType,
    )

    return StandardBars{T}(
        values(abstractBars)...,
        threshold,
    )
end

function constructBarsFromData(
    standardBars::StandardBars{T};
    data
)::Vector{Vector{Union{Int,Float64,DateTime}}} where {T<:Metric}
    """
    The function is used to construct bars from input ticks data.
    :param data: tabular data that contains date_time, price, and volume columns
    :return: constructed bars
    """
    barsList = Vector[]

    tickCounter = standardBars.tickCounter
    for row ∈ eachrow(data)
        tickCounter += 1
        (dateTime, price, volume) = row

        signedTick = tickRule(standardBars, price)
        updateBaseFields(standardBars, price, signedTick, volume)

        # If threshold reached then take a sample
        threshold = standardBars.threshold
        isConstructionConditionMet = barConstructionCondition(standardBars, threshold)
        if isConstructionConditionMet
            nextBar = constructNextBar(
                standardBars,
                dateTime,
                tickCounter,
                price,
                standardBars.highPrice,
                standardBars.lowPrice,
                threshold
            )

            push!(barsList, nextBar)

            # reset bars properties
            resetCachedFields(standardBars)
        end
    end

    return barsList
end

function barConstructionCondition(standardBars::StandardBars{Dollar}, threshold::Float64)::Bool
    """
    Compute the condition of whether next bar should sample with current and previous tick datas or not.
    :return: whether next bar should form with current and previous tick datas or not.
    """
    return standardBars.cumulativeDollar ≥ threshold
end

function barConstructionCondition(standardBars::StandardBars{Volume}, threshold::Float64)::Bool
    """
    Compute the condition of whether next bar should sample with current and previous tick datas or not.
    :return: whether next bar should form with current and previous tick datas or not.
    """
    return standardBars.cumulativeVolume ≥ threshold
end

function barConstructionCondition(standardBars::StandardBars{Tick}, threshold::Float64)::Bool
    """
    Compute the condition of whether next bar should sample with current and previous tick datas or not.
    :return: whether next bar should form with current and previous tick datas or not.
    """
    return standardBars.cumulativeTicks ≥ threshold
end