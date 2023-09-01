using DataFrames
using Dates

@field_inherit TimeBars TimeBarsType AbstractBars begin
    resolutionToMiliSeconds::Dict{String,Int}
    resolutionType::String
    resolutionUnits::Int
    thresholdInSeconds::Int

    timestamp::Union{Real,Nothing}
    timestampThreshold::Real
end

# todo: the output of the signature is not determined
function TimeBars(;
    resolutionType::String,
    resolutionUnits::Int
)
    """
    TimeBars constructor function

    :param resolutionType: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S'].
    :param resolutionUnits: (int) Number of days, minutes, etc.
    """

    abstractBars = AbstractBars(
        "time",
    )

    # seconds number in Day, Hour, Minute and Second
    resolutionToMiliSeconds = Dict(
        "W" => 60 * 60 * 24 * 7 * 1000,
        "D" => 60 * 60 * 24 * 1000,
        "H" => 60 * 60 * 1000,
        "MIN" => 60 * 1000,
        "S" => 1 * 1000
    )

    @assert resolutionType ∈ keys(resolutionToMiliSeconds) "$resolutionType type isn't defined."

    thresholdInSeconds::Int = resolutionUnits * resolutionToMiliSeconds[resolutionType]

    TimeBars(
        abstractBars...,
        resolutionToMiliSeconds,
        resolutionType,
        resolutionUnits,
        thresholdInSeconds,
        nothing,
        NaN
    )
end

function constructBarsFromData(
    timeBars::TimeBars;
    data
)::Vector{Vector{Union{Int,Float64,DateTime}}} where {T<:Metric}

    """
    The function is used to construct bars from input ticks data.
    :param data: tabular data that contains date_time, price, and volume columns
    :return: constructed bars
    """
    barsList = Vector[]
    tickCounter = timeBars.tickCounter
    for tickData ∈ eachrow(data)
        tickCounter += 1
        (dateTime, price, volume) = tickData
        signedTick = tickRule(timeBars, price)
        updateBaseFields(timeBars, price, signedTick, volume)

        miliSeconds = Dates.value(convert(Millisecond, dateTime))

        timeBars.timestampThreshold = (floor(miliSeconds / timeBars.thresholdInSeconds) + 1) * timeBars.thresholdInSeconds  # Current tick boundary timestamp

        if isnothing(timeBars.timestamp)
            timeBars.timestamp = timeBars.timestampThreshold

        elseif barConstructionCondition(timeBars, timeBars.timestampThreshold)

            nextBar = constructNextBar(
                timeBars,
                dateTime,
                tickCounter,
                price,
                timeBars.highPrice,
                timeBars.lowPrice,
                timeBars.timestampThreshold,
            )

            push!(barsList, nextBar)

            resetCachedFields(timeBars)

            timeBars.timestamp = timeBars.timestampThreshold
        end

        timeBars.closePrice = price
    end

    return barsList
end

function barConstructionCondition(
    timeBars::TimeBarsType,
    threshold::Float64,
)::Bool
    """
    Compute the condition of whether next bar should sample with current and previous tick datas or not.
    :return: whether next bar should form with current and previous tick datas or not.
    """

    timeBars.timestamp < threshold
end