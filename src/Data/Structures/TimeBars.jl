using DataFrames

include("AbstractBars.jl")

include("Constants.jl") # Constants

struct TimeBars
    parent::AbstractBar 
    thresholdMapping::Dict
    resolution::String # bar resolution type (e.g: ['D', 'H', 'MIN', 'S'])
    nResolutionUnits::Int # resolution units number (e.g: [3D, 4H, 6MIN])
    threshold::Int # nResolutionUnits * thresholdMapping[resolution]

    timestamp # timestamp used in time bars
end


function TimeBars(; resolution::String,
        nResolutionUnits::Int,
        batchSize::Int=20000000)

    parent = AbstractBar(metric=nothing,
            batchSize=batchSize,)

    # seconds number in Day, Hour, Minute and Second
    thresholdMapping::Dict = Dict(
        "D" => 86400,
        "H" => 3600,
        "MIN" => 60,
        "S" => 1
    )

    @assert resolution ∈ thresholdMapping "$resolution resolution not implemented."

    threshold::Int = nResolutionUnits * thresholdMapping[resolution]

    TimeBars(
        parent,
        resolution,
        nResolutionUnits,
        threshold,
        nothing,
    )
end


function resetBarsProperties(timeBars::TimeBars)

    timeBars.parent.openPrice = nothing
    timeBars.parent.highPrice = -Inf
    timeBars.parent.lowPrice = +Inf

    timeBars.parent.cumulativeStatistics = Dict(
        CUMULATIVE_TICKS => 0,
        CUMULATIVE_DOLLAR_VALUE => 0,
        CUMULATIVE_VOLUME => 0,
        CUMULATIVE_BUY_VOLUME => 0
    )
end


function extractBarsFromData(
    timeBars::TimeBars;
    data::Union{Vector,Tuple,Array,Matrix})::Vector

    barsList = []

    parent = timeBars.parent

    for row ∈ data
        dateTime = row[1]
        parent.nTicks += 1

        price = float(row[2])

        volume = row[3]
        dollarValue = price * volume

        signedTick = applyTickRule(parent, price)

        timestamp_threshold = (floor(int(float(dateTime)) / timeBars.threshold) + 1) * timeBars.threshold  # Current tick boundary timestamp

        if isnothing(parent.openPrice)
            parent.openPrice = price
        end

        # Update high low prices
        parent.highPrice, parent.lowPrice = highPriceAndLowPriceUpdate(parent, price)

        # Calculations
        parent.cumulativeStatistics[CUMULATIVE_TICKS] += 1
        parent.cumulativeStatistics[CUMULATIVE_DOLLAR_VALUE] += dollarValue
        parent.cumulativeStatistics[CUMULATIVE_VOLUME] += volume

        if signedTick == 1
            parent.cumulativeStatistics[CUMULATIVE_BUY_VOLUME] += volume
        end

        # If threshold reached then take a sample
        if parent.cumulativeStatistics[parent.metric] >= standard_bars.threshold  # pylint: disable=eval-used
            barsList = constructBarsWithParameter(
                parent,
                dateTime,
                price,
                parent.highPrice,
                parent.lowPrice,
                barsList
            )

            # Reset cache
            resetBarsProperties(standard_bars)
        end

    end

    return barsList
end

function calculateTimeBars(;
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame},
    resolution::String="D",
    nResolutionUnits::Int=1,
    batchSize::Int=20_000_000,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=None
)

    bars = TimeBars(
        resolution=resolution,
        nResolutionUnits=nResolutionUnits,
        batchSize=batchSize
    )

    timeBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose,
        toCSV=toCSV,
        outputPath=outputPath
    )

    return timeBars

end
