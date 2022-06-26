using DataFrames

include("AbstractBars.jl")

include("Constants.jl") # Constants

struct StandardBars 

    parent::AbstractBar # parent AbstractBar structure
    threshold::Int # threshold to sample
end

function StandardBars(; metric::String,
        threshold::Int=50000,
        batchSize::Int=20000000)

    parent = AbstractBar(
        metric=metric,
        batchSize=batchSize,
    )

    StandardBars(
        parent,
        threshold,
    )
end

function resetBarsProperties(standardBars::StandardBars)

    standardBars.parent.openPrice = NaN
    standardBars.parent.highPrice = -Inf
    standardBars.parent.lowPrice = +Inf

    standardBars.parent.cumulativeStatistics = Dict(
        CUMULATIVE_TICKS => 0,
        CUMULATIVE_DOLLAR_VALUE => 0.0,
        CUMULATIVE_VOLUME => 0.0,
        CUMULATIVE_BUY_VOLUME => 0.0
    )
end

function extractBarsFromData(
    standardBars::StandardBars;
    data::Union{Vector,Tuple,Array,Matrix})::Vector

    barsList = []

    parent = standardBars.parent  
    
    for row ∈ eachrow(data)
        dateTime = row[1]
        price = float(row[2])
        volume = float(row[3])
        
        dollarValue = price * volume

        parent.nTicks += 1

        # Incomplete from this line
        signedTick = applyTickRule(parent, price)
        
        if isnan(parent.openPrice)
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
        if parent.cumulativeStatistics[parent.metric] >= standardBars.threshold  
            constructBarsWithParameter(
                parent,
                dateTime,
                price,
                parent.highPrice,
                parent.lowPrice,
                barsList
            )

            # reset bars properties
            resetBarsProperties(standardBars)
        end

    end

    return barsList
end

function constructDollarBars( 
        inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
        threshold::Int=70_000_000,
        batchSize::Int=20_000_000,
        verbose::Bool=true,
        toCSV::Bool=false,
        outputPath::Union{Nothing,String}=nothing)

    bars = StandardBars(
        metric=CUMULATIVE_DOLLAR_VALUE,
        threshold=threshold,
        batchSize=batchSize
    )

    dollarBars = runOnBatches(
        bars,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose,
        toCSV=toCSV,
        outputPath=outputPath
    )

    return dollarBars
end

function constructVolumeBars(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    threshold::Int=70_000_000,
    batchSize::Int=20_000_000,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)

    bars = StandardBars(
        metric=CUMULATIVE_VOLUME,
        threshold=threshold,
        batchSize=batchSize
    )

    volumeBars = runOnBatches(
        bars,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose,
        toCSV=toCSV,
        outputPath=outputPath
    )

    return volumeBars
end

function constructTickBars(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    threshold::Int=70_000_000,
    batchSize::Int=20_000_000,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)

    bars = StandardBars(
        metric=CUMULATIVE_TICKS,
        threshold=threshold,
        batchSize=batchSize
    )

    tickBars = runOnBatches(
        bars,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose,
        toCSV=toCSV,
        outputPath=outputPath
    )

    return tickBars
end
