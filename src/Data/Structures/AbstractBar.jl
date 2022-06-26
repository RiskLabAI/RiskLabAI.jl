# DataFrames Lib (Pandas like in python)
using DataFrames

# Julia Generator 
using ResumableFunctions

# Read CSV File 
using CSV

# Keyword Arguments and Default Arguments
using Parameters

# Date
using Dates

# Exponential Weighted Moving Average (EWMA)
include("utils/ewma.jl")

# Constants
include("Constants.jl")

"""
function: splits the dataframe into batches
reference: n/a
methodology: n/a
"""
function splitDataFrameInBatches(dataframe::DataFrame,  
        batchSize::Int)

    chunksGenerator = (dataframe[i:min(k + batchSize - 1, end)] for k in 1:batchSize:length(dataframe))
    return chunksGenerator
end

"""
structure: base struct containing the shared information between the standard and information
driven bars
reference: n/a
methodology: n/a
"""
mutable struct AbstractBar
    # Base properties
    metric::String # bar type to create. (e.g: "dollar_imbalance").
    batchSize::Int # rows count to read in each batch.
    previousTickRule::Int # rule to use to calculate the previous tick.
    # Cache properties
    openPrice::Float64 # open price of current bar
    previousPrice::Float64 # previous bar price
    closePrice::Float64 # close price of current bar 

    highPrice::Float64 # high price of current bar
    lowPrice::Float64 # low price of current bar

    cumulativeStatistics::Dict # cumulative statistics for bars  

    nTicks::Int  # tick number when bar was formed
    # runOnBatches properties
    flag::Bool  # the first batch doesn't use the cache so at the begining, flag set to false 
end

# AbstractBar Constructor
function AbstractBar(; metric::String,
        batchSize::Int=2_000_000,
        previousTickRule=0,
        openPrice=NaN,
        previousPrice=NaN,
        closePrice=NaN,
        highPrice=-Inf,
        lowPrice=Inf,
        cumulativeStatistics=Dict(),
        nTicks=0,
        flag=false)

    cumulativeStatistics = Dict(CUMULATIVE_TICKS => 0,
            CUMULATIVE_DOLLAR_VALUE => 0.0,
            CUMULATIVE_VOLUME => 0.0,
            CUMULATIVE_BUY_VOLUME => 0.0)

    return AbstractBar(metric,
            batchSize,
            previousTickRule,
            openPrice,
            previousPrice,
            closePrice,
            highPrice,
            lowPrice,
            cumulativeStatistics,
            nTicks,
            flag,)
end

function runOnBatches(abstractBars; # 
        inputPathOrDataFrame::Union{String,Vector{String},DataFrame},
        verbose::Bool=true,
        toCSV::Bool=false,
        outputPath::Union{String,Nothing}=nothing)
        
    if toCSV
        header = true  
        # clean output
        open(outputPath, "w") do io
        end
    end

    if verbose
        println("reading data ...")
    end

    finalBars = []
    columnNames = ["date_time",
            "tick_num",
            "open",
            "high",
            "low",
            "close",
            "volume",
            CUMULATIVE_BUY_VOLUME,
            CUMULATIVE_TICKS,
            CUMULATIVE_DOLLAR_VALUE]

    count = 0 # count batch number
    for batch ∈ readBatchIterator(inputPathOrDataFrame)
        if verbose
            println("batch:", count)
        end
        # construct financial data structure from the batch data
        barsListInBatch = constructDataStructure(abstractBars, data=batch)

        if toCSV
            dataframe = DataFrame(barsListInBatch, columnNames)
            CSV.write(outputPath, dataframe, writeheader=header, append=true)
            header = false
        else
            # append result to finalBars  
            append!(finalBars, barsListInBatch)
        end

        # increase count
        count += 1
    end

    if verbose
        println("bars constructed ... \n")
    end

    if length(finalBars) != 0
        finalBarsDataFrame = DataFrame([[] for _ = columnNames], columnNames)
        for row ∈ finalBars
            push!(finalBarsDataFrame, row)
        end

        return finalBarsDataFrame
    end

    # dataframe is stored in .csv file
    return nothing
end;

@resumable function readBatchIterator(inputPathOrDataFrame::Union{String,Vector{String},
        DataFrame},)

    if inputPathOrDataFrame isa String
        checkCSVFormat(inputPathOrDataFrame)
        @yield DataFrame(CSV.File(inputPathOrDataFrame, dateformat="mm/dd/yyyy H:M:S.s", missingstring=["NAN",]))

    elseif inputPathOrDataFrame isa Vector{String}
        for inputPath ∈ inputPathOrDataFrame
            checkCSVFormat(inputPath)
        end

        for inputPath ∈ inputPathOrDataFrame
            @yield DataFrame(CSV.File(inputPath, dateformat="mm/dd/yyyy H:M:S.s", missingstring=["NAN",]))
        end

    elseif inputPathOrDataFrame isa DataFrame
        for batch in splitDataFrameInBatches(inputPathOrDataFrame, 2_000_000)
            @yield batch
        end

    else
        error(
            "inputPathOrDataFrame type is'nt String, Vector{String} or DataFrame"
        )
    end
end

function checkCSVFormat(inputPath::String)

    checkConditionsOnRow(DataFrame(CSV.File(inputPath, dateformat="mm/dd/yyyy H:M:S.s", missingstring=["NAN"]))[[1], :])
end

function constructDataStructure(abstractBars; 
        data::Union{Vector, Tuple, DataFrame})::Vector

    if data isa Vector
        values = data

    elseif data isa DataFrame
        values = Array(data)

    else
        error(
            "data type is'nt  Vector | Tuple | DataFrames.DataFrame"
        )
    end

    barsList = extractBarsFromData(abstractBars, data=values)

    abstractBars.parent.flag = true

    return barsList
end

function checkConditionsOnRow(testData::DataFrame)

    @assert size(testData)[2] == 3 "csv file format is [date_time, price, volume]."
    @assert typeof(testData[1, 1]) == DateTime "column 0 type is'nt DateTime format"
    @assert typeof(testData[1, 2]) == Float64 "price column type is'nt Float."
    @assert typeof(testData[1, 3]) != String "volume column type is'nt Int or Float."
end

function highPriceAndLowPriceUpdate(abstractBars::AbstractBar, price::Float64)::Tuple{Float64,Float64}

    if price > abstractBars.highPrice
        highPrice = price

    else
        highPrice = abstractBars.highPrice
    end

    if price < abstractBars.lowPrice
        lowPrice = price

    else
        lowPrice = abstractBars.lowPrice
    end

    return (highPrice, lowPrice)
end

function constructBarsWithParameter(
    abstractBars::AbstractBar,
    dateTime::DateTime,
    price::Float64,
    highPrice::Float64,
    lowPrice::Float64,
    extractedBarsList::Vector{Any})


    # bar properties
    openPrice = abstractBars.openPrice
    highPrice = max(highPrice, openPrice)
    lowPrice = min(lowPrice, openPrice)
    closePrice = price

    volume = abstractBars.cumulativeStatistics[CUMULATIVE_VOLUME]
    cumulativeBuyVolume = abstractBars.cumulativeStatistics[CUMULATIVE_BUY_VOLUME]
    cumulativeTicks = abstractBars.cumulativeStatistics[CUMULATIVE_TICKS]
    cumulativeDollarValue = abstractBars.cumulativeStatistics[CUMULATIVE_DOLLAR_VALUE]

    # Update bars
    row = [
        dateTime
        abstractBars.nTicks
        openPrice
        highPrice
        lowPrice
        closePrice
        volume
        cumulativeBuyVolume
        cumulativeTicks
        cumulativeDollarValue
    ]

    append!(extractedBarsList, [row])
end


function applyTickRule(abstractBars::AbstractBar, price::Float64)::Int
                
    if isnan(abstractBars.previousPrice)
        tickDifference = 0

    else
        tickDifference = price - abstractBars.previousPrice
    end

    if tickDifference != 0
        signedTick = sign(tickDifference)
        abstractBars.previousTickRule = signedTick

    else
        signedTick = abstractBars.previousTickRule
    end

    # update previous price
    abstractBars.previousPrice = price

    return signedTick
end

function calculateImbalance(abstractBars::AbstractBar, 
        price::Float64, signedTick::Int, volume::Float64)::Float64

    if abstractBars.metric == TICK_IMBALANCE || abstractBars.metric == TICK_RUN
        imbalance = signedTick

    elseif abstractBars.metric == DOLLAR_IMBALANCE || abstractBars.metric == DOLLAR_RUN
        imbalance = signedTick*volume*price

    elseif abstractBars.metric == VOLUME_IMBALANCE || abstractBars.metric == VOLUME_RUN
        imbalance = signedTick*volume

    else
        error("Unknown metric, possible values are tick/dollar/volume imbalance/run")

    end

    return imbalance
end

mutable struct AbstractImbalanceBars

    parent::AbstractBar # parent abstractbar structure instance
    imblanaceWindowSizeExpected::Int # window size to estimate imbalance expected 
    thresholds::Dict # thresholds for each metric
    imbalanceTickStatistics::Dict # tick imbalance statistics
    barsThresholdsStatistics::Union{Vector,Nothing} # thresholds used in imbalanced bar construction

end

function AbstractImbalanceBars(;
    metric::String,
    batchSize::Int,
    imblanaceWindowSizeExpected::Int,
    nTicksExpectedInitialEstimate::Int,
    returnThresholds::Bool
)

    parent = AbstractBar(metric=metric, batchSize=batchSize)

    # theresholds statistics
    thresholds = Dict(
        CUMULATIVE_Θ => 0.0,
        EXPECTED_IMBALANCE => NaN,
        EXPECTED_TICKS_NUMBER:nTicksExpectedInitialEstimate
    )

    # statistics of number of ticks of previous bars and previous tick imbalances 
    imbalanceTickStatistics = Dict(
        TICKS_BAR_NUMBER:[],
        IMBALANCE_VECTOR:[]
    )

    barsThresholdsStatistics = returnThresholds ? [] : nothing

    return AbstractImbalanceBars(
        parent,
        imblanaceWindowSizeExpected,
        thresholds,
        imbalanceTickStatistics,
        barsThresholdsStatistics
    )
end

function resetBarsProperties(abstractImbalanceBars::AbstractImbalanceBars)
                
    abstractImbalanceBars.parent.openPrice = 0
    abstractImbalanceBars.parent.highPrice = -Inf
    abstractImbalanceBars.parent.lowPrice = +Inf

    abstractImbalanceBars.parent.cumulativeStatistics = Dict(
        CUMULATIVE_TICKS => 0,
        CUMULATIVE_DOLLAR_VALUE => 0.0,
        CUMULATIVE_VOLUME => 0.0,
        CUMULATIVE_BUY_VOLUME => 0.0
    )

    abstractImbalanceBars.parent.thresholds[CUMULATIVE_Θ] = 0

end

function extractBarsFromData(
    abstractImbalanceBars;
    data::Tuple{Dict,DataFrame})::Vector

    parent = abstractImbalanceBars.parent

    # iterate rows in data
    barsList = []
    for row ∈ data
        dateTime = row[1]

        price = row[2]
        volume = row[3]
        dollarValue = price*volume

        parent.nTicks += 1

        signedTick = applyTickRule(parent, price)

        if isnothing(parent.openPrice)
            parent.openPrice = price
        end

        # update lowPrice and highPrice in parent abstractbar
        parent.highPrice, parent.lowPrice = highPriceAndLowPriceUpdate(parent, price)

        # calculate bars statistics
        parent.cumulativeStatistics[CUMULATIVE_TICKS] += 1
        parent.cumulativeStatistics[CUMULATIVE_DOLLAR_VALUE] += dollarValue
        parent.cumulativeStatistics[CUMULATIVE_VOLUME] += volume

        if signedTick == 1
            parent.cumulativeStatistics[CUMULATIVE_BUY_VOLUME] += volume
        end

        # calculate imbalance
        imbalance = calculateImbalance(parent, price, signedTick, volume)

        # append result to imbalance array
        append!(
            abstractImbalanceBars.imbalanceTickStatistics[IMBALANCE_VECTOR],
            imbalance
        )

        # update cum_data with calculated imbalance
        abstractImbalanceBars.thresholds[CUMULATIVE_Θ] += imbalance

        # when num_ticks_init passed, try to get expected imbalance 
        if length(barsList) != 0 && isnan(abstractImbalanceBars.thresholds[EXPECTED_IMBALANCE])
            abstractImbalanceBars.thresholds[EXPECTED_IMBALANCE] = calculateExpectedImbalance(
                abstractImbalanceBars,
                abstractImbalanceBars.imbalanceTickStatistics[IMBALANCE_VECTOR],
                abstractImbalanceBars.imblanaceWindowSizeExpected,
            )

        end

        # update dateTime in barsThresholdsStatistics
        if isnothing(abstractImbalanceBars.barsThresholdsStatistics)
            abstractImbalanceBars.thresholds["timestamp"] = dateTime
            append!(
                abstractImbalanceBars.barsThresholdsStatistics,
                Dict(abstractImbalanceBars.thresholds)
            )

        end

        expression = !isnan(abstractImbalanceBars.thresholds[EXPECTED_IMBALANCE]) ? abs(abstractImbalanceBars.thresholds[CUMULATIVE_Θ]) > abstractImbalanceBars.thresholds[EXPECTED_TICKS_NUMBER]*abs(abstractImbalanceBars.thresholds[EXPECTED_IMBALANCE]) : false
        if expression
            constructBarsWithParameter(
                parent,
                dateTime,
                price,
                highPrice,
                lowPrice,
                barsList
            )
            merge!(
                abstractImbalanceBars.imbalanceTickStatistics[TICKS_BAR_NUMBER],
                parent.cumulativeStatistics[CUMULATIVE_TICKS]
            )

            # ticks number expected on previously formed bars
            abstractImbalanceBars.thresholds[EXPECTED_TICKS_NUMBER] = calculateExpectedTicksNumber(abstractImbalanceBars)

            # calculate expected imbalance
            abstractImbalanceBars.thresholds[EXPECTED_IMBALANCE] = calculateExpectedImbalance(
                abstractImbalanceBars,
                abstractImbalanceBars.imbalanceTickStatistics[IMBALANCE_VECTOR],
                imblanaceWindowSizeExpected,
            )

            # reset properties
            resetBarsProperties(abstractImbalanceBars)
        end
    end

    barsList
end

function calculateExpectedImbalance(
    abstractBars,
    array,
    windowSize::Int;
    isWarmUp=false)::Vector

    if length(array) < abstractBars.thresholds[EXPECTED_TICKS_NUMBER] && isWarmUp
        # no array to fill for ewma now
        ewmaWindow = NaN
    else
        ewmaWindow = min(length(array), windowSize)
        @assert typeof(ewmaWindow) == Int "Window size of EWMA is'nt Integer type"
    end

    if isnothing(ewmaWindow) || isnan(ewmaWindow) 
        imbalanceExpectedValue = NaN

    else
        imbalanceExpectedValue = ewma(
            array[end - ewmaWindow:end],
            window=ewmaWindow
        )[end]
    end

    return imbalanceExpectedValue
end

struct AbstractRunBars

    parent::AbstractBar # parent AbstractBar structure 
    nPreviousBars::Int # previous bars number
    imblanaceWindowSizeExpected::Int # window size used to approximate expected imbalance 
    thresholds::Dict # thresholds statistics  
    imbalanceTickStatistics::Dict # imbalance tick statistics
    barsThresholdsStatistics::Union{Vector,Nothing}
    isWarmUp::Bool

end


function AbstractRunBars(;
    metric::String,
    batchSize::Int,
    nPreviousBars::Int,
    imblanaceWindowSizeExpected::Int,
    nTicksExpectedInitialEstimate::Int,
    returnThresholds::Bool
)

    parent = AbstractBar(metric=metric, batchSize=batchSize)

    thresholds = Dict(
        CUMULATIVE_Θ_BUY => 0.0,
        CUMULATIVE_Θ_SELL => 0.0,
        EXPECTED_IMBALANCE_BUY => NaN,
        EXPECTED_IMBALANCE_SELL => NaN,
        EXPECTED_TICKS_NUMBER => nTicksExpectedInitialEstimate,
        EXPECTED_BUY_TICKS_PROPORTION => NaN,
        BUY_TICK_NUMBER:0.0
    )

    # number of ticks of previous bars and previous tick imbalances
    imbalanceTickStatstics = Dict(
        TICKS_BAR_NUMBER => [],
        IMBALANCE_VECTOR_BUY => [],
        IMBALANCE_VECTOR_SELL => [],
        BUY_TICKS_PROPORTION => []
    )

    # dicts array: {"timestamp": value, CUMULATIVE_Θ: value, EXPECTED_TICKS_NUMBER: value, "exp_imbalance": value}
    barsThresholdsStatistics = returnThresholds ? [] : nothing

    isWarmUp = false

    return AbstractRunBars(parent,
            nPreviousBars,
            imblanaceWindowSizeExpected,
            thresholds,
            imbalanceTickStatistics,
            barsThresholdsStatistics,
            isWarmUp)
end

function resetBarsProperties(abstractRunBars::AbstractRunBars)
                
    abstractRunBars.parent.highPrice = -Inf
    abstractRunBars.parent.lowPrice = +Inf
    abstractRunBars.parent.openPrice = 0

    abstractRunBars.parent.cumulativeStatistics = Dict(
        CUMULATIVE_TICKS => 0,
        CUMULATIVE_DOLLAR_VALUE => 0.0,
        CUMULATIVE_VOLUME => 0.0,
        CUMULATIVE_BUY_VOLUME => 0.0
    )

    (abstractRunBars.parent.thresholds[CUMULATIVE_Θ_BUY], abstractRunBars.parent.thresholds[CUMULATIVE_Θ_SELL], abstractRunBars.parent.thresholds[BUY_TICK_NUMBER]) = (0, 0, 0)
end



function extractBarsFromData(
    abstractRunBars::AbstractRunBars;
    data::Tuple{Vector,Vector})::Vector

    # iterate over rows
    barsList = []
    for row ∈ data
        date_time = row[1]

        base_imbalance_bars.parent.nTicks += 1

        price = row[2]
        volume = row[3]

        dollar_value = price*volume

        signed_tick = applyTickRule(base_imbalance_bars.parent, price)

        if isnothing(base_imbalance_bars.parent.openPrice)
            base_imbalance_bars.parent.openPrice = price
        end

        # Update high low prices
        base_imbalance_bars.parent.highPrice, base_imbalance_bars.parent.lowPrice = highPriceAndLowPriceUpdate(base_imbalance_bars.parent, price)

        # Bar statistics calculations
        base_imbalance_bars.parent.cumulativeStatistics[CUMULATIVE_TICKS] += 1
        base_imbalance_bars.parent.cumulativeStatistics[CUMULATIVE_DOLLAR_VALUE] += dollar_value
        base_imbalance_bars.parent.cumulativeStatistics[CUMULATIVE_VOLUME] += volume
        if signed_tick == 1
            base_imbalance_bars.parent.cumulativeStatistics[CUMULATIVE_BUY_VOLUME] += volume
        end

        # Imbalance calculations
        imbalance = calculateImbalance(abstractRunBars.parent, price, signed_tick, volume)

        if imbalance > 0
            append!(abstractRunBars.imbalanceTickStatistics[IMBALANCE_VECTOR_BUY], imbalance)

            abstractRunBars.thresholds[CUMULATIVE_Θ_BUY] += +imbalance
            abstractRunBars.thresholds[BUY_TICK_NUMBER] += 1

        elseif imbalance < 0
            append!(abstractRunBars.imbalanceTickStatistics[IMBALANCE_VECTOR_SELL], -imbalance)

            abstractRunBars.thresholds[CUMULATIVE_Θ_SELL] += -imbalance

        end

        abstractRunBars.isWarmUp = any(isnan.([
            abstractRunBars.thresholds[EXPECTED_IMBALANCE_BUY],
            abstractRunBars.thresholds[EXPECTED_IMBALANCE_SELL]
        ]))

        # calculate expected imbalance when ticks init number passed
        if !isnothing(barsList) && length(barsList) != 0 && abstractRunBars.isWarmUp
            abstractRunBars.thresholds[EXPECTED_IMBALANCE_BUY] = calculateExpectedImbalance(
                abstractRunBars,
                abstractRunBars.imbalanceTickStatistics[IMBALANCE_VECTOR_BUY],
                abstractRunBars.imblanaceWindowSizeExpected,
                isWarmUp=true
            )

            abstractRunBars.thresholds[EXPECTED_IMBALANCE_SELL] = calculateExpectedImbalance(
                abstractRunBars,
                abstractRunBars.imbalanceTickStatistics[IMBALANCE_VECTOR_SELL],
                abstractRunBars.imblanaceWindowSizeExpected,
                isWarmUp=true
            )

            if !any(isnan.([abstractRunBars.thresholds[EXPECTED_IMBALANCE_BUY], abstractRunBars.thresholds[EXPECTED_IMBALANCE_SELL]]))
                abstractRunBars.thresholds[EXPECTED_BUY_TICKS_PROPORTION] = abstractRunBars.thresholds[BUY_TICK_NUMBER] / abstractRunBars.cum_statistics[CUMULATIVE_TICKS]
            end
        end

        if !isnothing(abstractRunBars.barsThresholdsStatistics)
            abstractRunBars.thresholds["timestamp"] = dateTime
            append!(
                abstractRunBars.barsThresholdsStatistics,
                Dict(abstractRunBars.thresholds)
            )
        end

        # generate possible bar
        max_proportion = max(
            abstractRunBars.thresholds[EXPECTED_IMBALANCE_BUY]*abstractRunBars.thresholds[EXPECTED_BUY_TICKS_PROPORTION],
            abstractRunBars.thresholds[EXPECTED_IMBALANCE_SELL]*(1 - abstractRunBars.thresholds[EXPECTED_BUY_TICKS_PROPORTION])
        )

        max_theta = max(
            abstractRunBars.thresholds[CUMULATIVE_Θ_BUY],
            abstractRunBars.thresholds[CUMULATIVE_Θ_SELL]
        )

        expression = max_theta > abstractRunBars.thresholds[EXPECTED_TICKS_NUMBER]*max_proportion && !isnan(max_proportion)
        if expression

            constructBarsWithParameter(
                parent,
                dateTime,
                price,
                highPrice,
                lowPrice,
                barsList
            )



            merge!(
                abstractRunBars.imbalanceTickStatistics[TICKS_BAR_NUMBER],
                parent.cumulativeStatistics[CUMULATIVE_TICKS]
            )

            append!(
                abstractRunBars.imbalanceTickStatistics[BUY_TICKS_PROPORTION],
                abstractRunBars.thresholds[BUY_TICK_NUMBER] / parent.cumulativeStatistics[CUMULATIVE_TICKS]
            )

            # ticks number expected value calculate based on formed bars
            abstractRunBars.thresholds[EXPECTED_TICKS_NUMBER] = calculateExpectedTicksNumber(abstractRunBars)

            # buy ticks proportion expected  based on formed bars
            buyTicksProportionExpectedValue = ewma(
                abstractRunBars.imbalanceTickStatistics[BUY_TICKS_PROPORTION][end-abstractRunBars.nPreviousBars:end],
                window=abstractRunBars.nPreviousBars
            )[end]

            abstractRunBars.thresholds[EXPECTED_BUY_TICKS_PROPORTION] = buyTicksProportionExpectedValue

            # calculate expected imbalance
            abstractRunBars.thresholds[EXPECTED_IMBALANCE_BUY] = calculateExpectedImbalance(
                abstractRunBars,
                abstractRunBars.imbalanceTickStatistics[IMBALANCE_VECTOR_BUY],
                abstractRunBars.imblanaceWindowSizeExpected,
            )

            abstractRunBars.thresholds[EXPECTED_IMBALANCE_SELL] = calculateExpectedImbalance(
                abstractRunBars,
                abstractRunBars.imbalanceTickStatistics[EXPECTED_IMBALANCE_SELL],
                abstractRunBars.imblanaceWindowSizeExpected,
            )

            # reset bars properties
            resetBarsProperties(abstractRunBars)

        end
    end

    return barsList
end
