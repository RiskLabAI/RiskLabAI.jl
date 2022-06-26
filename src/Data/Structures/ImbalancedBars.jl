
using DataFrames

include("AbstractBars.jl")
include("utils/ewma.jl")

# Constants
include("Constants.jl")


struct EMAImbalanceBars

    parent::AbstractRunBars # parent AbstractRunBars structure
    nPreviousBars::Int # number of previous bar 

    # Imbalance specific hyper parameters
    minimumTicksNumberExpected::Int # minimum possible number of expected ticks.
    maximumTicksNumberExpected::Int # maximum possible number of expected ticks.

end

function EMAImbalanceBars(;
    metric::String,
    nPreviousBars::Int,
    imblanaceWindowSizeExpected::Int,
    nTicksExpectedInitialEstimate::Int,
    constraintsOnTicksNumberExpected::Vector{Float64},
    batchSize::Int,
    returnThresholds::Bool
)

    parent = AbstractRunBars(
        metric=metric,
        batchSize=batchSize,
        nPreviousBars=nPreviousBars,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate,
        returnThresholds=returnThresholds
    )

    if isnothing(constraintsOnTicksNumberExpected)
        minimumTicksNumberExpected = 0
        maximumTicksNumberExpected = typemax(Int)

    else
        minimumTicksNumberExpected = constraintsOnTicksNumberExpected[1]
        maximumTicksNumberExpected = constraintsOnTicksNumberExpected[2]

    end

    EMAImbalanceBars(
        parent,
        nPreviousBars,
        minimumTicksNumberExpected,
        maximumTicksNumberExpected,
    )
end


function calculateExpectedTicksNumber(ema_imbalanced_bars::EMAImbalanceBars)::Int
    prev_num_of_ticks = ema_imbalanced_bars.parent.imbalance_tick_statistics[TICKS_BAR_NUMBER]
    exp_num_ticks = ewma(prev_num_of_ticks[end - ema_imbalanced_bars.nPreviousBars:end], ema_imbalanced_bars.nPreviousBars)[end]

    return min(max(exp_num_ticks, ema_imbalanced_bars.minimumTicksNumberExpected), ema_imbalanced_bars.maximumTicksNumberExpected)
end


struct ConstImbalanceBars

    parent::AbstractImbalanceBars # parent abstractImbalanceBars structure

end

function ConstImbalanceBars(;
    metric::String, 
    batchSize::Int,
    imblanaceWindowSizeExpected::Int,
    nTicksExpectedInitialEstimate::Int,
    returnThresholds::Bool
)

    parent = AbstractImbalanceBars(
        metric=metric,
        batchSize=batchSize,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate,
        returnThresholds=returnThresholds
    )

    ConstImbalanceBars(
        parent,
    )
end

function calculateExpectedTicksNumber(const_imbalanced_bars::ConstImbalanceBars)::Int
    return const_imbalanced_bars.parent.thresholds[EXPECTED_TICKS_NUMBER]
end

function calcualteImbalanceBarsEMADollar(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPreviousBars::Int=3,
    imblanaceWindowSizeExpected::Int=10000,
    nTicksExpectedInitialEstimate::Int=20000,
    constraintsOnTicksNumberExpected::Vector{Float64}=nothing,
    batchSize::Int=20_000_000,
    returnThresholds::Bool=false,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)

    bars = EMAImbalancedBars(
        metric=DOLLAR_IMBALANCE,
        nPreviousBars=nPreviousBars,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate, 
        constraintsOnTicksNumberExpected=constraintsOnTicksNumberExpected,
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    imbalanceBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose, 
        toCSV=toCSV,
        outputPath=outputPath
    )

    return imbalanceBars, DataFrame(bars.parent.barsThresholdsStatistics)

end


function calcualteImbalanceBarsEMAVolume(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPreviousBars::Int=3,
    imblanaceWindowSizeExpected::Int=10000,
    nTicksExpectedInitialEstimate::Int=20000,
    constraintsOnTicksNumberExpected::Vector{Float64}=nothing,
    batchSize::Int=20_000_000,
    returnThresholds::Bool=false,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)

    bars = EMAImbalanceBars(
        metric=VOLUME_IMBALANCE,
        nPreviousBars=nPreviousBars,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate, 
        constraintsOnTicksNumberExpected=constraintsOnTicksNumberExpected,
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    imbalanceBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose, 
        toCSV=toCSV,
        outputPath=outputPath
    )


    return imbalanceBars, DataFrame(bars.parent.barsThresholdsStatistics)

end

function calcualteImbalanceBarsEMAVolume(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPreviousBars::Int=3,
    imblanaceWindowSizeExpected::Int=10000,
    nTicksExpectedInitialEstimate::Int=20000,
    constraintsOnTicksNumberExpected::Vector{Float64}=nothing,
    batchSize::Int=20_000_000,
    returnThresholds::Bool=false,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)

    bars = EMAImbalanceBars(
        metric=TICK_IMBALANCE,
        nPreviousBars=nPreviousBars,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate, 
        constraintsOnTicksNumberExpected=constraintsOnTicksNumberExpected,
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    imbalanceBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose, 
        toCSV=toCSV,
        outputPath=outputPath
    )


    return imbalanceBars, DataFrame(bars.parent.barsThresholdsStatistics)

end

function calcualteImbalanceBarsConstDollar(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPreviousBars::Int=3,
    imblanaceWindowSizeExpected::Int=10000,
    nTicksExpectedInitialEstimate::Int=20000,
    constraintsOnTicksNumberExpected::Vector{Float64}=nothing,
    batchSize::Int=20_000_000,
    returnThresholds::Bool=false,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)

    bars = ConstImbalanceBars(
        metric=DOLLAR_IMBALANCE,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate, 
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    imbalanceBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose, 
        toCSV=toCSV,
        outputPath=outputPath
    )


    return imbalanceBars, DataFrame(bars.parent.barsThresholdsStatistics)
end

function calcualteImbalanceBarsConstVolume(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPreviousBars::Int=3,
    imblanaceWindowSizeExpected::Int=10000,
    nTicksExpectedInitialEstimate::Int=20000,
    constraintsOnTicksNumberExpected::Vector{Float64}=nothing,
    batchSize::Int=20_000_000,
    returnThresholds::Bool=false,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)

    bars = ConstImbalanceBars(
        metric=VOLUME_IMBALANCE,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate, 
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    imbalanceBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose, 
        toCSV=toCSV,
        outputPath=outputPath
    )


    return imbalanceBars, DataFrame(bars.parent.barsThresholdsStatistics)
end

function calcualteImbalanceBarsConstTick(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPreviousBars::Int=3,
    imblanaceWindowSizeExpected::Int=10000,
    nTicksExpectedInitialEstimate::Int=20000,
    constraintsOnTicksNumberExpected::Vector{Float64}=nothing,
    batchSize::Int=20_000_000,
    returnThresholds::Bool=false,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)

    bars = ConstImbalanceBars(
        metric=TICK_IMBALANCE,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate, 
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    imbalanceBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose, 
        toCSV=toCSV,
        outputPath=outputPath
    )


    return imbalanceBars, DataFrame(bars.parent.barsThresholdsStatistics)
end
