using DataFrames

include("AbstractBars.jl")
include("utils/ewma.jl")

# Constants
include("Constants.jl")


struct EMARunBars

    parent::AbstractRunBars
    minimumTicksNumberExpected::Int
    maximumTicksNumberExpected::Int

end

function EMARunBars(;
    metric::String,
    nPrevoiusBars::Int,
    imblanaceWindowSizeExpected::Int,
    nTicksExpectedInitialEstimate::Int,
    constraintsOnTicksNumberExpected::Vector{Float64},
    batchSize::Int,
    returnThresholds::Bool
)

    parent = AbstractRunBars(
        metric=metric,
        batchSize=batchSize,
        nPrevoiusBars=nPrevoiusBars,
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

    EMARunBars(
        parent,
        minimumTicksNumberExpected,
        maximumTicksNumberExpected,
    )
end

function calculateExpectedTicksNumber(ema_run_bars::EMARunBars)::Int
    prev_num_of_ticks = ema_run_bars.parent.imbalance_tick_statistics[TICKS_BAR_NUMBER]
    exp_num_ticks = ewma(prev_num_of_ticks[end-ema_run_bars.nPrevoiusBars:end], ema_run_bars.nPrevoiusBars)[end]

    return min(max(exp_num_ticks, ema_run_bars.minimumTicksNumberExpected), ema_run_bars.maximumTicksNumberExpected)
end

struct ConstRunBars
    
    parent::AbstractRunBars # parent abstractImbalanceBars structure

end

function ConstRunBars(;
    metric::String,
    nPrevoiusBars::Int,
    imblanaceWindowSizeExpected::Int,
    nTicksExpectedInitialEstimate::Int,
    batchSize::Int,
    returnThresholds::Bool
)

    parent = AbstractRunBars(
        metric=metric,
        batchSize=batchSize,
        nPrevoiusBars=nPrevoiusBars,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate,
        returnThresholds=returnThresholds
    )

    ConstRunBars(
        parent
    )
end

function calculateExpectedTicksNumber(ema_run_bars::EMARunBars)::Int
    return ema_run_bars.parent.thresholds[EXPECTED_TICKS_NUMBER]
end

function calcualteRunBarsEMADollar(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPrevoiusBars::Int=3,
    imblanaceWindowSizeExpected::Int=10000,
    nTicksExpectedInitialEstimate::Int=20000,
    constraintsOnTicksNumberExpected::Vector{Float64}=nothing,
    batchSize::Int=20_000_000,
    returnThresholds::Bool=false,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)

    bars = EMARunBars(
        metric=DOLLAR_RUN,
        nPrevoiusBars=nPrevoiusBars,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate,
        constraintsOnTicksNumberExpected=constraintsOnTicksNumberExpected,
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    runBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose,
        toCSV=toCSV,
        outputPath=outputPath
    )


    return runBars, DataFrame(bars.parent.barsThresholdsStatistics)

end

function calcualteRunBarsEMAVolume(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPrevoiusBars::Int=3,
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
        metric=VOLUME_RUN,
        nPrevoiusBars=nPrevoiusBars,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate,
        constraintsOnTicksNumberExpected=constraintsOnTicksNumberExpected,
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    runBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose,
        toCSV=toCSV,
        outputPath=outputPath
    )


    return runBars, DataFrame(bars.parent.barsThresholdsStatistics)
end

function calcualteRunBarsEMATick(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPrevoiusBars::Int=3,
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
        metric=TICK_RUN,
        nPrevoiusBars=nPrevoiusBars,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate,
        constraintsOnTicksNumberExpected=constraintsOnTicksNumberExpected,
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    runBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose,
        toCSV=toCSV,
        outputPath=outputPath
    )


    return runBars, DataFrame(bars.parent.barsThresholdsStatistics)

end

function calcualteRunBarsConstDollar(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPrevoiusBars::Int=3,
    imblanaceWindowSizeExpected::Int=10000,
    nTicksExpectedInitialEstimate::Int=20000,
    batchSize::Int=20_000_000,
    returnThresholds::Bool=false,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)

    bars = ConstImbalanceBars(
        metric=DOLLAR_RUN,
        nPrevoiusBars=nPrevoiusBars,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate,
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    runBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose,
        toCSV=toCSV,
        outputPath=outputPath
    )


    return runBars, DataFrame(bars.parent.barsThresholdsStatistics)
end

function calcualteRunBarsConstVolume(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPrevoiusBars::Int=3,
    imblanaceWindowSizeExpected::Int=10000,
    nTicksExpectedInitialEstimate::Int=20000,
    batchSize::Int=20_000_000,
    returnThresholds::Bool=false,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)

    bars = ConstImbalanceBars(
        metric=VOLUME_RUN,
        nPrevoiusBars=nPrevoiusBars,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate,
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    runBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose,
        toCSV=toCSV,
        outputPath=outputPath
    )


    return runBars, DataFrame(bars.parent.barsThresholdsStatistics)
end

function calcualteRunBarsConstTick(
    inputPathOrDataFrame::Union{String,Vector{String},DataFrame};
    nPrevoiusBars::Int=3,
    imblanaceWindowSizeExpected::Int=10000,
    nTicksExpectedInitialEstimate::Int=20000,
    batchSize::Int=20_000_000,
    returnThresholds::Bool=false,
    verbose::Bool=true,
    toCSV::Bool=false,
    outputPath::Union{Nothing,String}=nothing
)
    
    bars = ConstImbalanceBars(
        metric=TICK_RUN,
        nPrevoiusBars=nPrevoiusBars,
        imblanaceWindowSizeExpected=imblanaceWindowSizeExpected,
        nTicksExpectedInitialEstimate=nTicksExpectedInitialEstimate,
        batchSize=batchSize,
        returnThresholds=returnThresholds
    )

    runBars = runOnBatches(
        bars.parent,
        inputPathOrDataFrame=inputPathOrDataFrame,
        verbose=verbose,
        toCSV=toCSV,
        outputPath=outputPath
    )


    return runBars, DataFrame(bars.parent.barsThresholdsStatistics)
end
