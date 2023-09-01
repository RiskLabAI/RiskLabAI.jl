using DataFrames

# todo: maybe we should remove the following includes:?
# include("abstract_bars.jl")
# include("controller/data_structure_controller.jl")
# include("abstract_information_driven_bars.jl")
# include("abstract_run_bars.jl")
# include("utils/ewma.jl")
# include("utils/constants.jl")
# include("types.jl")

#todo done: I think the pre-name expected is taken from mlfinlab. wheat does it stand for? (expected moving average?) Maybe we can replace it with "Average"
@field_inherit ExpectedRunBars{T<:Metric} ExpectedRunBarsType{T} AbstractRunBars{T} where {T<:Metric} begin
    # Imbalance specific hyper parameters
    expectedTicksNumberLowerBound::Float64 # minimum possible number of expected ticks.
    expectedTicksNumberUpperBound::Float64 # maximum possible number of expected ticks.
end

function ExpectedRunBars{T}(;
    barType::String,
    windowSizeForExpectedNTicksEstimation::Int,
    expectedImbalanceWindow::Int,
    initialEstimateOfExpectedNTicksInBar::Float64,
    expectedTicksNumberBounds::Union{Tuple{Int,Int},Nothing},
    doesAnalyseThresholds::Bool
) where {T<:Metric}
    """
    ExpectedRunBars constructor function
    :param barType: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
    :param windowSizeForExpectedNTicksEstimation: window size used to estimate number of ticks expectation
    :param initialEstimateOfExpectedNTicksInBar: initial estimate of number of ticks expectation window size
    :param expectedImbalanceWindow: window size used to estimate imbalance expectation
    :param expectedTicksNumberBounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
    :param doesAnalyseThresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    """

    abstractRunBars = AbstractRunBars{T}(
        barType=barType,
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        doesAnalyseThresholds=doesAnalyseThresholds
    )

    if isnothing(expectedTicksNumberBounds)
        expectedTicksNumberLowerBound, expectedTicksNumberUpperBound = 0, typemax(Int)
    else
        expectedTicksNumberLowerBound, expectedTicksNumberUpperBound = expectedTicksNumberBounds
    end

    ExpectedRunBars{T}(
        values(abstractRunBars)...,
        expectedTicksNumberLowerBound,
        expectedTicksNumberUpperBound,
    )
end

function expectedNumberOfTicks(
    expectedRunBars::ExpectedRunBarsType
)::Float64
    """
    Calculate number of ticks expectation when new imbalance bar is sampled.

    :return: number of ticks expectation.
    """

    previousBarsNTicksList = expectedRunBars.previousBarsNumberOfTicks
    expectedTicksNumber = ewma(
        last(previousBarsNTicksList, expectedRunBars.windowSizeForExpectedNTicksEstimation),
        expectedRunBars.windowSizeForExpectedNTicksEstimation
    )[end]

    return min(max(expectedTicksNumber, expectedRunBars.expectedTicksNumberLowerBound), expectedRunBars.expectedTicksNumberUpperBound)
end

@field_inherit FixedRunBars{T<:Metric} FixedRunBarsType{T} AbstractRunBars{T} where {T<:Metric} begin end

function FixedRunBars{T}(;
    barType::String,
    windowSizeForExpectedNTicksEstimation::Union{Int,Nothing},
    expectedImbalanceWindow,
    initialEstimateOfExpectedNTicksInBar::Float64,
    doesAnalyseThresholds::Bool
) where {T<:Metric}
    """
    Constructor.

    :param barType: (str) Type of run bar to create. Example: "dollar_run".
    :param windowSizeForExpectedNTicksEstimation: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation).
    :param expectedImbalanceWindow: (int) Expected window used to estimate expected run.
    :param initialEstimateOfExpectedNTicksInBar: (int) Initial number of expected ticks.
    :param doesAnalyseThresholds: (bool) Flag to save  and return thresholds used to sample run bars.
    """

    abstractRunBars = AbstractRunBars{T}(
        barType=barType,
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        doesAnalyseThresholds=doesAnalyseThresholds
    )

    FixedRunBars{T}(
        values(abstractRunBars)...,
    )
end

function expectedNumberOfTicks(
    constRunBars::FixedRunBarsType
)::Int
    """
    Calculate number of ticks expectation when new imbalance bar is sampled.

    :return: number of ticks expectation.
    """
    return constRunBars.expectedTicksNumber
end
