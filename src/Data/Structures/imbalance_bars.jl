using DataFrames

# include("abstract_bars.jl")
# include("controller/data_structure_controller.jl")
# include("abstract_information_driven_bars.jl")
# include("abstract_imbalance_bars.jl")
# include("utils/ewma.jl")
# include("utils/constants.jl")

@field_inherit ExpectedImbalanceBars{T<:Metric} ExpectedImbalanceBarsType{T} AbstractImbalanceBars{T} where {T<:Metric} begin
    # Imbalance specific hyper parameters
    expectedTicksNumberLowerBound::Float64 # minimum possible number of expected ticks.
    expectedTicksNumberUpperBound::Float64 # maximum possible number of expected ticks.
end

function ExpectedImbalanceBars{T}(;
    barType::String,
    windowSizeForExpectedNTicksEstimation::Int,
    expectedImbalanceWindow::Int,
    initialEstimateOfExpectedNTicksInBar::Float64,
    expectedTicksNumberBounds::Union{Tuple{Float64,Float64},Nothing},
    doesAnalyseThresholds::Bool
) where {T<:Metric}
    """
    ExpectedImbalanceBars constructor function
    :param barType: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
    :param windowSizeForExpectedNTicksEstimation: window size used to estimate number of ticks expectation
    :param initialEstimateOfExpectedNTicksInBar: initial estimate of number of ticks expectation window size
    :param expectedImbalanceWindow: window size used to estimate imbalance expectation
    :param expectedTicksNumberBounds lower and upper bound of possible number of expected ticks that used to force bars sampling convergence.
    :param doesAnalyseThresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    """

    abstractImbalanceBars = AbstractImbalanceBars{T}(
        barType=barType,
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        doesAnalyseThresholds=doesAnalyseThresholds
    )

    if isnothing(expectedTicksNumberBounds)
        expectedTicksNumberLowerBound, expectedTicksNumberUpperBound = 0, typemax(Float64)
    else
        expectedTicksNumberLowerBound, expectedTicksNumberUpperBound = expectedTicksNumberBounds
    end

    return ExpectedImbalanceBars{T}(
        values(abstractImbalanceBars)...,
        expectedTicksNumberLowerBound,
        expectedTicksNumberUpperBound,
    )
end

function expectedNumberOfTicks(
    expectedImbalanceBars::ExpectedImbalanceBarsType
)::Float64
    """
    Calculate number of ticks expectation when new imbalance bar is sampled.

    :return: number of ticks expectation.
    """

    expectedTicksNumber = ewma(
        last(expectedImbalanceBars.previousBarsNumberOfTicks, expectedImbalanceBars.windowSizeForExpectedNTicksEstimation),
        expectedImbalanceBars.windowSizeForExpectedNTicksEstimation
    )[end]

    return min(max(expectedTicksNumber, expectedImbalanceBars.expectedTicksNumberLowerBound), expectedImbalanceBars.expectedTicksNumberUpperBound)
end

#todo: this is from mlfinlab. I guess de prado does not define constance imbalance bars. where can we refer to type of bar to? is there any reference other than mlfinlab?
@field_inherit FixedImbalanceBars{T<:Metric} FixedImbalanceBarsType{T} AbstractImbalanceBars{T} where {T<:Metric} begin end

function FixedImbalanceBars{T}(;
    barType::String,
    windowSizeForExpectedNTicksEstimation::Union{Int,Nothing},
    expectedImbalanceWindow::Int,
    initialEstimateOfExpectedNTicksInBar::Float64,
    doesAnalyseThresholds::Bool
) where {T<:Metric}
    """
    FixedImbalanceBars constructor function
    :param barType: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
    :param windowSizeForExpectedNTicksEstimation: window size used to estimate number of ticks expectation
    :param initialEstimateOfExpectedNTicksInBar: initial estimate of number of ticks expectation window size
    :param expectedImbalanceWindow: window size used to estimate imbalance expectation
    :param doesAnalyseThresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    """

    abstractImbalanceBars = AbstractImbalanceBars{T}(
        barType=barType,
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
        expectedImbalanceWindow=expectedImbalanceWindow,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        doesAnalyseThresholds=doesAnalyseThresholds
    )

    FixedImbalanceBars{T}(
        values(abstractImbalanceBars)...,
    )
end

function expectedNumberOfTicks(
    constImbalanceBars::FixedImbalanceBarsType # abstract bar struct to use to calcualte expected number of ticks
)::Int
    """
    Calculate number of ticks expectation when new imbalance bar is sampled.

    :return: number of ticks expectation.
    """
    return constImbalanceBars.expectedTicksNumber
end
