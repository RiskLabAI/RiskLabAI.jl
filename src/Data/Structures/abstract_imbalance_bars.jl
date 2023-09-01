using DataFrames
using ResumableFunctions
using CSV
using TimerOutputs
using Parameters
using Dates
using ProfileView

@field_inherit AbstractImbalanceBars{T<:Metric} AbstractImbalanceBarsType{T} AbstractInformationDrivenBars{T} where {T<:Metric} begin
    cumulativeθ::Float64
    expectedImbalance::Float64

    # previous bars number of ticks list and previous tick imbalances list
    previousBarsNumberOfTicks::Vector{Int}
    previousTickImbalances::Vector{Float64}
    analyseThresholds::Union{Vector,Nothing}
end

function AbstractImbalanceBars{T}(;
    barType::String,
    windowSizeForExpectedNTicksEstimation::Union{Int,Nothing},
    expectedImbalanceWindow::Int,
    initialEstimateOfExpectedNTicksInBar::Float64,
    doesAnalyseThresholds::Bool
) where {T<:Metric}
    """
    AbstractImbalanceBars constructor function
    :param barType: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_imbalance_bars etc.
    :param windowSizeForExpectedNTicksEstimation: window size used to estimate number of ticks expectation
    :param initialEstimateOfExpectedNTicksInBar: initial estimate of number of ticks expectation window size
    :param expectedImbalanceWindow: window size used to estimate imbalance expectation
    :param doesAnalyseThresholds: whether return thresholds values (θ, number of ticks expectation, imbalance expectation) in a tabular format
    """

    abstractInformationDrivenBars = AbstractInformationDrivenBars{T}(
        barType=barType,
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
        initialEstimateOfExpectedNTicksInBar=initialEstimateOfExpectedNTicksInBar,
        expectedImbalanceWindow=expectedImbalanceWindow
    )

    cumulativeθ = 0
    expectedImbalance = NaN
    previousBarsNumberOfTicks = Int[]
    previousTickImbalances = Float64[]

    if doesAnalyseThresholds
        analyseThresholds = []
    else
        analyseThresholds = nothing
    end

    return (
        abstractInformationDrivenBars...,
        cumulativeθ=cumulativeθ,
        expectedImbalance=expectedImbalance,

        # previous bars number of ticks list and previous tick imbalances list
        previousBarsNumberOfTicks=previousBarsNumberOfTicks,
        previousTickImbalances=previousTickImbalances,
        analyseThresholds=analyseThresholds,
    )
end

function constructBarsFromData(
    abstractImbalanceBars::AbstractImbalanceBarsType{T};
    data
)::Vector{Vector{Union{Int,Float64,DateTime}}} where {T<:Metric}

    """
    The function is used to construct bars from input ticks data.
    :param data: tabular data that contains date_time, price, and volume columns
    :return: constructed bars
    """

    barsList = Vector[]
    tickCounter = abstractImbalanceBars.tickCounter

    nRow = size(data)[1]
    abstractImbalanceBars.previousTickImbalances = zeros(Float64, nRow)

    begin
        for (k, tickData) ∈ eachrow(data) |> enumerate
            tickCounter += 1
            (dateTime, price, volume) = tickData

            signedTick = tickRule(abstractImbalanceBars, price)
            updateBaseFields(abstractImbalanceBars, price, signedTick, volume)

            # calculate imbalance
            imbalance = imbalanceAtTick(T, price, signedTick, volume)

            abstractImbalanceBars.previousTickImbalances[k] = imbalance
            # push!(abstractImbalanceBars.previousTickImbalances, imbalance)
            abstractImbalanceBars.cumulativeθ += imbalance

            if isnan(abstractImbalanceBars.expectedImbalance)
                abstractImbalanceBars.expectedImbalance = ewmaExpectedImbalance(
                    abstractImbalanceBars,
                    view(abstractImbalanceBars.previousTickImbalances, 1:k),
                    abstractImbalanceBars.expectedImbalanceWindow,
                    warmUp=true
                )
            end

            threshold = abstractImbalanceBars.expectedTicksNumber * abs(abstractImbalanceBars.expectedImbalance)
            isConstructionConditionMet = barConstructionCondition(abstractImbalanceBars, threshold)
            if isConstructionConditionMet

                nextBar = constructNextBar(
                    abstractImbalanceBars,
                    dateTime,
                    tickCounter,
                    price,
                    abstractImbalanceBars.highPrice,
                    abstractImbalanceBars.lowPrice,
                    threshold,
                )

                push!(barsList, nextBar)

                push!(
                    abstractImbalanceBars.previousBarsNumberOfTicks,
                    abstractImbalanceBars.cumulativeTicks
                )

                # expectedNumberOfTicks function was implemented in sub struct 
                # update expected number of ticks based on formed bars
                abstractImbalanceBars.expectedTicksNumber = expectedNumberOfTicks(abstractImbalanceBars)

                # calculate expected imbalance
                abstractImbalanceBars.expectedImbalance = ewmaExpectedImbalance(
                    abstractImbalanceBars,
                    view(abstractImbalanceBars.previousTickImbalances, 1:k),
                    abstractImbalanceBars.expectedImbalanceWindow,
                    warmUp=true
                )

                # reset properties
                resetCachedFields(abstractImbalanceBars)
            end
        end
    end


    barsList
end

function barConstructionCondition(
    abstractImbalanceBars::AbstractImbalanceBarsType,
    threshold::Float64,
)::Bool
    """
    Compute the condition of whether next bar should sample with current and previous tick datas or not.
    :return: whether next bar should form with current and previous tick datas or not.
    """

    if !isnan(abstractImbalanceBars.expectedImbalance)
        cumulativeθ = abstractImbalanceBars.cumulativeθ
        # expectedTicksNumber = abstractImbalanceBars.expectedTicksNumber
        # expectedImbalance = abstractImbalanceBars.expectedImbalance

        conditionIsMet = abs(cumulativeθ) > threshold

        return conditionIsMet

    else
        return false
    end
end


function resetCachedFields(abstractImbalanceBars::AbstractImbalanceBarsType)
    """
    This function are used (directly or override) by all concrete or abstract subtypes. The function is used to reset cached fields in bars construction process when next bar is sampled.
    :return:
    """

    # call super reset cache fields
    invoke(
        resetCachedFields,
        Tuple{AbstractBarsType},
        abstractImbalanceBars
    )

    abstractImbalanceBars.cumulativeθ = 0
end
