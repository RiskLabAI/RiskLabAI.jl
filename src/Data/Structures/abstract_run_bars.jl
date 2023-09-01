using DataFrames
using ResumableFunctions
using CSV
using Parameters
using Dates

@field_inherit AbstractRunBars{T<:Metric} AbstractRunBarsType{T} AbstractInformationDrivenBars{T} where {T<:Metric} begin
    cumulativeBuyθ::Float64
    cumulativeSellθ::Float64

    expectedBuyImbalance::Float64
    expectedSellImbalance::Float64

    expectedBuyTicksProportion::Float64

    buyTicksNumber::Int

    previousBarsNumberOfTicks::Vector{Int}
    previousBuyTickImbalances::Vector{Float64}
    previousSellTickImbalances::Vector{Float64}
    previousBarsBuyTicksProportions::Vector{Float64}

    analyseThresholds::Union{Vector,Nothing}
end

function AbstractRunBars{T}(;
    barType::String,
    windowSizeForExpectedNTicksEstimation::Union{Int,Nothing},
    expectedImbalanceWindow::Int,
    initialEstimateOfExpectedNTicksInBar::Float64,
    doesAnalyseThresholds::Bool
) where {T<:Metric}
    """
    AbstractRunBars constructor function
    :param barType: type of bar. e.g. expected_dollar_run_bars, fixed_tick_run_bars etc.
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

    cumulativeBuyθ = 0.0
    cumulativeSellθ = 0.0

    expectedBuyImbalance = NaN
    expectedSellImbalance = NaN

    expectedBuyTicksProportion = NaN

    buyTicksNumber = 0

    previousBarsNumberOfTicks = Int[]
    previousBuyTickImbalances = Float64[]
    previousSellTickImbalances = Float64[]
    previousBarsBuyTicksProportions = Float64[]

    if doesAnalyseThresholds
        analyseThresholds = []
    else
        analyseThresholds = nothing
    end

    return (
        abstractInformationDrivenBars...,
        cumulativeBuyθ=cumulativeBuyθ,
        cumulativeSellθ=cumulativeSellθ, expectedBuyImbalance=expectedBuyImbalance,
        expectedSellImbalance=expectedSellImbalance,
        expectedBuyTicksProportion=expectedBuyTicksProportion, buyTicksNumber=buyTicksNumber, previousBarsNumberOfTicks=previousBarsNumberOfTicks,
        previousBuyTickImbalances=previousBuyTickImbalances,
        previousSellTickImbalances=previousSellTickImbalances,
        previousBarsBuyTicksProportions=previousBarsBuyTicksProportions,
        analyseThresholds=analyseThresholds,
    )
end

function constructBarsFromData(
    abstractRunBars::AbstractRunBarsType{T};
    data
)::Vector{Vector{Union{Int,Float64,DateTime}}} where {T<:Metric}
    """
    The function is used to construct bars from input ticks data.
    :param data: tabular data that contains date_time, price, and volume columns
    :return: constructed bars
    """
    barsList = Vector[]

    tickCounter = abstractRunBars.tickCounter

    begin
        for tickData ∈ eachrow(data)
            tickCounter += 1
            (dateTime, price, volume) = tickData

            signedTick = tickRule(abstractRunBars, price)

            updateBaseFields(abstractRunBars, price, signedTick, volume)

            # calculate imbalance
            imbalance = imbalanceAtTick(T, price, signedTick, volume)


            if imbalance > 0
                append!(abstractRunBars.previousBuyTickImbalances, imbalance)
                abstractRunBars.cumulativeBuyθ += imbalance
                abstractRunBars.buyTicksNumber += 1

            elseif imbalance < 0
                append!(abstractRunBars.previousSellTickImbalances, -imbalance)
                abstractRunBars.cumulativeSellθ += -imbalance

            end



            warmUp = isnan(abstractRunBars.expectedBuyImbalance) || isnan(abstractRunBars.expectedSellImbalance)
            # warmUp = isnan.([abstractRunBars.expectedBuyImbalance,
            #                  abstractRunBars.expectedSellImbalance]) |> any

            if length(barsList) == 0 && warmUp
                abstractRunBars.expectedBuyImbalance = ewmaExpectedImbalance(
                    abstractRunBars,
                    abstractRunBars.previousBuyTickImbalances,
                    abstractRunBars.expectedImbalanceWindow,
                    warmUp=warmUp
                )

                abstractRunBars.expectedSellImbalance = ewmaExpectedImbalance(
                    abstractRunBars,
                    abstractRunBars.previousSellTickImbalances,
                    abstractRunBars.expectedImbalanceWindow,
                    warmUp=warmUp
                )

                if !(isnan(abstractRunBars.expectedBuyImbalance) || isnan(abstractRunBars.expectedSellImbalance))
                    abstractRunBars.expectedBuyTicksProportion = abstractRunBars.buyTicksNumber / abstractRunBars.cumulativeTicks
                end
            end

            # # update dateTime in barsThresholdsStatistics
            # if !isnothing(abstractRunBars.analyseThresholds)
            #     runBarsStatistics[TIMESTAMP] = dateTime
            #     append!(
            #         abstractRunBars.analyseThresholds,
            #         runBarsStatistics
            #     )
            # end


            maxProportion::Float64 = max(
                abstractRunBars.expectedImbalanceWindow * abstractRunBars.expectedBuyTicksProportion, abstractRunBars.expectedSellImbalance * (1 - abstractRunBars.expectedBuyTicksProportion)
            )
            threshold = abstractRunBars.expectedTicksNumber * maxProportion

            isConstructionConditionMet = barConstructionCondition(abstractRunBars, threshold)
            if isConstructionConditionMet

                nextBar = constructNextBar(
                    abstractRunBars,
                    dateTime,
                    tickCounter,
                    price,
                    abstractRunBars.highPrice,
                    abstractRunBars.lowPrice,
                    threshold,
                )

                push!(barsList, nextBar)

                append!(abstractRunBars.previousBarsNumberOfTicks, abstractRunBars.cumulativeTicks)
                append!(abstractRunBars.previousBarsBuyTicksProportions,
                    abstractRunBars.buyTicksNumber / abstractRunBars.cumulativeTicks)


                # update expected number of ticks based on formed bars
                abstractRunBars.expectedTicksNumber = expectedNumberOfTicks(abstractRunBars)

                # update buy ticks proportions
                abstractRunBars.expectedBuyTicksProportion = ewma(
                    last(abstractRunBars.previousBarsBuyTicksProportions, abstractRunBars.windowSizeForExpectedNTicksEstimation),
                    abstractRunBars.windowSizeForExpectedNTicksEstimation,
                )[end]

                abstractRunBars.expectedBuyImbalance = ewmaExpectedImbalance(
                    abstractRunBars,
                    abstractRunBars.previousBuyTickImbalances,
                    abstractRunBars.expectedImbalanceWindow
                )

                abstractRunBars.expectedSellImbalance = ewmaExpectedImbalance(
                    abstractRunBars,
                    abstractRunBars.previousSellTickImbalances,
                    abstractRunBars.expectedImbalanceWindow
                )

                resetCachedFields(abstractRunBars)
            end
        end
    end


    return barsList
end

function barConstructionCondition(
    abstractRunBars::AbstractRunBarsType,
    threshold::Float64,
)::Bool
    """
    Compute the condition of whether next bar should sample with current and previous tick datas or not.
    :return: whether next bar should form with current and previous tick datas or not.
    """

    maxProportion::Float64 = max(
        abstractRunBars.expectedImbalanceWindow * abstractRunBars.expectedBuyTicksProportion, abstractRunBars.expectedSellImbalance * (1 - abstractRunBars.expectedBuyTicksProportion)
    )

    if !isnan(maxProportion)
        maxθ = max(
            abstractRunBars.cumulativeBuyθ,
            abstractRunBars.cumulativeSellθ
        )

        conditionIsMet = maxθ > threshold
        return conditionIsMet
    else
        return false
    end
end

function resetCachedFields(
    abstractRunBars::AbstractRunBarsType, # abstract run bars structure to reset its fields. 
)
    """
    This function are used (directly or override) by all concrete or abstract subtypes. The function is used to reset cached fields in bars construction process when next bar is sampled.
    :return:
    """

    invoke(
        resetCachedFields,
        Tuple{AbstractBarsType},
        abstractRunBars
    )

    abstractRunBars.cumulativeBuyθ = 0
    abstractRunBars.cumulativeSellθ = 0
    abstractRunBars.buyTicksNumber = 0
end