using DataFrames
using ResumableFunctions
using CSV
using Parameters
using Dates

mutable struct AbstractBars <: AbstractBarsType
    """
    Abstract structure that contains the base properties which are shared between the subtypes.
    This structure subtypes are as follows:
        1- AbstractImbalanceBars
        2- AbstractRunBars
        3- StandardBars
        4- TimeBars
    """

    # Base properties
    const barType::String
    tickCounter::Int

    # Cache properties
    previousTickPrice::Float64

    openPrice::Float64
    closePrice::Float64

    lowPrice::Float64
    highPrice::Float64

    previousTickRule::Float64
    cumulativeTicks::Float64
    cumulativeDollar::Float64
    cumulativeVolume::Float64
    cumulativeBuyVolume::Float64
    nTicksOnBarFormation::Int
end

function AbstractBars(barType::String)::NamedTuple
    """
    AbstractBars constructor function
    :param bar_type: type of bar. e.g. time_bars, expected_dollar_imbalance_bars, fixed_tick_run_bars, volume_standard_bars etc.
    """

    tickCounter = 0
    previousTickPrice = NaN

    openPrice = NaN
    closePrice = NaN

    lowPrice = NaN
    highPrice = NaN

    return (
        barType=barType,
        tickCounter=tickCounter,
        previousTickPrice=previousTickPrice,
        openPrice=openPrice,
        closePrice=closePrice,
        lowPrice=lowPrice,
        highPrice=highPrice,
        previousTickRule=0,
        cumulativeTicks=0,
        cumulativeDollar=0.0,
        cumulativeVolume=0,
        cumulativeBuyVolume=0,
        nTicksOnBarFormation=0,
    )
end

function updateBaseFields(abstractBars::AbstractBarsType, price::Float64, tickRule::Float64, volume::Float64)
    """
    Update the base fields (that all bars have them.) with price, tick rule and volume of current tick
    :param price: price of current tick
    :param tick_rule: tick rule of current tick computed before
    :param volume: volume of current tick
    :return:
    """

    dollarValue = price * volume

    abstractBars.openPrice = isnan(abstractBars.openPrice) ? price : abstractBars.openPrice
    abstractBars.highPrice, abstractBars.lowPrice = highAndLowPriceUpdate(abstractBars, price)
    abstractBars.cumulativeTicks += 1

    abstractBars.cumulativeDollar += dollarValue
    abstractBars.cumulativeVolume += volume

    if tickRule == 1
        abstractBars.cumulativeBuyVolume += volume
    end
end

function resetCachedFields(abstractBars::AbstractBarsType)
    """
    This function is used (directly or override) by all concrete or abstract subtypes. The function is used to reset cached fields in bars construction process when next bar is sampled.
    :return:
    """

    abstractBars.openPrice = NaN
    abstractBars.highPrice, abstractBars.lowPrice = -Inf, +Inf


    abstractBars.cumulativeTicks = 0
    abstractBars.cumulativeDollar = 0
    abstractBars.cumulativeVolume = 0
    abstractBars.cumulativeBuyVolume = 0
end

function tickRule(
    abstractBars::AbstractBarsType,
    price::Float64,
)::Float64
    """
    Compute the tick rule term as explained on page 29 of Advances in Financial Machine Learning
    :param price: price of current tick
    :return: tick rule
    """

    tickDifference = isnan(abstractBars.previousTickPrice) ? 0 : price - abstractBars.previousTickPrice

    if tickDifference != 0
        abstractBars.previousTickRule = signedTick = sign(tickDifference)
    else
        signedTick = abstractBars.previousTickRule
    end

    # update previous price
    abstractBars.previousTickPrice = price

    return signedTick
end

function highAndLowPriceUpdate(
    abstractBars::AbstractBarsType, # abstract bar struct to use to construct the financial data structure.
    price::Float64, # price to be replaced with old prices if they are lower or higher than the current price.
)::Tuple{Float64,Float64}
    """
    Update the high and low prices using the current tick price.
    :param price: price of current tick
    :return: updated high and low prices
    """

    highPrice = max(price, abstractBars.highPrice)
    lowPrice = min(price, abstractBars.lowPrice)

    return (highPrice, lowPrice)
end

function constructNextBar(
    abstractBars::AbstractBarsType, # abstract bar struct to use to construct the financial data structure.
    dateTime::Union{DateTime,String}, # bar timeframe 
    tickIndex::Int,
    price::Float64, # current price
    highPrice::Float64, # high price
    lowPrice::Float64, # low price
    threshold::Float64, # threshold
)::Vector{Union{DateTime,Int,Float64}}
    """
    sample next bar, given ticks data. the bar's fields are as follows:
        1- date_time
        2- open
        3- high
        4- low
        5- close
        6- cumulative_volume: total cumulative volume of to be constructed bar ticks
        7- cumulative_buy_volume: total cumulative buy volume of to be constructed bar ticks
        8- cumulative_ticks total cumulative ticks number of to be constructed bar ticks
        9- cumulative_dollar_value total cumulative dollar value (price * volume) of to be constructed bar ticks

    the bar will have appended to the total list of sampled bars.

    :param date_time: timestamp of the to be constructed bar
    :param tick_index:
    :param price: price of last tick of to be constructed bar (used as close price)
    :param high_price: highest price of ticks in the period of bar sampling process
    :param low_price: lowest price of ticks in the period of bar sampling process
    :return: sampled bar
    """

    openPrice = abstractBars.openPrice
    highPrice = max(highPrice, openPrice)
    lowPrice = min(lowPrice, openPrice)
    closePrice = price

    cumulativeVolume = abstractBars.cumulativeVolume
    cumulativeBuyVolume = abstractBars.cumulativeBuyVolume
    cumulativeSellVolume = cumulativeVolume - cumulativeBuyVolume
    cumulativeTicks = abstractBars.cumulativeTicks
    cumulativeDollarValue = abstractBars.cumulativeDollar

    nextBar = [
        dateTime,
        tickIndex,
        openPrice, highPrice, lowPrice, closePrice,
        cumulativeVolume, cumulativeBuyVolume, cumulativeSellVolume,
        cumulativeTicks,
        cumulativeDollarValue,
        threshold,
    ]

    return nextBar
end
