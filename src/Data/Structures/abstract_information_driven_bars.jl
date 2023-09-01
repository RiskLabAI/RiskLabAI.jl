using DataFrames
using ResumableFunctions
using CSV
using Parameters
using Dates

@field_inherit AbstractInformationDrivenBars{T<:Metric} AbstractInformationDrivenBarsType{T} AbstractBars begin
    expectedTicksNumber::Float64
    expectedImbalanceWindow::Int
    windowSizeForExpectedNTicksEstimation::Union{Int, Nothing} 
end

#todo: the list of parameters in the comments section is snake case but in the function signature it is camel case
function AbstractInformationDrivenBars{T}(;
    barType::String,
    windowSizeForExpectedNTicksEstimation::Union{Int, Nothing},
    initialEstimateOfExpectedNTicksInBar::Float64,
    expectedImbalanceWindow::Int,
    ) where {T<:Metric}
    """
    AbstractInformationDrivenBars constructor function
    :param barType: type of bar. e.g. expected_dollar_imbalance_bars, fixed_tick_run_bars etc.
    :param windowSizeForExpectedNTicksEstimation: window size used to estimate number of ticks expectation
    :param initialEstimateOfExpectedNTicksInBar: initial estimate of number of ticks expectation window size
    :param expectedImbalanceWindow: window size used to estimate imbalance expectation
    """

    abstractBars = AbstractBars(barType)
    
    expectedTicksNumber = initialEstimateOfExpectedNTicksInBar
    expectedImbalanceWindow = expectedImbalanceWindow

    return (
        abstractBars...,
        expectedTicksNumber = initialEstimateOfExpectedNTicksInBar,
        expectedImbalanceWindow = expectedImbalanceWindow,
        windowSizeForExpectedNTicksEstimation=windowSizeForExpectedNTicksEstimation,
    )
end

function ewmaExpectedImbalance(
    abstractInformationDrivenBars::AbstractInformationDrivenBarsType,
    array::Union{SubArray, Vector{Float64}},
    window::Int;
    warmUp::Bool=false
    )::Float64
    """
    Calculates expected imbalance (2P[b_t=1]-1) using EWMA as defined on page 29 of Advances in Financial Machine Learning.
    :param array: imbalances list
    :param window: EWMA window for expectation calculation
    :param warmUp: whether warm up period passed or not
    :return: expected_imbalance: 2P[b_t=1]-1 which approximated using EWMA expectation
    """

    if length(array) < abstractInformationDrivenBars.expectedTicksNumber && warmUp
        ewmaWindow = NaN
    else
        ewmaWindow = trunc(min(length(array), window))
    end

    if isnan(ewmaWindow)
        expectedImbalance = NaN
    else
        expectedImbalance = ewma(
            last(array, ewmaWindow),
            ewmaWindow
        )[end]
    end
    
    return expectedImbalance
end

#todo: what's the deal with the first argument?
#todo: the function seems quite trivial returning signedTick. why?
function imbalanceAtTick(
    ::Type{Tick},
    price::Float64,
    signedTick::Float64, 
    volume::Float64
    )::Float64
    """
    Calculate the imbalance at tick t (current tick) (θ_t) using tick data as defined on page 29 of Advances in Financial Machine Learning
    :param price: price of tick
    :param signed_tick: tick rule of current tick computed before
    :param volume: volume of current tick
    :return: imbalance: imbalance of current tick
    """ 
    return signedTick
end

function imbalanceAtTick(
    ::Type{Volume},
    price::Float64,
    signedTick::Float64, 
    volume::Float64
    )::Float64
    """
    Calculate the imbalance at tick t (current tick) (θ_t) using tick data as defined on page 29 of Advances in Financial Machine Learning
    :param price: price of tick
    :param signed_tick: tick rule of current tick computed before
    :param volume: volume of current tick
    :return: imbalance: imbalance of current tick
    """ 

    return signedTick * volume
end

function imbalanceAtTick(
    ::Type{Dollar},
    price::Float64,
    signedTick::Float64, 
    volume::Float64
    )::Float64
    """
    Calculate the imbalance at tick t (current tick) (θ_t) using tick data as defined on page 29 of Advances in Financial Machine Learning
    :param price: price of tick
    :param signed_tick: tick rule of current tick computed before
    :param volume: volume of current tick
    :return: imbalance: imbalance of current tick
    """ 

    return signedTick * volume * price
end