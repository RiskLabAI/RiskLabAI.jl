using Dates

"""
Function to show the progress bar.

Args:
    value (Int): Value of an event.
    endValue (Int): Length of that event.
    startTime (Int): Start time of the event.
    barLength (Int, optional): Length of the progress bar. Default is 20.

Reference:
    n/a

Methodology:
    n/a
"""
function showProgressBar(value,
                         endValue,
                         startTime,
                         barLength = 20)
    percent = Float64(value) / endValue

    if trunc(Int, round(percent * barLength) - 1) < 0
        progressValue = 0
    else
        progressValue = trunc(Int, round(percent * barLength) - 1)
    end
    arrow = string(repeat("-", progressValue), ">")
    spaces = repeat(" ", (barLength - length(arrow)))
    remaining = trunc(Int, ((Dates.value((Dates.now() - Dates.DateTime(1970, 1, 1, 00, 00, 00))) /
        1000 - startTime) / value) * (endValue - value) / 60)
    message = string(arrow, spaces)
    percent = trunc(Int, round(percent * 100))
    println("Completed: [$message] $percent% - $remaining minutes remaining.")
end

"""
Function to compute the exponential weighted moving average (EWMA), EWMA variance, and EWMA standard deviations.

Args:
    data (Array): Array of data.
    windowLength (Int, optional): Window for EWMA. Default is 100.

Reference:
    https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation

Methodology:
    n/a
"""
function computeEwma(data,
                     windowLength = 100)
    N = size(data)[1]
    ewmaMean = []
    ewmaVariance = []
    ewmaStd = []
    α = 2 / Float64(windowLength + 1)

    for i ∈ 1:N
        window = Array(data[1:i, 1])
        m = length(window)
        ω = (1 - α).^range(m - 1, step = -1, stop = 0)
        ewma = sum(ω .* window) / sum(ω)
        bias = sum(ω)^2 / (sum(ω)^2 - sum(ω .^ 2))
        var = bias * sum(ω .* (window .- ewma).^2) / sum(ω)
        std = sqrt(var)
        append!(ewmaMean, ewma)
        append!(ewmaVariance, var)
        append!(ewmaStd, std)
    end
    return ewmaMean, ewmaVariance, ewmaStd
end

"""
Function to group a dataframe based on a feature and then calculate thresholds.

Args:
    targetCol (Array): Target column of tick dataframe.
    tickExpectedInit (Int): Initial expected ticks.
    barSize (Float64): Initial expected size in each tick.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Page number: n/a

Methodology:
    Page number: n/a
"""
function groupAndCalculateThresholds(targetCol,
                                     tickExpectedInit,
                                     barSize)
    Δtimes, times = [], []
    timePrev, tickExpected, barExpectedValue = 0, tickExpectedInit, barSize
    N = size(targetCol)[1]
    θAbsolute, thresholds, θs, groupingId = zeros(N), zeros(N), zeros(N), zeros(N)
    θAbsolute[1], θCurrent = abs(targetCol[1]), targetCol[1]
    time = Dates.value((Dates.now() - Dates.DateTime(1970, 1, 1, 00, 00, 00))) / 1000
    groupingIdCurrent = 0

    for i ∈ 2:N
        θCurrent += targetCol[i]
        θs[i] = θCurrent
        thisθAbsolute = abs(θCurrent)
        θAbsolute[i] = thisθAbsolute
        threshold = tickExpected * barExpectedValue
        thresholds[i] = threshold
        groupingId[i] = groupingIdCurrent

        if thisθAbsolute ≥ threshold
            groupingIdCurrent += 1
            θCurrent = 0
            append!(Δtimes, Float64(i - timePrev))
            append!(times, i)
            timePrev = i
            tickExpected = ewma(Array(Δtimes), trunc(Int, length(Δtimes)))[1][end]
            barExpectedValue = abs(ewma(targetCol[1:i], trunc(Int, tickExpectedInit * 1))[1][end])
        end
    end
    return Δtimes, θAbsolute, thresholds, times, θs, groupingId
end

"""
Function to implement Information-Driven Bars.

Args:
    tickData (DataFrame): Dataframe of tick data.
    type (String, optional): Choose "tick", "volume" or "dollar" types. Default is "volume".
    tickExpectedInit (Int, optional): The value of the expected tick. Default is 2000.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Page number: 29

Methodology:
    Page number: 29
"""
function infoDrivenBar(tickData,
                       type::String = "volume",
                       tickExpectedInit = 2000)
    if type == "volume"
        inputData = tickData.volumelabeled
    elseif type == "tick"
        inputData = tickData.label
    elseif type == "dollar"
        inputData = tickData.dollars
    else
        println("Error: unknown type")
    end
    barExpectedValue = abs(mean(inputData))
    Δtimes, θAbsolute, thresholds, times, θs, groupingId = groupAndCalculateThresholds(inputData, tickExpectedInit, barExpectedValue)
    tickData[!,:groupingID] = groupingId
    dates = combine(DataFrames.groupby(tickData, :groupingID), :dates => first => :dates)
    tickDataGrouped = DataFrames.groupby(tickData, :groupingID)
    ohlcvDataframe = ohlcv(tickDataGrouped)
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates)
    colDrop = [:groupingID]
    ohlcvDataframe = select!(ohlcvDataframe, Not(colDrop))
    return ohlcvDataframe, θAbsolute, thresholds
end

"""
Function to take a grouped dataframe, combining and creating the new one with information about prices and volume.

Args:
    tickDataGrouped (GroupedDataFrame): Grouped dataframes.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: n/a
"""
function ohlcv(tickDataGrouped)
    ohlcvDataframe = combine(tickDataGrouped, :price => first => :Open,
                             :price => maximum => :High,
                             :price => minimum => :Low,
                             :price => last => :Close,
                             :size => sum => :Volume,
                             AsTable([:price, :size]) => x -> sum(x.price .* x.size) / sum(x.size),
                             :price => mean => :PriceMean,
                             :price => length => :TickCount)
    DataFrames.rename!(ohlcvDataframe, :price_size_function => :ValueOfTrades)
    ohlcvDataframe.PriceMeanLogReturn = log.(ohlcvDataframe.PriceMean) - log.(circshift(ohlcvDataframe.PriceMean, 1))
    ohlcvDataframe.PriceMeanLogReturn[1] = NaN
    return ohlcvDataframe
end

"""
Function to take a dataframe and generate a time bar dataframe.

Args:
    tickData (DataFrame): Dataframe of tick data.
    frequency (Int, optional): Frequency for rounding date time. Default is 5.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: n/a
"""
function timeBar(tickData,
                  frequency = 5)
    dates = tickData.dates
    datesCopy = copy(dates)
    tickData.dates = floor.(datesCopy, Dates.Minute(frequency))
    tickDataGrouped = DataFrames.groupby(tickData, :dates)
    ohlcvDataframe = ohlcv(tickDataGrouped)
    tickData.dates = dates
    return ohlcvDataframe
end

"""
Function to take a dataframe and generate a tick bar dataframe.

Args:
    tickData (DataFrame): Dataframe of tick data.
    tickPerBar (Int, optional): Number of ticks in each bar. Default is 10.
    numberBars (Int, optional): Number of bars. Default is nothing.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: n/a
"""
function tickBar(tickData,
                  tickPerBar = 10,
                  numberBars = nothing)
    if tickPerBar == nothing
        tickPerBar = floor(size(tickData)[1] / numberBars)
    end
    tickData[!, :groupingID] = [x ÷ tickPerBar for x in 0:size(tickData)[1] - 1]
    tickGrouped = copy(tickData)
    dates = combine(DataFrames.groupby(tickGrouped, :groupingID), :dates => first => :dates)
    tickDataGrouped = DataFrames.groupby(tickGrouped, :groupingID)
    ohlcvDataframe = ohlcv(tickDataGrouped)
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates)
    colDrop = [:groupingID]
    ohlcvDataframe = select!(ohlcvDataframe, Not(colDrop))
    return ohlcvDataframe
end

"""
Function to take a dataframe and generate a volume bar dataframe.

Args:
    tickData (DataFrame): Dataframe of tick data.
    volumePerBar (Int, optional): Volumes in each bar. Default is 10000.
    numberBars (Int, optional): Number of bars. Default is nothing.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: n/a
"""
function volumeBar(tickData,
                    volumePerBar = 10000,
                    numberBars = nothing)
    tickData[!, :volumecumulated] = cumsum(tickData.size)
    if volumePerBar == nothing
        totalVolume = tickData.volumecumulated[end]
        volumePerBar = totalVolume / numberBars
        volumePerBar = round(volumePerBar; sigdigits = 2)
    end
    tickData[!, :groupingID] = [x ÷ volumePerBar for x in tickData[!, :volumecumulated]]
    tickGrouped = copy(tickData)
    dates = combine(DataFrames.groupby(tickGrouped, :groupingID), :dates => first => :dates)
    tickDataGrouped = DataFrames.groupby(tickGrouped, :groupingID)
    ohlcvDataframe = ohlcv(tickDataGrouped)
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates)
    colDrop = [:groupingID]
    ohlcvDataframe = select!(ohlcvDataframe, Not(colDrop))
    return ohlcvDataframe
end

"""
Function to take a dataframe and generate a dollar bar dataframe.

Args:
    tickData (DataFrame): Dataframe of tick data.
    dollarPerBar (Int, optional): Dollars in each bar. Default is 100000.
    numberBars (Int, optional): Number of bars. Default is nothing.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: n/a
"""
function dollarBar(tickData,
                    dollarPerBar = 100000,
                    numberBars = nothing)
    tickData[!, :dollar] = tickData.price .* tickData.size
    tickData[!, :CumulativeDollars] = cumsum(tickData.dollar)
    if dollarPerBar == nothing
        dollarsTotal = tickData.CumulativeDollars[end]
        dollarPerBar = dollarsTotal / numberBars
        dollarPerBar = round(dollarPerBar; sigdigits = 2)
    end
    tickData[!, :groupingID] = [x ÷ dollarPerBar for x in tickData[!, :CumulativeDollars]]
    tickGrouped = copy(tickData)
    dates = combine(DataFrames.groupby(tickGrouped, :groupingID), :dates => first => :dates)
    tickDataGrouped = DataFrames.groupby(tickGrouped, :groupingID)
    ohlcvDataframe = ohlcv(tickDataGrouped)
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates)
    colDrop = [:groupingID]
    ohlcvDataframe = select!(ohlcvDataframe, Not(colDrop))
    return ohlcvDataframe
end

"""
Function to calculate hedging weights using covariance matrix, risk distribution (riskDist), and risk target (σ).

Args:
    cov (Matrix): Covariance matrix.
    riskDistribution (Vector, optional): Risk distribution. Default is nothing.
    σ (Float64, optional): Risk target. Default is 1.0.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Page 36
"""
function pcaWeights(cov,
                    riskDistribution = nothing,
                    σ = 1.0)
    Λ = eigvals(cov)
    V = eigvecs(cov)
    indices = reverse(sortperm(Λ))
    Λ = Λ[indices]
    V = V[:, indices]
    
    if riskDistribution == nothing
        riskDistribution = zeros(size(cov)[1])
        riskDistribution[end] = 1.0
    end
    
    loads = σ * (riskDistribution ./ Λ) .^ 0.5
    weights = V * loads
    return weights
end

"""
Function to implement the symmetric CUSUM filter.

Args:
    input (DataFrame): Dataframe of prices and dates.
    threshold (Float64): Threshold.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Page 39
"""
function symmetricCusumFilter(input,
                              threshold)
    timeEvents, shiftPositive, shiftNegative = [], 0, 0
    Δprice = DataFrame(hcat(input[2:end, 1], diff(input[:, 2])), :auto)
    
    for i ∈ Δprice[:, 1]
        shiftPositive = max(0, shiftPositive + Δprice[Δprice[:, 1] .== i, 2][1])
        shiftNegative = min(0, shiftNegative + Δprice[Δprice[:, 1] .== i, 2][1])
        
        if shiftNegative < -threshold
            shiftNegative = 0
            append!(timeEvents, [i])
        elseif shiftPositive > threshold
            shiftPositive = 0
            append!(timeEvents, [i])
        end
    end
    
    return timeEvents
end
