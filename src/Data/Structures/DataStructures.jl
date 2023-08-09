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
function show_progress_bar(value,
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
function compute_ewma(data,
                      windowLength = 100)
    N = size(data)[1]
    ewma_mean = []
    ewma_variance = []
    ewma_std = []
    α = 2 / Float64(windowLength + 1)

    for i ∈ 1:N
        window = Array(data[1:i, 1])
        m = length(window)
        ω = (1 - α).^range(m - 1, step = -1, stop = 0)
        ewma = sum(ω .* window) / sum(ω)
        bias = sum(ω)^2 / (sum(ω)^2 - sum(ω .^ 2))
        var = bias * sum(ω .* (window .- ewma).^2) / sum(ω)
        std = sqrt(var)
        append!(ewma_mean, ewma)
        append!(ewma_variance, var)
        append!(ewma_std, std)
    end
    return ewma_mean, ewma_variance, ewma_std
end

"""
Function to group a dataframe based on a feature and then calculate thresholds.

Args:
    target_col (Array): Target column of tick dataframe.
    tick_expected_init (Int): Initial expected ticks.
    bar_size (Float64): Initial expected size in each tick.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Page number: n/a

Methodology:
    Page number: n/a
"""
function group_and_calculate_thresholds(target_col,
                                        tick_expected_init,
                                        bar_size)
    Δtimes, times = [], []
    time_prev, tick_expected, bar_expected_value = 0, tick_expected_init, bar_size
    N = size(target_col)[1]
    θ_absolute, thresholds, θs, grouping_id = zeros(N), zeros(N), zeros(N), zeros(N)
    θ_absolute[1], θ_current = abs(target_col[1]), target_col[1]
    time = Dates.value((Dates.now() - Dates.DateTime(1970, 1, 1, 00, 00, 00))) / 1000
    grouping_id_current = 0

    for i ∈ 2:N
        θ_current += target_col[i]
        θs[i] = θ_current
        this_θ_absolute = abs(θ_current)
        θ_absolute[i] = this_θ_absolute
        threshold = tick_expected * bar_expected_value
        thresholds[i] = threshold
        grouping_id[i] = grouping_id_current

        if this_θ_absolute >= threshold
            grouping_id_current += 1
            θ_current = 0
            append!(Δtimes, Float64(i - time_prev))
            append!(times, i)
            time_prev = i
            tick_expected = ewma(Array(Δtimes), trunc(Int, length(Δtimes)))[1][end]
            bar_expected_value = abs(ewma(target_col[1:i], trunc(Int, tick_expected_init * 1))[1][end])
        end
    end
    return Δtimes, θ_absolute, thresholds, times, θs, grouping_id
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
function info_driven_bar(tickData,
                         type::String = "volume",
                         tickExpectedInit = 2000)
    if type == "volume"
        input_data = tickData.volumelabeled
    elseif type == "tick"
        input_data = tickData.label
    elseif type == "dollar"
        input_data = tickData.dollars
    else
        println("Error: unknown type")
    end
    bar_expected_value = abs(mean(input_data))
    Δtimes, θ_absolute, thresholds, times, θs, grouping_id = grouping(input_data, tickExpectedInit, bar_expected_value)
    tickData[!,:groupingID] = grouping_id
    dates = combine(DataFrames.groupby(tickData, :groupingID), :dates => first => :dates)
    tickDataGrouped = DataFrames.groupby(tickData, :groupingID)
    ohlcvDataframe = ohlcv(tickDataGrouped)
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates)
    col_drop = [:groupingID]
    ohlcvDataframe = select!(ohlcvDataframe, Not(col_drop))
    return ohlcvDataframe, θ_absolute, thresholds
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
function time_bar(tickData,
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
function tick_bar(tickData,
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
    col_drop = [:groupingID]
    ohlcvDataframe = select!(ohlcvDataframe, Not(col_drop))
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
function volume_bar(tickData,
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
    col_drop = [:groupingID]
    ohlcvDataframe = select!(ohlcvDataframe, Not(col_drop))
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
function dollar_bar(tickData,
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
    col_drop = [:groupingID]
    ohlcvDataframe = select!(ohlcvDataframe, Not(col_drop))
    return ohlcvDataframe
end
   

"""
Function to calculate hedging weights using covariance matrix, risk distribution (risk_dist), and risk target (σ).

Args:
    cov (Matrix): Covariance matrix.
    riskDisturbution (Vector, optional): Risk distribution. Default is nothing.
    σ (Float64, optional): Risk target. Default is 1.0.

Reference:
    De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: Page 36
"""
function pca_weights(cov,
                     riskDisturbution = nothing,
                     σ = 1.0)
    Λ = eigvals(cov)
    V = eigvecs(cov)
    indices = reverse(sortperm(Λ))
    Λ = Λ[indices]
    V = V[:, indices]
    
    if riskDisturbution == nothing
        riskDisturbution = zeros(size(cov)[1])
        riskDisturbution[end] = 1.0
    end
    
    loads = σ * (riskDisturbution ./ Λ) .^ 0.5
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
function symmetric_cusum_filter(input,
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
