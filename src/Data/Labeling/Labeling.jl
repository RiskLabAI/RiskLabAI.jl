using DataFrames, Dates

"""
    symmetricCusumFilter(input::DataFrame, threshold::Float64) :: Vector{Date}

Implementation of the symmetric CUSUM filter.

# Arguments
- `input::DataFrame`: Input data frame with two columns: Dates and Prices.
- `threshold::Float64`: Threshold for the filter.

# Returns
- `Vector{Date}`: Vector of dates when events occur.

# Reference
De Prado, M (2018) Advances in financial machine learning, Page 39.
"""
function symmetricCusumFilter(
    input::DataFrame,
    threshold::Float64
) :: Vector{Date}
    timeEvents = Date[]
    shiftPositive = 0.0
    shiftNegative = 0.0
    priceDiff = diff(input[:, 2])
    Δprice = DataFrame(Dates=input[2:end, 1], ΔPrice=priceDiff)
    
    for i ∈ Δprice[:, 1]
        shiftPositive = max(0.0, shiftPositive + Δprice[Δprice[:, 1] .== i, :ΔPrice][1])
        shiftNegative = min(0.0, shiftNegative + Δprice[Δprice[:, 1] .== i, :ΔPrice][1])
        
        if shiftNegative < -threshold
            shiftNegative = 0.0
            push!(timeEvents, i)
        elseif shiftPositive > threshold
            shiftPositive = 0.0
            push!(timeEvents, i)
        end
    end
    
    return timeEvents
end

using DataFrames

"""
    dailyVolatility(close::DataFrame, span::Int=100)::Tuple{DataFrame, DataFrame}

Computes the daily volatility at intraday estimation points.

# Arguments
- `close::DataFrame`: A DataFrame containing the daily closing prices with columns "Dates" and the price in the second column.
- `span::Int=100`: The window length for the exponential weighted moving average (EWMA) calculation.

# Returns
- A tuple of DataFrames, the first containing the daily returns and the second containing the daily EWMA standard deviations.

# References
De Prado, M (2018) Advances in financial machine learning. Methodology: Page 44
"""
function dailyVolatility(
    close::DataFrame,
    span::Int=100
)::Tuple{DataFrame, DataFrame}
    index = []

    for i in close.Dates .- Dates.Day(1)
        thisIndex = searchsortedfirst(close.Dates, i)
        push!(index, thisIndex)
    end

    index = index[index .> 1]
    dataframe = DataFrame(
        Dates=close.Dates[lastindex(close.Dates) - length(index) + 1:end],
        PreviousDates=close.Dates[index .- 1]
    )

    returns = []

    for i in 1:nrow(dataframe)
        day1 = dataframe[i, :Dates]
        day2 = dataframe[i, :PreviousDates]
        r = close[close.Dates .== day1, 2][1] / close[close.Dates .== day2, 2][1] - 1
        push!(returns, r)
    end

    returnsDataframe = DataFrame(Dates=dataframe[:, :Dates], Returns=returns)
    ewmaStdDataframe = DataFrame(Dates=dataframe[:, :Dates], EwmaStd=ewma(returnsDataframe.Returns, span)[3])

    return returnsDataframe, ewmaStdDataframe
end

using DataFrames
using Dates

"""
    ewma(data::Vector{Float64}, windowLength::Int)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}

Computes the exponential weighted moving average (EWMA), EWMA variance, and EWMA standard deviations.

# Arguments
- `data::Vector{Float64}`: A vector of floats, representing the data series.
- `windowLength::Int`: The length of the window for the EWMA calculation.

# Returns
- A tuple containing three vectors: the first is the EWMA mean, the second is the EWMA variance, and the third is the EWMA standard deviation.

# References
https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
"""
function ewma(
    data::Vector{Float64}, 
    windowLength::Int
)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    N = length(data)
    ewmaMean = Float64[]
    ewmaVariance = Float64[]
    ewmaStd = Float64[]
    α = 2.0 / (Float64(windowLength) + 1.0)
    
    for i in 1:N
        window = data[1:i]
        m = length(window)
        ω = (1.0 - α) .^ (m - 1:-1:0)
        ewma = sum(ω .* window) / sum(ω)
        bias = sum(ω)^2 / (sum(ω)^2 - sum(ω.^2))
        var = bias * sum(ω .* (window .- ewma).^2) / sum(ω)
        std = sqrt(var)
        push!(ewmaMean, ewma)
        push!(ewmaVariance, var)
        push!(ewmaStd, std)
    end
    
    return ewmaMean, ewmaVariance, ewmaStd
end

"""
    tripleBarrier(
        close::DataFrame, 
        events::DataFrame, 
        profitTakingStopLoss::Vector{Float64},
        molecule::Vector{Date}
    )::DataFrame

Implements the triple-barrier method.

# Arguments
- `close::DataFrame`: A DataFrame containing the daily closing prices with a column "Dates" and the price in the second column.
- `events::DataFrame`: A DataFrame containing event data.
- `profitTakingStopLoss::Vector{Float64}`: A vector of two floats, the first is the profit taking threshold and the second is the stop loss threshold.
- `molecule::Vector{Date}`: A vector of dates representing the time period for the analysis.

# Returns
- A DataFrame containing the dates, timestamps, stop loss dates, and profit taking dates for each event.

# References
De Prado, M (2018) Advances in financial machine learning. Methodology: Page 45
"""
function tripleBarrier(
    close::DataFrame, 
    events::DataFrame, 
    profitTakingStopLoss::Vector{Float64},
    molecule::Vector{Date}
)::DataFrame
    eventsFiltered = filter(row -> row[:date] ∈ molecule, events)
    
    output = DataFrame(
        Dates=eventsFiltered.date, 
        Timestamp=eventsFiltered.timestamp,
        StopLoss=Vector{Union{Float64, Date}}(missing, nrow(eventsFiltered)),
        ProfitTaking=Vector{Union{Float64, Date}}(missing, nrow(eventsFiltered))
    )
    
    profitTaking = profitTakingStopLoss[1] > 0 ? profitTakingStopLoss[1] .* eventsFiltered.target : repeat([missing], nrow(eventsFiltered))
    stopLoss = profitTakingStopLoss[2] > 0 ? -profitTakingStopLoss[2] .* eventsFiltered.target : repeat([missing], nrow(eventsFiltered))

    timestampReplaced = replace(eventsFiltered.timestamp, missing => last(close.Dates))
    dateDataframe = DataFrame(Dates=eventsFiltered.date, Timestamp=timestampReplaced)
    
    for i in 1:nrow(dateDataframe)
        location, timestamp = dateDataframe[i, :Dates], dateDataframe[i, :Timestamp]
        dataframe = filter(row -> row[:Dates] ∈ collect(location:Dates.Day(1):timestamp), close)
        pathReturns = (dataframe[:, 2] ./ close[close.Dates .== location, 2][1] .- 1) .* eventsFiltered[eventsFiltered.date .== location, :side]
        dataframe = DataFrame(Dates=dataframe.Dates, PathReturns=pathReturns)
        
        stopLossValue = stopLoss[stopLoss.Dates .== location, :StopLoss][1]
        profitTakingValue = profitTaking[profitTaking.Dates .== location, :ProfitTaking][1]
        
        output[output.Dates .== location, :StopLoss] = !isempty(dataframe[dataframe.PathReturns .< stopLossValue, :PathReturns]) ? first(dataframe[dataframe.PathReturns .< stopLossValue, :Dates]) : missing
        output[output.Dates .== location, :ProfitTaking] = !isempty(dataframe[dataframe.PathReturns .> profitTakingValue, :PathReturns]) ? first(dataframe[dataframe.PathReturns .> profitTakingValue, :Dates]) : missing
    end
    
    return output
end
using DataFrames, Dates

"""
    firstBarrierTouchTime(close, timeEvents, ptsl, target, returnMin)

Finds the time of the first barrier touch.

Reference: De Prado, M (2018) Advances in Financial Machine Learning, Page 48

Parameters
----------
- close : DataFrame
    DataFrame with closing prices.
- timeEvents : Vector{Date}
    Vector of time events.
- ptsl : Float64
    Profit-taking and stop-loss limits.
- target : DataFrame
    Target data.
- returnMin : Float64
    Minimum return value.

Returns
-------
- DataFrame
    DataFrame with the time of the first barrier touch.

"""
function firstBarrierTouchTime(
    close::DataFrame,
    timeEvents::Vector{Date},
    ptsl::Float64,
    target::DataFrame,
    returnMin::Float64
)::DataFrame
    targetFiltered = filter(row -> row[:x1] ∈ timeEvents, target)
    timestamp = ifelse(ismissing(targetFiltered[:x2][1]),
        repeat(Vector{Union{Float64, Date}}(missing), length(timeEvents)),
        vcat(targetFiltered[:x2], repeat(Vector{Union{Float64, Date}}(missing), length(timeEvents) - length(targetFiltered[:x2])))
    )
    sidePosition = repeat([1.0], length(timeEvents))
    events = DataFrame(Dates=timeEvents, Timestamp=timestamp, Target=targetFiltered[:x2], Side=sidePosition)
    events = filter(row -> row[:Target] > returnMin, events)
    rename!(events, Symbol[:Dates, :Timestamp, :Target, :Side] => ["date", "timestamp", "target", "side"])
    dataframe = tripleBarrier(close, events, [ptsl, ptsl], events.date)
    columnPtSl = dataframe[:, 2:4]
    columnPtSl = replace(Matrix(columnPtSl), missing => Date(9999))
    events.timestamp = minimum.(eachrow(columnPtSl))
    select!(events, Not([:side]))

    return events
end

"""
    verticalBarrier(close, timeEvents, numberDays)

Defines a vertical barrier.

Reference: De Prado, M (2018) Advances in Financial Machine Learning, Page 49

Parameters
----------
- close : DataFrame
    DataFrame with closing prices.
- timeEvents : Vector{Date}
    Vector of time events.
- numberDays : Int
    Number of days for the vertical barrier.

Returns
-------
- DataFrame
    DataFrame with the vertical barrier information.

"""
function verticalBarrier(
    close::DataFrame,
    timeEvents::Vector{Date},
    numberDays::Int
)::DataFrame
    timestampArray = map(i -> searchsortedfirst(close.Dates, i), timeEvents .+ Dates.Day(numberDays))
    timestampArray = timestampArray[timestampArray .< size(close)[1]]
    timestampArray = DataFrame(Dates=timeEvents[1:size(timestampArray)[1]], VerticalBarrier=close.Dates[timestampArray])

    return timestampArray
end

using DataFrames, Dates

"""
    labelObservations(events, close)

Labels the observations.

Reference: De Prado, M (2018) Advances in Financial Machine Learning, Page 49

Parameters
----------
- events : DataFrame
    DataFrame with event information.
- close : DataFrame
    DataFrame with closing prices.

Returns
-------
- DataFrame
    DataFrame with labeled observations.

"""
function labelObservations(
    events::DataFrame,
    close::DataFrame
)::DataFrame
    eventsFiltered = filter(row -> !ismissing(row[:timestamp]), events)
    unique!(eventsFiltered, :timestamp)
    allDates = union(eventsFiltered.date, eventsFiltered.timestamp)
    closeFiltered = filter(row -> row[:Dates] ∈ allDates, close)
    out = DataFrame(Dates=eventsFiltered.date)
    out.ret = filter(row -> row[:Dates] ∈ eventsFiltered.timestamp, closeFiltered)[:, 2] ./ filter(row -> row[:Dates] ∈ eventsFiltered.date, closeFiltered)[:, 2] .- 1
    out.bin = sign.(out.ret)
    return out
end

"""
    eventsMetaLabeling(close, timeEvents, ptsl, target, returnMin, timestamp, side)

Expands events to incorporate meta-labeling.

Reference: De Prado, M (2018) Advances in Financial Machine Learning, Page 50

Parameters
----------
- close : DataFrame
    DataFrame with closing prices.
- timeEvents : Vector{Date}
    Vector of time events.
- ptsl : Vector{Float64}
    Profit-taking and stop-loss limits.
- target : DataFrame
    Target data.
- returnMin : Float64
    Minimum return value.
- timestamp : Union{Bool, Vector{Union{Float64, Date}}}
    Timestamp information.
- side : Any
    Side information.

Returns
-------
- DataFrame
    DataFrame with meta-labeled events.

"""
function eventsMetaLabeling(
    close::DataFrame,
    timeEvents::Vector{Date},
    ptsl::Vector{Float64},
    target::DataFrame,
    returnMin::Float64,
    timestamp::Union{Bool, Vector{Union{Float64, Date}}} = false,
    side = nothing
)::DataFrame
    targetFiltered = filter(row -> row[:x1] ∈ timeEvents, target)
    timestamp = ifelse(timestamp == false,
        repeat(Vector{Union{Float64, Date}}(missing), length(timeEvents)),
        vcat(timestamp, repeat(Vector{Union{Float64, Date}}(missing), length(timeEvents) - length(timestamp)))
    )
    if isnothing(side)
        sidePosition = repeat([1.0], length(timeEvents))
        profitLoss = [ptsl[1], ptsl[1]]
    else
        sidePosition = filter(row -> row[:x1] ∈ target[:, 1], side)
        profitLoss = copy(ptsl)
    end
    events = DataFrame(Dates=timeEvents, Timestamp=timestamp, Target=targetFiltered[:x2], Side=sidePosition)
    events = filter(row -> row[:Target] > returnMin, events)
    rename!(events, Symbol[:Dates, :Timestamp, :Target, :Side] => ["date", "timestamp", "target", "side"])
    dataframe = tripleBarrier(close, events, profitLoss, events.date)
    columnPtSl = dataframe[:, 2:4]
    columnPtSl = replace(Matrix(columnPtSl), missing => Date(9999))
    events.timestamp = minimum.(eachrow(columnPtSl))
    select!(events, Not([:side]))
    return events
end

"""
    labelMetaLabeling(events, close)

Expands labels to incorporate meta-labeling.

Reference: De Prado, M (2018) Advances in Financial Machine Learning, Page 51

Parameters
----------
- events : DataFrame
    DataFrame with event information.
- close : DataFrame
    DataFrame with closing prices.

Returns
-------
- DataFrame
    DataFrame with meta-labeled labels.

"""
function labelMetaLabeling(
    events::DataFrame,
    close::DataFrame
)::DataFrame
    eventsFiltered = filter(row -> !ismissing(row[:timestamp]), events)
    unique!(eventsFiltered, :timestamp)
    allDates = union(eventsFiltered.date, eventsFiltered.timestamp)
    closeFiltered = filter(row -> row[:Dates] ∈ allDates, close)
    out = DataFrame(Dates=eventsFiltered.date)
    out.ret = filter(row -> row[:Dates] ∈ eventsFiltered.timestamp, closeFiltered)[:, 2] ./ filter(row -> row[:Dates] ∈ eventsFiltered.date, closeFiltered)[:, 2] .- 1
    if :side in names(eventsFiltered)
        out.ret .*= eventsFiltered.side
    end
    out.bin = sign.(out.ret)
    if :side in names(eventsFiltered)
        replace!(x -> x <= 0 ? 0 : x, out.bin)
    end
    return out
end

"""
    dropLabels(events, percentMin)

Drops labels with insufficient examples.

Reference: De Prado, M (2018) Advances in Financial Machine Learning, Page 54

Parameters
----------
- events : DataFrame
    DataFrame with event information.
- percentMin : Float64
    Minimum percentage of labels.

Returns
-------
- DataFrame
    DataFrame with filtered labels.

"""
function dropLabels(
    events::DataFrame,
    percentMin::Float64
)::DataFrame
    while true
        dataframe = combine(groupby(events, :bin), nrow)
        dataframe.percent = dataframe.nrow ./ sum(dataframe.nrow)
        if minimum(dataframe.percent) > percentMin || size(dataframe)[1] < 3
            break
        end
        println("Dropped label", dataframe.bin[argmin(dataframe.percent)], minimum(dataframe.percent))
        events = filter(row -> row[:bin] != dataframe.bin[argmin(dataframe.percent)], events)
    end
    return events
end
