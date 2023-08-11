using DataFrames

"""
Implementation of the symmetric CUSUM filter

Reference: De Prado, M (2018) Advances in financial machine learning
Methodology: Page 39
"""
function symmetricCusumFilter(input::DataFrame, threshold::Float64)::Vector{Date}
    timeEvents, shiftPositive, shiftNegative = Date[], 0.0, 0.0
    Δprice = DataFrame(Dates=input[2:end, 1], ΔPRICE=diff(input[:, 2]))
    
    for i ∈ Δprice[:, 1]
        shiftPositive = max(0.0, shiftPositive + Δprice[Δprice[:, 1] .== i, 2][1])
        shiftNegative = min(0.0, shiftNegative + Δprice[Δprice[:, 1] .== i, 2][1])
        
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

"""
Computes the daily volatility at intraday estimation points

Reference: De Prado, M (2018) Advances in financial machine learning
Methodology: Page 44
"""
function dailyVolatility(close::DataFrame, span::Int=100)::Tuple{DataFrame, DataFrame}
    index = []
    
    for i ∈ close.Dates .- Dates.Day(1)
        thisIndex = searchsortedfirst(close.Dates, i)
        push!(index, thisIndex)
    end
    
    index = index[index .> 1]
    dataframe = DataFrame(Dates=close.Dates[size(close)[1] - size(index)[1] + 1:end],
                          PreviousDates=close.Dates[index .- 1])
    
    returns = []
    
    for i ∈ 1:size(dataframe)[1]
        day1 = dataframe[i, 1]
        day2 = dataframe[i, 2]
        r = close[close.Dates .== day1, 2][1] / close[close.Dates .== day2, 2][1] - 1
        push!(returns, r)
    end
    
    returnsDataframe = DataFrame(Dates=dataframe[:, 1], Returns=returns)
    ewmaStdDataframe = DataFrame(Dates=dataframe[:, 1], EwmaStd=ewma(returnsDataframe.Returns, span)[3])
    
    return returnsDataframe, ewmaStdDataframe
end

"""
Computes the exponential weighted moving average (EWMA), EWMA variance, and EWMA standard deviations

Reference: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
Methodology: N/A
"""
function ewma(data::Vector{Float64}, windowLength::Int)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    N = length(data)
    ewmaMean = Float64[]
    ewmaVariance = Float64[]
    ewmaStd = Float64[]
    α = 2.0 / (Float64(windowLength) + 1.0)
    
    for i ∈ 1:N
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
Implements the triple-barrier method

Reference: De Prado, M (2018) Advances in financial machine learning
Methodology: Page 45
"""
function tripleBarrier(close::DataFrame, events::DataFrame, profitTakingStopLoss::Vector{Float64},
                        molecule::Vector{Date})::DataFrame
    eventsFiltered = filter(row -> row[:date] ∈ molecule, events)
    
    output = DataFrame(Dates=eventsFiltered.date, Timestamp=eventsFiltered.timestamp,
                       StopLoss=Vector{Union{Float64, Date}}(missing, size(eventsFiltered)[1]),
                       ProfitTaking=Vector{Union{Float64, Date}}(missing, size(eventsFiltered)[1]))
    
    if profitTakingStopLoss[1] > 0
        profitTaking = profitTakingStopLoss[1] .* eventsFiltered.target
        profitTaking = DataFrame(Dates=eventsFiltered.date, ProfitTaking=profitTaking)
    else
        profitTaking = DataFrame(Dates=eventsFiltered.date, ProfitTaking=repeat([missing], size(eventsFiltered)[1]))
    end
    
    if profitTakingStopLoss[2] > 0
        stopLoss = -profitTakingStopLoss[2] .* eventsFiltered.target
        stopLoss = DataFrame(Dates=eventsFiltered.date, StopLoss=stopLoss)
    else
        stopLoss = DataFrame(Dates=eventsFiltered.date, StopLoss=repeat([missing], size(eventsFiltered)[1]))
    end
    
    timestampReplaced = replace(eventsFiltered.timestamp, missing => close.Dates[end])
    dateDataframe = DataFrame(Dates=eventsFiltered.date, Timestamp=timestampReplaced)
    
    for i ∈ 1:size(dateDataframe)[1]
        location, timestamp = dateDataframe[i, 1], dateDataframe[i, 2]
        dataframe = filter(row -> row[:Dates] ∈ collect(location:Dates.Day(1):timestamp), close)
        dataframe = DataFrame(Dates=dataframe.Dates, PathReturns=((dataframe[:, 2] ./ close[close.Dates .== location, 2][1] .- 1) .* eventsFiltered[eventsFiltered.date .== location, :side]))
        
        if !isempty(dataframe[dataframe.PathReturns .< stopLoss[stopLoss.Dates .== location, :StopLoss][1], :PathReturns])
            output[output.Dates .== location, :StopLoss] =
                [first(dataframe[dataframe.PathReturns .< stopLoss[stopLoss.Dates .== location, :StopLoss][1], :Dates])]
        else
            output[output.Dates .== location, :StopLoss] = missing
        end
        
        if !isempty(dataframe[dataframe.PathReturns .> profitTaking[profitTaking.Dates .== location, :ProfitTaking][1], :PathReturns])
            output[output.Dates .== location, :ProfitTaking] =
                [first(dataframe[dataframe.PathReturns .> profitTaking[profitTaking.Dates .== location, :ProfitTaking][1], :Dates])]
        else
            output[output.Dates .== location, :ProfitTaking] = missing
        end
    end
    
    return output
end

"""
Finds the time of the first barrier touch

Reference: De Prado, M (2018) Advances in financial machine learning
Methodology: Page 48
"""
function firstBarrierTouchTime(close::DataFrame, timeEvents::Vector{Date}, ptsl::Float64,
                                target::DataFrame, returnMin::Float64)::DataFrame
    targetFiltered = filter(row -> row[:x1] ∈ timeEvents, target)
    
    if ismissing(targetFiltered[:x2][1])
        timestamp = repeat(Vector{Union{Float64, Date}}(missing), length(timeEvents))
    else
        timestamp = vcat(targetFiltered[:x2], repeat(Vector{Union{Float64, Date}}(missing), length(timeEvents) - length(targetFiltered[:x2])))
    end
    
    sidePosition = repeat([1.0], length(timeEvents))
    events = DataFrame(Dates=timeEvents, Timestamp=timestamp, Target=targetFiltered[:x2], Side=sidePosition)
    events = filter(row -> row[:Target] > returnMin, events)
    rename!(events, Symbol[:Dates, :Timestamp, :Target, :Side] => ["date", "timestamp", "target", "side"])
    
    dataframe = tripleBarrier(close, events, [ptsl, ptsl], events.date)
    columnPtSl = dataframe[:, 2:4]
    columnPtSl = replace(Matrix(columnPtSl), missing => Dates(9999))
    events.timestamp = minimum.(eachrow(columnPtSl))
    select!(events, Not([:side]))
    
    return events
end

"""
Defines a vertical barrier

Reference: De Prado, M (2018) Advances in financial machine learning
Methodology: Page 49
"""
function verticalBarrier(close::DataFrame, timeEvents::Vector{Date}, numberDays::Int)::DataFrame
    timestampArray = []
    
    for i ∈ timeEvents .+ Dates.Day(numberDays)
        index = searchsortedfirst(close.Dates, i)
        push!(timestampArray, index)
    end
    
    timestampArray = timestampArray[timestampArray .< size(close)[1]]
    timestampArray = DataFrame(Dates=timeEvents[1:size(timestampArray)[1]], VerticalBarrier=close.Dates[timestampArray])
    
    return timestampArray
end

"""
Labels the observations

Reference: De Prado, M (2018) Advances in financial machine learning
Methodology: Page 49
"""
function labelObservations(events::DataFrame, close::DataFrame)::DataFrame
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
Expands events to incorporate meta-labeling

Reference: De Prado, M (2018) Advances in financial machine learning
Methodology: Page 50
"""
function eventsMetaLabeling(close::DataFrame, timeEvents::Vector{Date}, ptsl::Vector{Float64},
                              target::DataFrame, returnMin::Float64, timestamp::Union{Bool, Vector{Union{Float64, Date}}}=false,
                              side=nothing)::DataFrame
    targetFiltered = filter(row -> row[:x1] ∈ timeEvents, target)
    
    if timestamp == false
        timestamp = repeat(Vector{Union{Float64, Date}}(missing), length(timeEvents))
    else
        timestamp = vcat(timestamp, repeat(Vector{Union{Float64, Date}}(missing), length(timeEvents) - length(timestamp)))
    end
    
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
    columnPtSl = replace(Matrix(columnPtSl), missing => Dates(9999))
    events.timestamp = minimum.(eachrow(columnPtSl))
    select!(events, Not([:side]))
    
    return events
end

"""
Expands labels to incorporate meta-labeling

Reference: De Prado, M (2018) Advances in financial machine learning
Methodology: Page 51
"""
function labelMetaLabeling(events::DataFrame, close::DataFrame)::DataFrame
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
Drops labels with insufficient examples

Reference: De Prado, M (2018) Advances in financial machine learning
Methodology: Page 54
"""
function dropLabels(events::DataFrame, percentMin::Float64)::DataFrame
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
