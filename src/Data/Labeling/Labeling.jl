"""
    Function: Implementation of the symmetric CUSUM filter.
    Reference: De Prado, M (2018) Advances in financial machine learning.
    Methodology: Page 39.
"""
function events(input, # DataFrame of prices and dates
                threshold) # Threshold
    
    timeEvents, shiftPositive, shiftNegative = [], 0, 0
    # DataFrame with price differences
    Δprice = DataFrame(hcat(input[2:end, 1], diff(input[:, 2])), :auto) 
    for i ∈ Δprice[:, 1]
        # Compute shiftNegative/shiftPositive with min/max of 0 and ΔPRICE in each day
        shiftPositive = max(0, shiftPositive + Δprice[Δprice[:, 1] .== i, 2][1])
        shiftNegative = min(0, shiftNegative + Δprice[Δprice[:, 1] .== i, 2][1])
        if shiftNegative < -threshold
            shiftNegative = 0 # Reset shiftNegative to 0
            append!(timeEvents, [i]) # Append this time into timeEvents
        elseif shiftPositive > threshold
            shiftPositive = 0 # Reset shiftPositive to 0
            append!(timeEvents, [i]) # Append this time into timeEvents
        end
    end
    return timeEvents
end

"""
    Function: Computes the daily volatility at intraday estimation points.
    Reference: De Prado, M (2018) Advances in financial machine learning.
    Methodology: Page 44.
"""
function dailyVol(close, # DataFrame with columns Dates and close prices
                  span = 100) # Window for EWMA
    
    index = [] # Array contains index after searchsorting
    for i ∈ close.Dates .- Dates.Day(1)
        thisIndex = searchsortedfirst(close.Dates, i) # Searchsort a lag of one day in dates column
        append!(index, thisIndex) # Append this index into index
    end
    index = index[index .> 1] # Drop indexes when it's lower than 1
    dataframe = DataFrame(hcat(close.Dates[size(close)[1] - 
                size(index)[1] + 1:end], close.Dates[index .- 1]), :auto) 
                # DataFrame of dates and a lag of them
    returns = [] # Array of returns
    for i ∈ 1:size(dataframe)[1]
        day1 = dataframe[i, 1] # Each day
        day2 = dataframe[i, 2] # Previous day
        r = close[close[:, 1] .== day1, 2][1] / close[close[:, 1] .== day2, 2][1] - 1 # Calculate returns for each day
        append!(returns, r) # Append return into array
    end
    returnsDataframe = DataFrame(hcat(dataframe[:, 1], returns), :auto) # DataFrame of returns
    stdDataframe = DataFrame(hcat(dataframe[:, 1], ewma(returnsDataframe[:, 2], span)[3]), :auto) # DataFrame of EWMA stds
    return returnsDataframe, stdDataframe
end

"""
    Function: Computes the EWMA, EWMA variance, and EWMA stds.
    Reference: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
    Methodology: N/A.
"""
function ewma(data, # Array of data
              windowLength = 100) # Window for EWMA
    N = size(data)[1] # Length of array
    ewmaMean = [] # Array for output 
    ewmaVariance = [] # Array for output 
    ewmaStd = [] # Array for output 
    α = 2 / Float64(windowLength + 1) # Tune parameter for EWMA
    for i ∈ 1:N
        window = Array(data[1:i, 1]) # Get window
        # Get weights: ω
        m = length(window)
        ω = (1 - α) .^ range(m - 1, step = -1, stop = 0) # This is reverse order to match Series order
        ewma = sum(ω .* window) / sum(ω) # Calculate exponential moving average
        bias = sum(ω)^2 / (sum(ω)^2 - sum(ω .^ 2)) # Calculate bias
        var = bias * sum(ω .* (window .- ewma) .^ 2) / sum(ω) # Calculate exponential moving variance with bias
        std = sqrt(var) # Calculate standard deviation
        append!(ewmaMean, ewma) # Append calculated value into array 
        append!(ewmaVariance, var) # Append calculated value into array 
        append!(ewmaStd, std) # Append calculated value into array 
    end
    return ewmaMean, ewmaVariance, ewmaStd
end

"""
    Function: Implements the triple-barrier method.
    Reference: De Prado, M (2018) Advances in financial machine learning.
    Methodology: Page 45.
"""
function tripleBarrier(close, # DataFrame of prices and dates
                       events, # DataFrame, with columns: timestamp for vertical barrier and target for unit width of the horizontal barriers
                       profitTakingStopLoss, # List of two non-negative float values that multiply target
                       molecule) # A list with the subset of event indices that will be processed by a single thread
    eventsFiltered = filter(row -> row[:date] ∈ molecule, events) # Subset of events for a single thread
    # DataFrame of output with vertical barrier and two horizontal barriers
    output = eventsFiltered[:, [:date, :timestamp]]
    output.stoploss = repeat(Vector{Union{Float64, Date}}([NaN]), size(output)[1])
    output.profitTaking = repeat(Vector{Union{Float64, Date}}([NaN]), size(output)[1])
    if profitTakingStopLoss[1] > 0
        profitTaking = profitTakingStopLoss[1] .* eventsFiltered.target # Factors multiply target to set the width of the upper barrier.
        profitTaking = DataFrame(hcat(eventsFiltered.date, profitTaking), :auto) # DataFrame of profitTaking
    else
        # If 0, there will not be an upper barrier
        profitTaking = DataFrame(hcat(eventsFiltered.date, repeat([NaN],
                                            length(eventsFiltered.date))), :auto)
    end
    if profitTakingStopLoss[2] > 0
        stoploss = -profitTakingStopLoss[2] .* eventsFiltered.target # Factors multiply target to set the width of the lower barrier.
        stoploss = DataFrame(hcat(eventsFiltered.date, stoploss), :auto) # DataFrame of stoploss
    else
        # If 0, there will not be a lower barrier
        stoploss = DataFrame(hcat(eventsFiltered.date, repeat([NaN],
                                        length(eventsFiltered.date))), :auto)
    end
    timestampReplaced = replace(eventsFiltered.timestamp, NaN => close.Dates[end]) # Replace NaN timestamp by last date of close DataFrame
    dateDataframe = eventsFiltered[:, [:date, :timestamp]]
    dateDataframe.timestamp = timestampReplaced

    for i ∈ 1:size(dateDataframe)[1]
        location, timestamp = dateDataframe[i, 1], dateDataframe[i, 2] # Date and vertical barrier
        # Path prices
        dataframe = filter(row -> row[:Dates] ∈ collect(location:Dates.Day(1):timestamp), 
                           close) # DataFrame of path price 
        dataframe = DataFrame(hcat(dataframe.Dates, (dataframe[:, 2] ./
                                         close[close[:, :Dates] .== location, 2][1] .- 1) .* 
                                         eventsFiltered[eventsFiltered[:, :date] .== location, :side]), :auto)
                                         # DataFrame of path returns
        if length(dataframe[dataframe[:, 2] .< stoploss[stoploss[:, 1] .== location, 2][1], 2]) != 0
            output[output[:, 1] .== location, :stoploss] = 
            [first(dataframe[dataframe[:, 2] .< stoploss[stoploss[:, 1] .== location, 2][1], 1])]
            # Earliest stop loss
        else
            output[output[:, 1] .== location, :stoploss] = [NaN] # If sl is not touched
        end
        # Earliest profit taking.
        if length(dataframe[dataframe[:, 2] .> profitTaking[profitTaking[:, 1] .== location, 2][1], 2]) != 0
            output[output[:, 1] .== location, :profitTaking] = 
            [first(dataframe[dataframe[:, 2] .> profitTaking[profitTaking[:, 1] .== location, 2][1], 1])]
        else
            output[output[:, 1] .== location, :profitTaking] = [NaN] # If pt is not touched
        end
    end
    return output
end

"""
    Function: Finds the time of the first barrier touch.
    Reference: De Prado, M (2018) Advances in financial machine learning.
    Methodology: Page 48.
"""
function events(close, # DataFrame of prices and dates
                timeEvents, # Vector of timestamps
                ptsl, # A non-negative float that sets the width of the two barriers
                target,  # DataFrame of targets, expressed in terms of absolute returns
                returnMin, # The minimum target return required for running a triple barrier
                timestamp = false) # Vector contains the timestamps of the vertical barriers

    target = filter(row -> row[:x1] ∈ timeEvents, target) # Get target DataFrame
    if timestamp == false
        timestamp = repeat(Vector{Union{Float64, Date}}([NaN]), length(timeEvents)) #  Get timestamp (max holding period)
    else
        timestamp = vcat(timestamp, repeat(Vector{Union{Float64, Date}}([NaN]), length(timeEvents) - length(timestamp)))
        # Get timestamp (max holding period) based on vertical barrier
    end
    
    # Form events object, apply stop loss on timestamp
    sidePosition = repeat([1.], length(timeEvents)) # Create side array
    events = DataFrame(hcat(timeEvents, timestamp, target[:, :x2], sidePosition), :auto) # Create events DataFrame
    events = filter(row -> row[:x3] > returnMin, events) # ReturnMin
    rename!(events, names(events) .=> ["date", "timestamp", "target", "side"]) # Rename DataFrame

    dataframe = tripleBarrier(close, events, [ptsl, ptsl], events.date) # Apply tripleBarrier function
    columnPtSl = dataframe[:, 2:4] # Dates that vertical/pt/sl touched
    columnPtSl = replace!(Matrix(columnPtSl), NaN => Date(9999))  # Replace NaN with year 9999
    events.timestamp = minimum.(eachrow(columnPtSl)) # Find which barrier touched first
    events = select!(events, Not([:side])) # Select all the columns but side
    return events
end

"""
    Function: Defines a vertical barrier.
    Reference: De Prado, M (2018) Advances in financial machine learning.
    Methodology: Page 49.
"""
function verticalBarrier(close, # DataFrame of prices and dates
                         timeEvents, # Vector of timestamps
                         numberDays) # A number of days for vertical barrier
    timestampArray = [] # Array contains index after searchsorting
    for i ∈ timeEvents .+ Dates.Day(numberDays)
        index = searchsortedfirst(close.Dates, i) # Searchsort a lag of numberDays dates column
        append!(timestampArray, index) # Append that index into timestampArray
    end
    timestampArray = timestampArray[timestampArray .< size(close)[1]]
    timestampArray = DataFrame(hcat(timeEvents[1:size(timestampArray)[1]], 
                     close.Dates[timestampArray]), :auto) # DataFrame with start and end of an event
    return timestampArray
end

"""
    Function: Labels the observations.
    Reference: De Prado, M (2018) Advances in financial machine learning.
    Methodology: Page 49.
"""
function label(events, # DataFrame, with columns: timestamp for vertical barrier and target for unit width of the horizontal barriers
               close) # DataFrame of prices and dates
    eventsFiltered = filter(row -> row[:timestamp] != NaN, events) # Filter events without NaN
    unique!(eventsFiltered, :timestamp)
    allDates = union(eventsFiltered.date, eventsFiltered.timestamp) # Get all dates
    closeFiltered = filter(row -> row[:Dates] ∈ allDates, close) # Prices aligned with events
    out = DataFrame(Dates = eventsFiltered.date) # Create output object
    out.ret = filter(row -> row[:Dates] ∈ eventsFiltered.timestamp, closeFiltered)[:, 2] ./
              filter(row -> row[:Dates] ∈ eventsFiltered.date, closeFiltered)[:, 2] .- 1 # Calculate returns
    out.bin = sign.(out.ret) # Get sign of returns
    return out
end

"""
    Function: Expands events to incorporate meta-labeling.
    Reference: De Prado, M (2018) Advances in financial machine learning.
    Methodology: Page 50.
"""
function eventsMeta(close, # DataFrame of prices and dates
                    timeEvents, # Vector of timestamps
                    ptsl, # List of two non-negative float values that multiply target
                    target, # DataFrame of targets, expressed in terms of absolute returns
                    returnMin, # The minimum target return required for running a triple barrier
                    timestamp = false, # Vector contains the timestamps of the vertical barriers
                    side = nothing) # When side is not nothing, the function understands that meta-labeling is in play

    target = filter(row -> row[:x1] ∈ timeEvents, target) # Get target DataFrame
    if timestamp == false
        timestamp = repeat(Vector{Union{Float64, Date}}([NaN]), length(timeEvents)) #  Get timestamp (max holding period)
    else
        timestamp = vcat(timestamp, repeat(Vector{Union{Float64, Date}}([NaN]), length(timeEvents) - length(timestamp)))
        # Get timestamp (max holding period) based on vertical barrier
    end
    # Form events object, apply stop loss on timestamp
    if isnothing(side)
        sidePosition = repeat([1.], length(timeEvents)) # Create side array
        profitLoss = [ptsl[1], ptsl[1]]
    else
        sidePosition = filter(row -> row[:x1] ∈ target[:, 1], side)
        profitLoss = copy(ptsl)
    end
  
    events = DataFrame(hcat(timeEvents, timestamp, target[:, :x2], sidePosition), :auto) # Create events DataFrame
    events = filter(row -> row[:x3] > returnMin, events) # ReturnMin
    rename!(events, names(events) .=> ["date", "timestamp", "target", "side"]) # Rename DataFrame

    dataframe = tripleBarrier(close, events, profitLoss, events.date) # Apply tripleBarrier function

    columnPtSl = dataframe[:, 2:4] # Dates that vertical/pt/sl touched
    columnPtSl = replace!(Matrix(columnPtSl), NaN => Date(9999))  # Replace NaN with year 9999
    events.timestamp = minimum.(eachrow(columnPtSl)) # Find which barrier touched first
    events = select!(events, Not([:side])) # Select all the columns but side
    return events
end

"""
    Function: Expands label to incorporate meta-labeling.
    Reference: De Prado, M (2018) Advances in financial machine learning.
    Methodology: Page 51.
"""
function labelMeta(events, # DataFrame, with columns: timestamp for vertical barrier and target for unit width of the horizontal barriers
                   close) # DataFrame of prices and dates
    eventsFiltered = filter(row -> row[:timestamp] != NaN, events) # Filter events without NaN
    unique!(eventsFiltered, :timestamp)
    allDates = union(eventsFiltered.date, eventsFiltered.timestamp) # Get all dates
    closeFiltered = filter(row -> row[:Dates] ∈ allDates, close) # Prices aligned with events
    out = DataFrame(Dates = eventsFiltered.date) # Create output object
    out.ret = filter(row -> row[:Dates] ∈ eventsFiltered.timestamp, closeFiltered)[:, 2] ./
              filter(row -> row[:Dates] ∈ eventsFiltered.date, closeFiltered)[:, 2] .- 1 # Calculate returns
    if "side" ∈ names(eventsFiltered)
        out.ret .*= eventsFiltered.side # Meta-labeling
    end
    out.bin = sign.(out.ret) # Get sign of returns
    if "side" ∈ names(eventsFiltered)
        replace!(x -> x <= 0 ? 0 : x, out.bin) # Meta-labeling
    end
    return out
end

"""
    Function: Presents a procedure that recursively drops observations associated with extremely rare labels.
    Reference: De Prado, M (2018) Advances in financial machine learning.
    Methodology: Page 54.
"""
function dropLabel(events; # DataFrame, with columns: Dates, ret, and bin
                   percentMin = 0.05) # A fraction to eliminate observation
    # Apply weights, drop labels with insufficient examples
    while true
        dataframe = combine(groupby(out4, :bin), nrow) # Group and combine based on label
        dataframe.percent = dataframe.nrow ./ sum(dataframe.nrow) # Calculate percentage frequency
        if minimum(dataframe.percent) > percentMin || size(dataframe)[1] < 3 # Check for eliminating
            break
        end
        println("Dropped label", dataframe.bin[argmin(dataframe.percent)], minimum(dataframe.percent)) # Print results
        events = filter(row -> row[:bin] != dataframe.bin[argmin(dataframe.percent)], events) # Update events
    end
    return events
end
