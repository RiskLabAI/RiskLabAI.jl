"""----------------------------------------------------------------------
function:  Implementation of the symmetric CUSUM filter
reference: De Prado, M (2018) Advances in financial machine learning
methodology: page 39
----------------------------------------------------------------------"""
function events(input, # dataframe of prices and dates
                threshold) # threshold
    
    timeEvents, shiftPositive, shiftNegative = [], 0, 0
    # dataframe with price differences
    Δprice = DataFrames.DataFrame(hcat(input[2:end, 1], diff(input[:, 2])), :auto) 
    for i ∈ Δprice[:, 1]
        # compute shiftNegative/shiftPositive with min/max of 0 and ΔPRICE in each day
        shiftPositive = max(0, shiftPositive + Δprice[Δprice[:, 1] .== i, 2][1]) # compare price diff with zero
        shiftNegative = min(0, shiftNegative + Δprice[Δprice[:, 1] .== i, 2][1]) # compare price diff with zero
        if shiftNegative < -threshold
            shiftNegative = 0 # reset shiftNegative to 0
            append!(timeEvents, [i]) # append this time into timeEvents
        elseif shiftPositive > threshold
            shiftPositive = 0 # reset shiftPositive to 0
            append!(timeEvents, [i])  # append this time into timeEvents
        end
    end
    return timeEvents
end

"""----------------------------------------------------------------------
    function: computes the daily volatility at intraday estimation points
    reference: De Prado, M (2018) Advances in financial machine learning
    methodology: Page 44
----------------------------------------------------------------------"""
function dailyVol(close, # dataframe with columns Dates and close prices
                  span = 100) # window for ewma
    
    index = [] # array contains index after searchsorting
    for i ∈ close.Dates .- Dates.Day(1)
        thisIndex = searchsortedfirst(close.Dates, i) # searchsort a lag of one day in dates column
        append!(index, thisIndex) # append this index into index
    end
    index = index[index .> 1] # drop indexes when it's lower than 1
    dataframe = DataFrames.DataFrame(hcat(close.Dates[size(close)[1] - 
                size(index)[1] + 1:end], close.Dates[index .- 1]), :auto) 
                # dataframe of dates and a lag of them
    returns=[] # array of returns
    for i ∈ 1:size(dataframe)[1]
        day1 = dataframe[i, 1] # each day
        day2 = dataframe[i, 2] # previous day
        r = close[close[:, 1] .== day1, 2][1]/close[close[:, 1] .== day2, 2][1] - 1 # calculate returns for each day
        append!(returns, r) # append return into array
    end
    returnsDataframe = DataFrames.DataFrame(hcat(dataframe[:, 1], returns), :auto) # dataframe of returns
    stdDataframe = DataFrames.DataFrame(hcat(dataframe[:, 1], ewma(returnsDataframe[:, 2], span)[3]), :auto) # dataframe of ewma stds
    return returnsDataframe, stdDataframe
end

"""----------------------------------------------------------------------
    function: computes the ewma, ewma var, and ewma stds
    reference: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
    methodology: n/a
----------------------------------------------------------------------"""
function ewma(data, # array of data
              windowLength = 100) # window for ewma
    N = size(data)[1] # length of array
    ewmaMean = [] # array for output 
    ewmaVariance = [] # array for output 
    ewmaStd = [] # array for output 
    α = 2/Float64(windowLength + 1) # tune parameter for ewma
    for i ∈ 1:N
        window = Array(data[1:i,1]) # Get window
        # Get weights: ω
        m = length(window)
        ω = (1 - α).^range(m - 1, step = -1, stop = 0) # This is reverse order to match Series order
        ewma = sum(ω.*window)/sum(ω) # Calculate exponential moving average
        bias = sum(ω)^2/(sum(ω)^2 - sum(ω.^2)) # Calculate bias
        var = bias*sum(ω.*(window .- ewma).^2)/sum(ω) # Calculate exponential moving variance with bias
        std = sqrt(var) # Calculate standard deviation
        append!(ewmaMean,ewma) # append calculated value into array 
        append!(ewmaVariance,var) # append calculated value into array 
        append!(ewmaStd,std) # append calculated value into array 
    end
    return ewmaMean, ewmaVariance, ewmaStd
end


"""----------------------------------------------------------------------
    function:  implements the triple-barrier method
    reference: De Prado, M (2018) Advances in financial machine learning
    methodology: Page 45
----------------------------------------------------------------------"""
function tripleBarrier(close, # dataframe of prices and dates
                       events, # dataframe, with columns,timestamp for vertical barrier and targt for unit width of the horizontal barriers
                       profitTakingStopLoss, # list of two non-negative float values that multiply targt
                       molecule) # a list with the subset of event indices that will be processed by a single thread
    eventsFiltered = filter(row -> row[:date] ∈ molecule, events) # subset of events for a single thread
    # dataframe of output with vertical barrier and two horizontal barriers
    output = eventsFiltered[:, [:date, :timestamp]]
    output.stoploss = repeat(Vector{Union{Float64, Date}}([NaN]), size(output)[1])
    output.profitTaking = repeat(Vector{Union{Float64, Date}}([NaN]), size(output)[1])
    if profitTakingStopLoss[1] > 0
        profitTaking = profitTakingStopLoss[1].*eventsFiltered.target # factors multiply targt to set the width of the upper barrier.
        profitTaking = DataFrames.DataFrame(hcat(eventsFiltered.date, profitTaking), :auto) # dataframe of profitTaking
    else
        # If 0, there will not be an upper barrier
        profitTaking = DataFrames.DataFrame(hcat(eventsFiltered.date, repeat([NaN],
                                            length(eventsFiltered.date))), :auto)
    end
    if profitTakingStopLoss[2] > 0
        stoploss = -profitTakingStopLoss[2].*eventsFiltered.target # factors multiply targt to set the width of the lower barrier.
        stoploss = DataFrames.DataFrame(hcat(eventsFiltered.date,stoploss), :auto) # dataframe of stoploss
    else
        # If 0, there will not be an lower barrier
        stoploss = DataFrames.DataFrame(hcat(eventsFiltered.date, repeat([NaN],
                                        length(eventsFiltered.date))), :auto)
    end
    timestampReplaced = replace(eventsFiltered.timestamp, NaN=>close.Dates[end]) # replace NaN timestamp by last date of close dataframe
    dateDataframe = eventsFiltered[:,[:date, :timestamp]]
    dateDataframe.timestamp = timestampReplaced

    for i ∈ 1:size(dateDataframe)[1]
        location, timestamp = dateDataframe[i, 1], dateDataframe[i, 2] # date and vartical barrier
        # path prices
        dataframe = filter(row -> row[:Dates] ∈ collect(location:Dates.Day(1):timestamp), 
                           close) # dataframe of path price 
        dataframe = DataFrames.DataFrame(hcat(dataframe.Dates, (dataframe[:, 2]./
                                         close[close[:, :Dates].==location, 2][1] .- 1).* 
                                         eventsFiltered[eventsFiltered[:, :date] .== location, :side]), :auto)
                                         # dataframe of path returns
        if length(dataframe[dataframe[:, 2] .< stoploss[stoploss[:, 1].==location,2][1],2]) != 0
            output[output[:, 1] .== location, :stoploss] = 
            [first(dataframe[dataframe[:, 2] .< stoploss[stoploss[:, 1] .== location,2][1], 1])]
            # earliest stop loss
        else
            output[output[:, 1] .== location, :stoploss] = [NaN] # if sl is not touched
        end
         # earliest profit taking.
        if length(dataframe[dataframe[:, 2] .> profitTaking[profitTaking[:,1] .== location,2][1],2]) != 0
            output[output[:, 1] .== location, :profitTaking] = 
            [first(dataframe[dataframe[:, 2] .> profitTaking[profitTaking[:, 1] .== location, 2][1], 1])]
        else
            output[output[:, 1] .== location, :profitTaking] = [NaN] # if pt is not touched
        end
    end
    return output
end

"""----------------------------------------------------------------------
function: finds the time of the first barrier touch
reference: De Prado, M (2018) Advances in financial machine learning
methodology: 48
----------------------------------------------------------------------"""
function events(close, # dataframe of prices and dates
                timeEvents, # vecotr of timestamps
                ptsl, # a non-negative float that sets the width of the two barriers
                target,  # dataframe of targets, expressed ∈ terms of absolute returns
                returnMin; # The minimum target return required for running a triple barrier
                timestamp = false) # vector contains the timestamps of the vertical barriers

    target = filter(row -> row[:x1] ∈ timeEvents, target) #get target dataframe
    if timestamp == false
        timestamp = repeat(Vector{Union{Float64, Date}}([NaN]), length(timeEvents)) #  get timestamp (max holding period)
    else
        timestamp = vcat(timestamp, repeat(Vector{Union{Float64, Date}}([NaN]), length(timeEvents) - length(timestamp)))
        # get timestamp (max holding period) based on vertical barrier
    end
    
    # form events object, apply stop loss on timestamp
    sidePosition = repeat([1.], length(timeEvents)) # create side array
    events = DataFrames.DataFrame(hcat(timeEvents, timestamp, target[:, :x2], sidePosition), :auto) #create events dataframe
    events = filter(row -> row[:x3] > returnMin, events) # returnMin
    DataFrames.rename!(events, names(events) .=> ["date", "timestamp", "target", "side"]) # rename dataframe

    dataframe = tripleBarrier(close, events, [ptsl,ptsl], events.date) #apply tripleBarrier function
    columnPtSl = dataframe[:, 2:4] # dates that vertical/pt/sl touched
    columnPtSl = replace!(Matrix(columnPtSl), NaN => Date(9999))  # replace NaN with year 9999
    events.timestamp = minimum.(eachrow(columnPtSl)) # find which barrier touched first
    events = select!(events, Not([:side])) # select all the columns but side
    return events
end

"""----------------------------------------------------------------------
    function: shows one way to define a vertical barrier
    reference: De Prado, M (2018) Advances in financial machine learning
    methodology: 49
----------------------------------------------------------------------"""
function verticalBarrier(close, # dataframe of prices and dates
                         timeEvents, # vecotr of timestamps
                         numberDays) # a number of days for vertical barrier
    timestampArray = [] # array contains index after searchsorting
    for i ∈ timeEvents .+ Dates.Day(numberDays)
        index = searchsortedfirst(close.Dates, i) # searchsort a lag of numberDays dates column
        append!(timestampArray, index) # append that index into 
    end
    timestampArray = timestampArray[timestampArray .< size(close)[1]]
    timestampArray = DataFrames.DataFrame(hcat(timeEvents[1:size(timestampArray)[1]], 
                     close.Dates[timestampArray]), :auto) # dataframe with start and end of an event
    return timestampArray
end

"""----------------------------------------------------------------------
    function: label the observations
    reference: De Prado, M (2018) Advances in financial machine learning
    methodology: 49
----------------------------------------------------------------------"""
function label(events, # dataframe, with columns,timestamp for vertical barrier and target for unit width of the horizontal barriers
               close) # dataframe of prices and dates
    eventsFiltered = filter(row -> row[:timestamp] != NaN, events) # filter events without NaN
    unique!(eventsFiltered, :timestamp)
    allDates = union(eventsFiltered.date, eventsFiltered.timestamp) # get all dates
    closeFiltered = filter(row->row[:Dates] ∈ allDates, close) # prices aligned with events
    out = DataFrames.DataFrame(Dates = eventsFiltered.date) # create output object
    out.ret = filter(row -> row[:Dates] ∈ eventsFiltered.timestamp, closeFiltered)[:, 2]./ 
              filter(row -> row[:Dates] ∈ eventsFiltered.date, closeFiltered)[:, 2] .- 1 # calculate returns
    out.bin = sign.(out.ret) # get sign of returns
    return out
end


"""----------------------------------------------------------------------
    function: expand events tO incorporate meta-labeling
    reference: De Prado, M (2018) Advances in financial machine learning
    methodology: 50
----------------------------------------------------------------------"""
function eventsMeta(close, # dataframe of prices and dates
                    timeEvents, # vecotr of timestamps
                    ptsl, # list of two non-negative float values that multiply targt
                    target, # dataframe of targets, expressed ∈ terms of absolute returns
                    returnMin; # The minimum target return required for running a triple barrier
                    timestamp = false, # vector contains the timestamps of the vertical barriers
                    side = nothing) # when side is not nothing, the function understands that meta-labeling is in play

    target = filter(row -> row[:x1] ∈ timeEvents, target) #get target dataframe
    if timestamp == false
        timestamp = repeat(Vector{Union{Float64, Date}}([NaN]), length(timeEvents)) #  get timestamp (max holding period)
    else
        timestamp = vcat(timestamp, repeat(Vector{Union{Float64, Date}}([NaN]), length(timeEvents) - length(timestamp)))
        # get timestamp (max holding period) based on vertical barrier
    end
    # form events object, apply stop loss on timestamp
    if isnothing(side)
        sidePosition = repeat([1.], length(timeEvents)) # create side array
        profitLoss = [ptsl[1], ptsl[1]]
    else
        sidePosition = filter(row -> row[:x1] ∈ target[:, 1], side)
        profitLoss = copy(ptsl)
    end
  
    events = DataFrames.DataFrame(hcat(timeEvents, timestamp, target[:, :x2], sidePosition), :auto) #create events dataframe
    events = filter(row -> row[:x3] > returnMin, events) # returnMin
    DataFrames.rename!(events, names(events) .=> ["date", "timestamp", "target", "side"]) # rename dataframe

    dataframe = tripleBarrier(close, events, profitLoss, events.date) #apply tripleBarrier function

    columnPtSl = dataframe[:, 2:4] # dates that vertical/pt/sl touched
    columnPtSl = replace!(Matrix(columnPtSl), NaN => Date(9999))  # replace NaN with year 9999
    events.timestamp = minimum.(eachrow(columnPtSl)) # find which barrier touched first
    events = select!(events, Not([:side])) # select all the columns but side
    return events
end

"""----------------------------------------------------------------------
    function: expand label tO incorporate meta-labeling
    reference: De Prado, M (2018) Advances in financial machine learning
    methodology: 51
----------------------------------------------------------------------"""
function labelMeta(events, # dataframe, with columns,timestamp for vertical barrier and target for unit width of the horizontal barriers
                   close) # dataframe of prices and dates
    eventsFiltered = filter(row -> row[:timestamp] != NaN, events) # filter events without NaN
    unique!(eventsFiltered, :timestamp)
    allDates = union(eventsFiltered.date, eventsFiltered.timestamp) # get all dates
    closeFiltered = filter(row->row[:Dates] ∈ allDates, close) # prices aligned with events
    out = DataFrames.DataFrame(Dates = eventsFiltered.date) # create output object
    out.ret = filter(row -> row[:Dates] ∈ eventsFiltered.timestamp, closeFiltered)[:, 2]./ 
              filter(row -> row[:Dates] ∈ eventsFiltered.date, closeFiltered)[:, 2] .- 1 # calculate returns
    if "side" ∈ names(eventsFiltered)
        out.ret .*= eventsFiltered.side # meta-labeling
    end
    out.bin = sign.(out.ret) # get sign of returns
    if "side" ∈ names(eventsFiltered)
        replace!(x -> x <= 0 ? 0 : x, out.bin) # meta-labeling
    end
    return out
end

"""----------------------------------------------------------------------
    function: presents a procedure that recursively drops observations associated with extremely rare labels
    reference: De Prado, M (2018) Advances in financial machine learning
    methodology: 54
----------------------------------------------------------------------"""
function dropLabel(events; # dataframe, with columns: Dates, ret, and bin
                   percentMin = 0.05) # a fraction to eliminate observation
    # apply weights, drop labels with insufficient examples
    while true
        dataframe = combine(DataFrames.groupby(out4, :bin), nrow) # group and combine based on label
        dataframe.percent = dataframe.nrow./sum(dataframe.nrow) # calculate percentage frequency
        if minimum(dataframe.percent) > percentMin || size(dataframe)[1] < 3 # check for eliminating
            break
        end
        println("dropped label", dataframe.bin[argmin(dataframe.percent)], minimum(dataframe.percent)) # print results
        events = filter(row -> rows[:bin] != dataframe.bin[argmin(dataframe.percent)], events) # update events
    end
    return events
end