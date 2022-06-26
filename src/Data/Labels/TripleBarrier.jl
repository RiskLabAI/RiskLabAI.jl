"""----------------------------------------------------------------------
    function:  implements the triple-barrier method
    reference: De Prado, M. (2018) Advances in financial machine learning.
    methodology: Page 45
----------------------------------------------------------------------"""
function tripleBarrier(close, # dataframe of prices and dates
                       events, # dataframe, with columns,timestamp for vertical barrier and targt for unit width of the horizontal barriers
                       profitTakingStopLoss, # list of two non-negative float values that multiply targt
                       molecule) # a list with the subset of event indices that will be processed by a single thread
    eventsfiltered = filter(row -> row[:date] ∈ molecule, events) # subset of events for a single thread
    # dataframe of output with vertical barrier and two horizontal barriers
    output = eventsfiltered[:, [:date, :timestamp]]
    output.stoploss = repeat(Vector{Union{Float64, Date}}([NaN]), size(output)[1])
    output.profitTaking = repeat(Vector{Union{Float64, Date}}([NaN]), size(output)[1])
    if profitTakingStopLoss[1] > 0
        profitTaking = profitTakingStopLoss[1].*eventsfiltered.target # factors multiply targt to set the width of the upper barrier.
        profitTaking = DataFrames.DataFrame(hcat(eventsfiltered.date, profitTaking), :auto) # dataframe of profitTaking
    else
        # If 0, there will not be an upper barrier
        profitTaking = DataFrames.DataFrame(hcat(eventsfiltered.date, repeat([NaN],
                                            length(eventsfiltered.date))), :auto)
    end
    if profitTakingStopLoss[2] > 0
        stoploss = -profitTakingStopLoss[2].*eventsfiltered.target # factors multiply targt to set the width of the lower barrier.
        stoploss = DataFrames.DataFrame(hcat(eventsfiltered.date,stoploss), :auto) # dataframe of stoploss
    else
        # If 0, there will not be an lower barrier
        stoploss = DataFrames.DataFrame(hcat(eventsfiltered.date, repeat([NaN],
                                        length(eventsfiltered.date))), :auto)
    end
    timestampreplaced = replace(eventsfiltered.timestamp, NaN=>close.Dates[end]) # replace NaN timestamp by last date of close dataframe
    datedataframe = eventsfiltered[:,[:date, :timestamp]]
    datedataframe.timestamp = timestampreplaced

    for i ∈ 1:size(datedataframe)[1]
        location, timestamp = datedataframe[i, 1], datedataframe[i, 2] # date and vartical barrier
        # path prices
        dataframe = filter(row -> row[:Dates] ∈ collect(location:Dates.Day(1):timestamp), 
                           close) # dataframe of path price 
        dataframe = DataFrames.DataFrame(hcat(dataframe.Dates, (dataframe[:, 2]./
                                         close[close[:, :Dates].==location, 2][1] .- 1).* 
                                         eventsfiltered[eventsfiltered[:, :date] .== location, :side]), :auto)
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
reference: De Prado, M. (2018) Advances in financial machine learning.
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
    colptsl = dataframe[:, 2:4] # dates that vertical/pt/sl touched
    colptsl = replace!(Matrix(colptsl), NaN => Date(9999))  # replace NaN with year 9999
    events.timestamp = minimum.(eachrow(colptsl)) # find which barrier touched first
    events = select!(events, Not([:side])) # select all the columns but side
    return events
end

"""----------------------------------------------------------------------
    function: shows one way to define a vertical barrier
    reference: De Prado, M. (2018) Advances in financial machine learning.
    methodology: 49
----------------------------------------------------------------------"""
function verticalbarrier(close, # dataframe of prices and dates
                         timeEvents, # vecotr of timestamps
                         daysnumber) # a number of days for vertical barrier
    timestamparray = [] # array contains index after searchsorting
    for i ∈ timeEvents .+ Dates.Day(daysnumber)
        index = searchsortedfirst(close.Dates, i) # searchsort a lag of daysnumber dates column
        append!(timestamparray, index) # append that index into 
    end
    timestamparray = timestamparray[timestamparray .< size(close)[1]]
    timestamparray = DataFrames.DataFrame(hcat(timeEvents[1:size(timestamparray)[1]], 
                     close.Dates[timestamparray]), :auto) # dataframe with start and end of an event
    return timestamparray
end
