"""----------------------------------------------------------------------
structure: shows the progress bar
reference: n/a
methodology: n/a
----------------------------------------------------------------------"""
mutable struct ImbalancedBar
  time::Float64
end

"""----------------------------------------------------------------------
function: shows the progress bar
reference: n/a
methodology: n/a
----------------------------------------------------------------------"""
function progressbar(value, # value of an event
                     endvalue, # length of that event
                     starttime, # start time of the event
                     barlength = 20) # length of bar
    
    percent = Float64(value)/endvalue # progress in percent
    
    if trunc(Int, round(percent*barlength) - 1) < 0 # show positive percents
        progressvalue = 0 # value percent to zero
    else 
        progressvalue = trunc(Int, round(percent*barlength) - 1) # update progressvalue
    end
    arrow = string(repeat("-", progressvalue), ">") # set the arrow
    spaces = repeat(" ", (barlength - length(arrow))) # show spaces
    # calculate remaining time to finish
    remaining = trunc(Int, ((Dates.value((Dates.now() - Dates.DateTime(1970, 1, 1, 00, 00, 00)))/
                1000 - starttime)/value)*(endvalue - value)/60) # remaining time since 1970.01.01
    message = string(arrow, spaces) # concatenate arrow and spaces
    percent = trunc(Int, round(percent*100)) # total percent completed
    println("Completed: [$message] $percent% - $remaining minutes remaining.") # print state of the progress
end

"""----------------------------------------------------------------------
function: computes the ewma, ewma var, and ewma stds
reference: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
methodology: n/a
----------------------------------------------------------------------"""
function ewma(data, # array of data
              windowlength = 100) # window for ewma
    N = size(data)[1] # length of array
    ewmamean = [] # array for output 
    ewmavariance = [] # array for output 
    ewmastd = [] # array for output 
    ?? = 2/Float64(windowlength + 1) # tune parameter for ewma
    for i ??? 1:N
        window = Array(data[1:i,1]) # Get window
        # Get weights: ??
        m = length(window)
        ?? = (1 - ??).^range(m - 1, step = -1, stop = 0) # This is reverse order to match Series order
        ewma = sum(??.*window)/sum(??) # Calculate exponential moving average
        bias = sum(??)^2/(sum(??)^2 - sum(??.^2)) # Calculate bias
        var = bias*sum(??.*(window .- ewma).^2)/sum(??) # Calculate exponential moving variance with bias
        std = sqrt(var) # Calculate standard deviation
        append!(ewmamean,ewma) # append calculated value into array 
        append!(ewmavariance,var) # append calculated value into array 
        append!(ewmastd,std) # append calculated value into array 
    end
    return ewmamean, ewmavariance, ewmastd
end

"""----------------------------------------------------------------------
function: grouping dataframe based on a feature and then calculates thresholds
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: page number
----------------------------------------------------------------------"""
function grouping(targetcol, # target column of tick dataframe
                  tickexpectedinit, # initial expected ticks
                  barsize) # initial expected size in each tick
    ??times, times = [], []
    timeprev, tickexpected, barexpectedvalue = 0, tickexpectedinit, barsize
    N = size(targetcol)[1] # number of dates in dataframe
    ??absolute, thresholds, ??s, groupingID = zeros(N), zeros(N), zeros(N), zeros(N)
    ??absolute[1], ??current = abs(targetcol[1]), targetcol[1] # set initial value of ?? and |??|
    time = Dates.value((Dates.now() - Dates.DateTime(1970, 1, 1, 00, 00, 00)))/1000 # value of time from 1970 in ms
    groupingIDcurrent = 0 # set first groupingID to 0
    for i ??? 2:N
        ??current +=  targetcol[i] # update ??current by adding next value of target
        ??s[i] = ??current # update ??s
        this??absolute = abs(??current) # absolute value of ??current
        ??absolute[i] = this??absolute  # update ??absolute
        threshold = tickexpected*barexpectedvalue # multiply expected ticks and expected value of target to calculating threshold 
        thresholds[i] = threshold  # update thresholds
        groupingID[i] = groupingIDcurrent # update groupingID
        # this stage is for going to next groupingID and resetting parameters
        if this??absolute >= threshold
            groupingIDcurrent += 1
            ??current = 0
            append!(??times, Float64(i - timeprev)) # append the length of time values that took untill passing threshold
            append!(times, i) # append the number of time value that we passed threshold in it
            timeprev = i
            tickexpected = ewma(Array(??times), trunc(Int, length(??times)))[1][end] # update expected ticks with ewma
            barexpectedvalue = abs(ewma(targetcol[1:i], trunc(Int,tickexpectedinit*1))[1][end]) # update expected value of b with ewma
        end 
        # progressbar(i,n,time) # show progress bar
    end
    return ??times, ??absolute, thresholds, times, ??s, groupingID
end

"""----------------------------------------------------------------------
function: implements Information-Driven Bars
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: page 29
----------------------------------------------------------------------"""
function infobar(tickdata, # dataframe of tick data
                 type::String = "volume", # choose "tick", "volume" or "dollar" types
                 tickexpectedinit = 2000) # The value of the expected tick
    if type == "volume"
        inputdata = tickdata.volumelabeled # use volume column with sign of log return in same day
    elseif type == "tick"
        inputdata = tickdata.label # use sign of log return column
    elseif type == "dollar"
        inputdata = tickdata.dollars # use the value of price * volume with sign of log return
    else
        println("Error: unknown type")
    end
    barexpectedvalue = abs(mean(inputdata)) # expected value of inputdata 
    ??times, ??absolute, thresholds, times, ??s, groupingID = grouping(inputdata, tickexpectedinit, barexpectedvalue) # calculate thresholds
    tickdata[!,:groupingID] = groupingID # generate groupingID column
    dates = combine(DataFrames.groupby(tickdata, :groupingID),:dates => first => :dates) # combine dates by grouping based on Id
    tickdatagrouped = DataFrames.groupby(tickdata, :groupingID) # groupe date times based on groupingID
    ohlcvdataframe = ohlcv(tickdatagrouped) # create a dataframe based on bars
    insertcols!(ohlcvdataframe, 1, :dates => dates.dates) # set date column first
    coldrop = [:groupingID] 
    ohlcvdataframe = select!(ohlcvdataframe, Not(coldrop)) # drop groupingID
    return ohlcvdataframe, ??absolute, thresholds
end

"""----------------------------------------------------------------------
    function: Takes grouped dataframe, combining and creating the new one with info. about prices and volume.
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: n/a
----------------------------------------------------------------------"""
function ohlcv(tickdatagrouped) #grouped dataframes
    # combining groups
    ohlcvdataframe = combine(tickdatagrouped, :price => first => :Open, # find open price
                   :price => maximum => :High, # find highest price
                   :price => minimum => :Low,  # find lowest price
                   :price => last => :Close, # find close price
                   :size => sum => :Volume, # find volume traded
                   AsTable([:price, :size]) =>x -> sum(x.price.*x.size)/sum(x.size), # find value of trades
                   :price => mean => :PriceMean, # mean of price
                   :price => length => :TickCount) # number of ticks
    DataFrames.rename!(ohlcvdataframe, :price_size_function => :ValueOfTrades) # rename to value of trades
    ohlcvdataframe.PriceMeanLogReturn = log.(ohlcvdataframe.PriceMean) - log.(circshift(ohlcvdataframe.PriceMean, 1)) # calculate log return
    ohlcvdataframe.PriceMeanLogReturn[1] = NaN # set first return to NaN
    return ohlcvdataframe
end

"""----------------------------------------------------------------------
function: Takes dataframe and generating time bar dataframe
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
----------------------------------------------------------------------"""
function timebar(tickdata, # dataframe of tick data
                 frequency = 5) # frequency for rounding date time
    dates = tickdata.dates #date time column
    datescopy = copy(dates)
    tickdata.dates = floor.(datescopy, Dates.Minute(frequency)) # round down date times with frequency freq
    tickdatagrouped = DataFrames.groupby(tickdata, :dates) # group date times based on rounding
    ohlcvdataframe = ohlcv(tickdatagrouped) # create a dataframe based on time bars with frequency freq
    tickdata.dates = dates # recovery date times
    return ohlcvdataframe
end

"""----------------------------------------------------------------------
function: Takes dataframe and generating tick bar dataframe
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
----------------------------------------------------------------------"""
function tickbar(tickData, # dataframe of tick data
                 tickperbar = 10,  # number of ticks in each bar
                 barsnumber = nothing) # number of bars
    # if tickperbar is not mentioned, then calculate it with number of all ticks divided by number of bars
    if tickperbar == nothing
        tickperbar = floor(size(tickdata)[1]/barsnumber)
    end
    tickdata[!, :groupingID] = [x??tickperbar for x in 0:size(tickdata)[1] - 1] # generate groupingID column for division based on ticks
    tickgrouped = copy(tickdata)
    dates = combine(DataFrames.groupby(tickgrouped, :groupingID), :dates => first => :dates) # combine dates 
    tickdatagrouped = DataFrames.groupby(tickgrouped, :groupingID) # group date times based on groupingID
    ohlcvdataframe = ohlcv(tickdatagrouped) # create a dataframe based on tickperbar ticks
    insertcols!(ohlcvdataframe, 1, :dates => dates.dates) # set date column first
    coldrop = [:groupingID] 
    ohlcvdataframe = select!(ohlcvdataframe, Not(coldrop)) # drop groupingID column
    return ohlcvdataframe
end

"""----------------------------------------------------------------------
function: Takes dataframe and generating volume bar dataframe
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
----------------------------------------------------------------------"""
function volumebar(tickdata, # dataframe of tick data
                   volumeperbar = 10000, # volumes in each bar
                   barsnumber = nothing) # number of bars
    tickdata[!, :volumecumulative] = cumsum(tickdata.size) # cumulative sum of size(volume)
    # if volumeperbar is not mentioned, then calculate it with all volumes divided by number of bars
    if volumeperbar == nothing
        TotalVolume = tickdata.volumecumulative[end]
        volumeperbar = TotalVolume/barsnumber
        volumeperbar = round(volumeperbar; sigdigits = 2) # round to the nearest hundred
    end
    tickdata[!,:groupingID] = [x??volumeperbar for x in tickdata[!, :volumecumulative]] # generate groupingID column for division based on volums
    tickgrouped = copy(tickdata)
    dates = combine(DataFrames.groupby(tickgrouped, :groupingID),:dates => first => :dates) # combine dates based on Id
    tickdatagrouped = DataFrames.groupby(tickgrouped, :groupingID) # group date times based on groupingID
    ohlcvdataframe = ohlcv(tickdatagrouped) # create a dataframe based on volumeperbar bars
    insertcols!(ohlcvdataframe, 1, :dates => dates.dates) # set date column first 
    coldrop = [:groupingID] 
    ohlcvdataframe = select!(ohlcvdataframe, Not(coldrop)) # drop groupingID column
    return ohlcvdataframe
end

"""----------------------------------------------------------------------
function: Takes dataframe and generating volume bar dataframe
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
----------------------------------------------------------------------"""
function dollarbar(tickdata, # dataframe of tick data
                   dollarperbar = 100000, # dollars in each bar
                   barsnumber = nothing) # number of bars
    tickdata[!, :dollar] = tickdata.price.*tickdata.size # generate dollar column by multiplying price and size
    tickdata[!, :CumulativeDollars] = cumsum(tickdata.dollar) # cumulative sum of dollars
    # if dollarperbar is not mentioned, then calculate it with dollars divided by number of bars
    if dollarperbar == nothing
        dollarstotal = tickdata.CumulativeDollars[end]
        dollarperbar = dollarstotal/barsnumber
        dollarperbar = round(dollarperbar; sigdigits = 2) # round to the nearest hundred
    end
    tickdata[!, :groupingID] = [x??dollarperbar for x in tickdata[!, :CumulativeDollars]] # generate groupingID column for division based on dollars
    tickgrouped = copy(tickdata)
    dates = combine(DataFrames.groupby(tickgrouped, :groupingID),:dates => first => :dates) # combine dates based on Id
    tickdatagrouped = DataFrames.groupby(tickgrouped, :groupingID) # group date times based on groupingID
    ohlcvdataframe = ohlcv(tickdatagrouped) # create a dataframe based on volume_per_bar bars
    insertcols!(ohlcvdataframe, 1, :dates => dates.dates) # set date column first
    coldrop = [:groupingID] 
    ohlcvdataframe = select!(ohlcvdataframe, Not(coldrop)) # drop groupingID column
    return ohlcvdataframe
end
