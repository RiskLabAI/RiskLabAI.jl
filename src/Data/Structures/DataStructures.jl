"""
    function: shows the progress bar
    reference: n/a
    methodology: n/a
"""
function progressBar(value, # value of an event
                     endValue, # length of that event
                     startTime, # start time of the event
                     barLength = 20) # length of bar
    
    percent = Float64(value)/endValue # progress in percent
    
    if trunc(Int, round(percent*barLength) - 1) < 0 # show positive percents
        progressValue = 0 # value percent to zero
    else 
        progressValue = trunc(Int, round(percent*barLength) - 1) # update progressValue
    end
    arrow = string(repeat("-", progressValue), ">") # set the arrow
    spaces = repeat(" ", (barLength - length(arrow))) # show spaces
    # calculate remaining time to finish
    remaining = trunc(Int, ((Dates.value((Dates.now() - Dates.DateTime(1970, 1, 1, 00, 00, 00)))/
                1000 - startTime)/value)*(endValue - value)/60) # remaining time since 1970.01.01
    message = string(arrow, spaces) # concatenate arrow and spaces
    percent = trunc(Int, round(percent*100)) # total percent completed
    println("Completed: [$message] $percent% - $remaining minutes remaining.") # print state of the progress
end

"""
    function: computes the ewma, ewma var, and ewma stds
    reference: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
    methodology: n/a
"""
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

"""
    function: grouping dataframe based on a feature and then calculates thresholds
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: page number
"""
function grouping(targetCol, # target column of tick dataframe
                  tickExpectedInit, # initial expected ticks
                  barSize) # initial expected size in each tick
    Δtimes, times = [], []
    timePrev, tickExpected, barExpectedValue = 0, tickExpectedInit, barSize
    N = size(targetCol)[1] # number of dates in dataframe
    θabsolute, thresholds, θs, groupingID = zeros(N), zeros(N), zeros(N), zeros(N)
    θabsolute[1], θcurrent = abs(targetCol[1]), targetCol[1] # set initial value of θ and |θ|
    time = Dates.value((Dates.now() - Dates.DateTime(1970, 1, 1, 00, 00, 00)))/1000 # value of time from 1970 in ms
    groupingIDCurrent = 0 # set first groupingID to 0
    for i ∈ 2:N
        θcurrent +=  targetCol[i] # update θcurrent by adding next value of target
        θs[i] = θcurrent # update θs
        thisθabsolute = abs(θcurrent) # absolute value of θcurrent
        θabsolute[i] = thisθabsolute  # update θabsolute
        threshold = tickExpected*barExpectedValue # multiply expected ticks and expected value of target to calculating threshold 
        thresholds[i] = threshold  # update thresholds
        groupingID[i] = groupingIDCurrent # update groupingID
        # this stage is for going to next groupingID and resetting parameters
        if thisθabsolute >= threshold
            groupingIDCurrent += 1
            θcurrent = 0
            append!(Δtimes, Float64(i - timePrev)) # append the length of time values that took untill passing threshold
            append!(times, i) # append the number of time value that we passed threshold in it
            timePrev = i
            tickExpected = ewma(Array(Δtimes), trunc(Int, length(Δtimes)))[1][end] # update expected ticks with ewma
            barExpectedValue = abs(ewma(targetCol[1:i], trunc(Int,tickExpectedInit*1))[1][end]) # update expected value of b with ewma
        end 
        # progressBar(i,n,time) # show progress bar
    end
    return Δtimes, θabsolute, thresholds, times, θs, groupingID
end

"""
    function: implements Information-Driven Bars
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: page 29
"""
function infoBar(tickData, # dataframe of tick data
                 type::String = "volume", # choose "tick", "volume" or "dollar" types
                 tickExpectedInit = 2000) # The value of the expected tick
    if type == "volume"
        inputData = tickData.volumelabeled # use volume column with sign of log return in same day
    elseif type == "tick"
        inputData = tickData.label # use sign of log return column
    elseif type == "dollar"
        inputData = tickData.dollars # use the value of price * volume with sign of log return
    else
        println("Error: unknown type")
    end
    barExpectedValue = abs(mean(inputData)) # expected value of inputData 
    Δtimes, θabsolute, thresholds, times, θs, groupingID = grouping(inputData, tickExpectedInit, barExpectedValue) # calculate thresholds
    tickData[!,:groupingID] = groupingID # generate groupingID column
    dates = combine(DataFrames.groupby(tickData, :groupingID),:dates => first => :dates) # combine dates by grouping based on Id
    tickDataGrouped = DataFrames.groupby(tickData, :groupingID) # groupe date times based on groupingID
    ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on bars
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates) # set date column first
    colDrop = [:groupingID] 
    ohlcvDataframe = select!(ohlcvDataframe, Not(colDrop)) # drop groupingID
    return ohlcvDataframe, θabsolute, thresholds
end

"""
    function: Takes grouped dataframe, combining and creating the new one with info. about prices and volume.
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: n/a
"""
function ohlcv(tickDataGrouped) #grouped dataframes
    # combining groups
    ohlcvDataframe = combine(tickDataGrouped, :price => first => :Open, # find open price
                   :price => maximum => :High, # find highest price
                   :price => minimum => :Low,  # find lowest price
                   :price => last => :Close, # find close price
                   :size => sum => :Volume, # find volume traded
                   AsTable([:price, :size]) =>x -> sum(x.price.*x.size)/sum(x.size), # find value of trades
                   :price => mean => :PriceMean, # mean of price
                   :price => length => :TickCount) # number of ticks
    DataFrames.rename!(ohlcvDataframe, :price_size_function => :ValueOfTrades) # rename to value of trades
    ohlcvDataframe.PriceMeanLogReturn = log.(ohlcvDataframe.PriceMean) - log.(circshift(ohlcvDataframe.PriceMean, 1)) # calculate log return
    ohlcvDataframe.PriceMeanLogReturn[1] = NaN # set first return to NaN
    return ohlcvDataframe
end

"""
    function: Takes dataframe and generating time bar dataframe
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: n/a
"""
function timeBar(tickData, # dataframe of tick data
                 frequency = 5) # frequency for rounding date time
    dates = tickData.dates #date time column
    datesCopy = copy(dates)
    tickData.dates = floor.(datesCopy, Dates.Minute(frequency)) # round down date times with frequency freq
    tickDataGrouped = DataFrames.groupby(tickData, :dates) # group date times based on rounding
    ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on time bars with frequency freq
    tickData.dates = dates # recovery date times
    return ohlcvDataframe
end

"""
    function: Takes dataframe and generating tick bar dataframe
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: n/a
"""
function tickBar(tickData, # dataframe of tick data
                 tickPerBar = 10,  # number of ticks in each bar
                 numberBars = nothing) # number of bars
    # if tickPerBar is not mentioned, then calculate it with number of all ticks divided by number of bars
    if tickPerBar == nothing
        tickPerBar = floor(size(tickData)[1]/numberBars)
    end
    tickData[!, :groupingID] = [x÷tickPerBar for x in 0:size(tickData)[1] - 1] # generate groupingID column for division based on ticks
    tickGrouped = copy(tickData)
    dates = combine(DataFrames.groupby(tickGrouped, :groupingID), :dates => first => :dates) # combine dates 
    tickDataGrouped = DataFrames.groupby(tickGrouped, :groupingID) # group date times based on groupingID
    ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on tickPerBar ticks
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates) # set date column first
    colDrop = [:groupingID] 
    ohlcvDataframe = select!(ohlcvDataframe, Not(colDrop)) # drop groupingID column
    return ohlcvDataframe
end

"""
function: Takes dataframe and generating volume bar dataframe
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
"""
function volumeBar(tickData, # dataframe of tick data
                   volumePerBar = 10000, # volumes in each bar
                   numberBars = nothing) # number of bars
    tickData[!, :volumecumulated] = cumsum(tickData.size) # cumulative sum of size(volume)
    # if volumePerBar is not mentioned, then calculate it with all volumes divided by number of bars
    if volumePerBar == nothing
        totalVolume = tickData.volumecumulated[end]
        volumePerBar = totalVolume/numberBars
        volumePerBar = round(volumePerBar; sigdigits = 2) # round to the nearest hundred
    end
    tickData[!,:groupingID] = [x÷volumePerBar for x in tickData[!, :volumecumulated]] # generate groupingID column for division based on volums
    tickGrouped = copy(tickData)
    dates = combine(DataFrames.groupby(tickGrouped, :groupingID),:dates => first => :dates) # combine dates based on Id
    tickDataGrouped = DataFrames.groupby(tickGrouped, :groupingID) # group date times based on groupingID
    ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on volumePerBar bars
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates) # set date column first 
    colDrop = [:groupingID] 
    ohlcvDataframe = select!(ohlcvDataframe, Not(colDrop)) # drop groupingID column
    return ohlcvDataframe
end

"""
    function: Takes dataframe and generating volume bar dataframe
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: n/a
"""
function dollarBar(tickData, # dataframe of tick data
                   dollarPerBar = 100000, # dollars in each bar
                   numberBars = nothing) # number of bars
    tickData[!, :dollar] = tickData.price.*tickData.size # generate dollar column by multiplying price and size
    tickData[!, :CumulativeDollars] = cumsum(tickData.dollar) # cumulative sum of dollars
    # if dollarPerBar is not mentioned, then calculate it with dollars divided by number of bars
    if dollarPerBar == nothing
        dollarsTotal = tickData.CumulativeDollars[end]
        dollarPerBar = dollarsTotal/numberBars
        dollarPerBar = round(dollarPerBar; sigdigits = 2) # round to the nearest hundred
    end
    tickData[!, :groupingID] = [x÷dollarPerBar for x in tickData[!, :CumulativeDollars]] # generate groupingID column for division based on dollars
    tickGrouped = copy(tickData)
    dates = combine(DataFrames.groupby(tickGrouped, :groupingID),:dates => first => :dates) # combine dates based on Id
    tickDataGrouped = DataFrames.groupby(tickGrouped, :groupingID) # group date times based on groupingID
    ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on volume_per_bar bars
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates) # set date column first
    colDrop = [:groupingID] 
    ohlcvDataframe = select!(ohlcvDataframe, Not(colDrop)) # drop groupingID column
    return ohlcvDataframe
end
   
"""
    function: Calculates hedging weights using cov, risk distribution(risk_dist) and σ
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: page 36
"""
function PCAWeights(cov, # covariance matrix
                    riskDisturbution = nothing,  # risk distribution
                    σ = 1.0) # risk target
    Λ = eigvals(cov)
    V = eigvecs(cov)
    indices = reverse(sortperm(Λ)) # arguments for sorting eVal descending
    Λ = Λ[indices] # sort eigen values
    V = V[:, indices] # sort eigen vectors
    # if riskDisturbution is nothing, it will assume all risk must be allocated to the principal component with
    # smallest eigenvalue, and the weights will be the last eigenvector re-scaled to match σ
    if riskDisturbution == nothing
        riskDisturbution = zeros(size(cov)[1])
        riskDisturbution[end] = 1.0
    end
    loads = σ*(riskDisturbution./Λ).^0.5 # represent the allocation in the new (orthogonal) basis
    weights = V*loads # calculate weights
    return weights
end

"""
    function: Implementation of the symmetric CUSUM filter
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: page 39
"""
function events(input, # dataframe of prices and dates
                threshold) # threshold
    timeEvents, shiftPositive, shiftNegative = [], 0, 0
    # dataframe with price differences
    Δprice = DataFrames.DataFrame(hcat(input[2:end, 1], diff(input[:, 2])), :auto) 
    for i ∈ Δprice[:, 1]
        # compute shiftNegative/shiftPositive with min/max of 0 and ΔPRICE in each day
        shiftPositive = max(0, shiftPositive+Δprice[Δprice[:, 1] .== i, 2][1]) # compare price diff with zero
        shiftNegative = min(0, shiftNegative+Δprice[Δprice[:, 1] .== i, 2][1]) # compare price diff with zero
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
