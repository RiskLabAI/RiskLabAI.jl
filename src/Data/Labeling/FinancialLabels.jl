using GLM
using DataFrames
using TimeSeries

"""
    function: calculates the t-value of a linear trend
    refernce: De Prado, M (2020) Machine Learning for Asset Managers
    methodology: page 68, snippet 5.1
"""
function tValuesLinearRegression(
    close::TimeArray # time series of close prices
)::Float64
    data = DataFrame(Close = values(close)[:,1], Index = collect(1:length(values(close)[:,1]))) # create regression data

    OLS = lm(@formula(Close ~ Index), data) # fit linear regression
    tValue = coef(OLS)[2] / stderror(OLS)[2] # calculate t-value

    return tValue
end

"""
    function: implements the trend scanning method
    refernce: De Prado, M (2020) Machine Learning for Asset Managers
    methodology: page 68, snippet 5.2
"""
function binsFromTrend(
    molecule::Array, #  index of observations we wish to label
    close::TimeArray, # time series of close prices
    spans::Array # the list of values of span lenghts that the algorithm will evaluate, in search for the maximum absolute t-value
)
    
    outputs = TimeArray((timestamp = Vector{DateTime}(), End_Time = [], tStatistic = [], Trend = [], Close = []), timestamp = :timestamp) # initialize outputs

    for index in molecule
        tValues = TimeArray((timestamp = Vector{DateTime}(), tStatistic = []), timestamp = :timestamp) # initialize t-value series
        location = findfirst(isequal(index), timestamp(close)) # find observation location

        if location + maximum(spans) <= length(values(close)) # check if the window goes out of range
            for span in spans
                tail = timestamp(close)[location + span] # get window tail index
                windowPrices = close[index:timestamp(close)[2]-timestamp(close)[1]:tail] # get window prices
    
                tValueTrend = tValuesLinearRegression(windowPrices) # get trend t-value 
                tValue = TimeArray((timestamp = [tail], tStatistic = [tValueTrend]), timestamp = :timestamp) # get trend t-value as series
                tValues = [tValues; tValue] # update t-values
            end

            modifyTValue(x) = isinf(x) || isnan(x) ? 0.0 : abs(x) # modify for validity
            modifiedTvalues = modifyTValue.(tValues[:tStatistic]) # modify for validity
            bestTail = findwhen(modifiedTvalues[:tStatistic] .== maximum(values(modifiedTvalues[:tStatistic])))[1] # find the t-value's window tail index
            
            try # skip infinite t-values
                bestTValue = values(tValues[:tStatistic][bestTail])[1] # get best t-value
                output = TimeArray((timestamp = [index], End_Time = [timestamp(tValues)[end]], tStatistic = [bestTValue], Trend = [Int64(sign(bestTValue))], Close = [values(close[index])[1]]), timestamp = :timestamp) # create output row
                outputs = [outputs; output] # add result
            catch
            end
        end
    end

    return outputs
end