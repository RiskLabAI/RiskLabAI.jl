using GLM
using DataFrames
using TimeSeries

"""
Function: Calculate the t-value of a linear trend.

Calculates the t-value of a linear trend using linear regression on the provided time series of close prices and their indices.

:param close::TimeArray: Time series of close prices.
:return: tValue::Float64: The t-value of the linear trend.
"""
function calculateLinearTValue(
    close::TimeArray{Float64, 1}
)::Float64
    data = DataFrame(Close = values(close), Index = 1:length(close)) # Create regression data

    OLS = lm(@formula(Close ~ Index), data) # Fit linear regression
    tValue = coef(OLS)[2] / stderror(OLS)[2] # Calculate t-value

    return tValue
end

"""
Function: Implement the trend scanning method.

Implements the trend scanning method based on the provided molecule of observation indices, time series of close prices, and spans of window lengths.

:param molecule::Array{Int}: Index of observations to label.
:param close::TimeArray{Float64, 1}: Time series of close prices.
:param spans::Array{Int}: List of span lengths to evaluate.
:return: outputs::TimeArray: Time series of trend scanning results.
"""
function trendScanning(
    molecule::Array{Int},
    close::TimeArray{Float64, 1},
    spans::Array{Int}
)::TimeArray{Float64, 2}
    outputs = TimeArray((timestamp = DateTime[], End_Time = DateTime[], tStatistic = Float64[], Trend = Int[], Close = Float64[]), timestamp = :timestamp) # Initialize outputs

    for index in molecule
        tValues = TimeArray((timestamp = DateTime[], tStatistic = Float64[]), timestamp = :timestamp) # Initialize t-value series
        location = findfirst(isequal(index), timestamp(close)) # Find observation location

        if location + maximum(spans) <= length(values(close)) # Check if the window goes out of range
            for span in spans
                tail = timestamp(close)[location + span] # Get window tail index
                windowPrices = close[index:timestamp(close)[2] - timestamp(close)[1]:tail] # Get window prices
    
                tValueTrend = calculateLinearTValue(windowPrices) # Get trend t-value 
                tValue = TimeArray((timestamp = [tail], tStatistic = [tValueTrend]), timestamp = :timestamp) # Get trend t-value as series
                tValues = [tValues; tValue] # Update t-values
            end

            modifyTValue(x) = isinf(x) || isnan(x) ? 0.0 : abs(x) # Modify for validity
            modifiedTValues = modifyTValue.(tValues[:tStatistic]) # Modify for validity
            bestTail = findwhen(modifiedTValues[:tStatistic] .== maximum(values(modifiedTValues[:tStatistic])))[1] # Find the t-value's window tail index
            
            try # Skip infinite t-values
                bestTValue = values(tValues[:tStatistic][bestTail])[1] # Get best t-value
                output = TimeArray((timestamp = [index], End_Time = [timestamp(tValues)[end]], tStatistic = [bestTValue], Trend = [Int(sign(bestTValue))], Close = [values(close[index])[1]]), timestamp = :timestamp) # Create output row
                outputs = [outputs; output] # Add result
            catch
            end
        end
    end

    return outputs
end
