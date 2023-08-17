using GLM
using DataFrames
using TimeSeries

"""
    calculateLinearTValue(close::TimeArray{Float64, 1})::Float64

Calculate the t-value of a linear trend using linear regression on the provided time series of close prices and their indices.

# Arguments
- `close::TimeArray{Float64, 1}`: Time series of close prices.

# Returns
- `tValue::Float64`: The t-value of the linear trend.

# Mathematical Formula
The t-value is calculated using the formula:
t-value = coefficient / standard error

where coefficient is the slope of the linear regression line, and the standard error is the standard error of the coefficient.
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
    trendScanning(molecule::Array{Int}, close::TimeArray{Float64, 1}, spans::Array{Int})::TimeArray{Float64, 2}

Implements the trend scanning method based on the provided molecule of observation indices, time series of close prices, and spans of window lengths.

# Arguments
- `molecule::Array{Int}`: Index of observations to label.
- `close::TimeArray{Float64, 1}`: Time series of close prices.
- `spans::Array{Int}`: List of span lengths to evaluate.

# Returns
- `outputs::TimeArray{Float64, 2}`: Time series of trend scanning results.
"""
function trendScanning(
    molecule::Array{Int},
    close::TimeArray{Float64, 1},
    spans::Array{Int}
)::TimeArray{Float64, 2}
    outputs = TimeArray((timestamp = DateTime[], EndTime = DateTime[], tStatistic = Float64[], Trend = Int[], Close = Float64[]), timestamp = :timestamp) # Initialize outputs

    for index in molecule
        tValues = TimeArray((timestamp = DateTime[], tStatistic = Float64[]), timestamp = :timestamp) # Initialize t-value series
        location = findfirst(isequal(index), timestamp(close)) # Find observation location

        if location + maximum(spans) <= length(values(close)) # Check if the window goes out of range
            for span in spans
                tail = timestamp(close)[location + span] # Get window tail index
                windowPrices = close[index:Dates.Day(1):tail] # Get window prices
    
                tValueTrend = calculateLinearTValue(windowPrices) # Get trend t-value 
                tValue = TimeArray((timestamp = [tail], tStatistic = [tValueTrend]), timestamp = :timestamp) # Get trend t-value as series
                tValues = vcat(tValues, tValue) # Update t-values
            end

            modifyTValue(x) = isinf(x) || isnan(x) ? 0.0 : abs(x) # Modify for validity
            modifiedTValues = modifyTValue.(tValues[:tStatistic]) # Modify for validity
            bestTail = argmax(modifiedTValues[:tStatistic]) # Find the t-value's window tail index
            
            try # Skip infinite t-values
                bestTValue = values(tValues[:tStatistic][bestTail])[1] # Get best t-value
                output = TimeArray((timestamp = [index], EndTime = [timestamp(tValues)[end]], tStatistic = [bestTValue], Trend = [Int(sign(bestTValue))], Close = [values(close[index])[1]]), timestamp = :timestamp) # Create output row
                outputs = vcat(outputs, output) # Add result
            catch
            end
        end
    end

    return outputs
end
