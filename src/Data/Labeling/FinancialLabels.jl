using GLM
using DataFrames
using TimeSeries

"""
    Function: Calculates the t-value of a linear trend.
    Reference: De Prado, M (2020) Machine Learning for Asset Managers.
    Methodology: Page 68, Snippet 5.1.
"""
function t_values_linear_regression(
    close::TimeArray # Time series of close prices
)::Float64
    data = DataFrame(Close = values(close)[:,1], Index = collect(1:length(values(close)[:,1]))) # Create regression data

    OLS = lm(@formula(Close ~ Index), data) # Fit linear regression
    t_value = coef(OLS)[2] / stderror(OLS)[2] # Calculate t-value

    return t_value
end

"""
    Function: Implements the trend scanning method.
    Reference: De Prado, M (2020) Machine Learning for Asset Managers.
    Methodology: Page 68, Snippet 5.2.
"""
function bins_from_trend(
    molecule::Array, # Index of observations we wish to label
    close::TimeArray, # Time series of close prices
    spans::Array # The list of values of span lengths that the algorithm will evaluate, in search for the maximum absolute t-value
)
    outputs = TimeArray((timestamp = Vector{DateTime}(), End_Time = [], tStatistic = [], Trend = [], Close = []), timestamp = :timestamp) # Initialize outputs

    for index in molecule
        t_values = TimeArray((timestamp = Vector{DateTime}(), tStatistic = []), timestamp = :timestamp) # Initialize t-value series
        location = findfirst(isequal(index), timestamp(close)) # Find observation location

        if location + maximum(spans) <= length(values(close)) # Check if the window goes out of range
            for span in spans
                tail = timestamp(close)[location + span] # Get window tail index
                window_prices = close[index:timestamp(close)[2]-timestamp(close)[1]:tail] # Get window prices
    
                t_value_trend = t_values_linear_regression(window_prices) # Get trend t-value 
                t_value = TimeArray((timestamp = [tail], tStatistic = [t_value_trend]), timestamp = :timestamp) # Get trend t-value as series
                t_values = [t_values; t_value] # Update t-values
            end

            modify_t_value(x) = isinf(x) || isnan(x) ? 0.0 : abs(x) # Modify for validity
            modified_t_values = modify_t_value.(t_values[:tStatistic]) # Modify for validity
            best_tail = findwhen(modified_t_values[:tStatistic] .== maximum(values(modified_t_values[:tStatistic])))[1] # Find the t-value's window tail index
            
            try # Skip infinite t-values
                best_t_value = values(t_values[:tStatistic][best_tail])[1] # Get best t-value
                output = TimeArray((timestamp = [index], End_Time = [timestamp(t_values)[end]], tStatistic = [best_t_value], Trend = [Int64(sign(best_t_value))], Close = [values(close[index])[1]]), timestamp = :timestamp) # Create output row
                outputs = [outputs; output] # Add result
            catch
            end
        end
    end

    return outputs
end
