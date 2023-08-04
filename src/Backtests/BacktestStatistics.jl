using Dates
using TimeSeries
using DayCounts

"""
    function: returns timing of bets when positions flatten or flip
    reference: De Prado, M (2018) Advances in financial machine learning
    methodology: page 197, snippet 14.1
"""
function betTiming(targetPositions::TimeArray) # series of target positions

    columnName(x) = colnames(x)[1] # get first column name

    zeroPositions = timestamp(targetPositions[targetPositions[columnName(targetPositions)] .== 0]) # get zero positions timestamps
    
    laggedNonZeroPositions = lag(targetPositions, padding=true) # lag the positions
    laggedNonZeroPositions = timestamp(laggedNonZeroPositions[laggedNonZeroPositions[columnName(laggedNonZeroPositions)] .!= 0]) # get lagged non zero positions timestamps
    
    bets = intersect(zeroPositions, laggedNonZeroPositions) # get flattening timestamps
    zeroPositions = targetPositions[2:end] .* targetPositions[1:end-1] # find flips
    bets = sort(union(bets, timestamp(zeroPositions[zeroPositions[columnName(zeroPositions)] .< 0]))) # get flips' timestamps
    
    if timestamp(targetPositions)[end] ∉ bets
        append!(bets, [timestamp(targetPositions)[end]]) # add the last bet
    end
    
    return bets
end

"""
    function: derives avgerage holding period (in days) using avg entry time pairing algo
    refernce: De Prado, M (2018) Advances in financial machine learning
    methodology: page 197, snippet 14.2
"""
function holdingPeriod(targetPositions::TimeArray) # series of target positions

    holdPeriod, timeEntry = DataFrame(index = [], dT = [], w = []), 0.0 # initialize holding periods and entry time

    positionDifference, timeDifference = diff(targetPositions; padding=true), # find position difference and elapssed time
                                              Dates.value.(Day.(timestamp(targetPositions) .- timestamp(targetPositions)[1])) 

    for i in 2:length(targetPositions)
        if values(positionDifference)[i] * values(targetPositions)[i-1] >= 0 # find if position is increased or unchanged
            if values(targetPositions)[i] != 0 # find if target position is non zero
                timeEntry = (timeEntry * values(targetPositions)[i-1] + timeDifference[i] * values(positionDifference)[i]) / values(targetPositions)[i] # update entry time
            end
        else # find if position is decreased
            if values(targetPositions)[i] * values(targetPositions)[i-1] < 0 # find if there is a flip
                append!(holdPeriod, DataFrame(index = [timestamp(targetPositions)[i]], # add the new holding period
                                      dT = [timeDifference[i] - timeEntry], w = [abs(values(targetPositions)[i-1])]))        
                timeEntry = timeDifference[i] # reset entry time
            else
                append!(holdPeriod, DataFrame(index = [timestamp(targetPositions)[i]], # add the new holding period
                                      dT = [timeDifference[i] - timeEntry], w = [abs(values(positionDifference)[i])]))                      
            end
        end
    end
    if sum(holdPeriod[:, :w]) > 0 # find if there are holding periods
        mean = sum(holdPeriod[:, :dT] .* holdPeriod[:, :w]) / sum(holdPeriod[:, :w]) # calculate the average holding period
    else 
        return nothing
    end
    return (TimeArray(holdPeriod; timestamp = :index), mean)
end    

"""
    function: derives the algorithm for deriving hhi concentration
    refernce: De Prado, M (2018) Advances in financial machine learning
    methodology: page 201, snippet 14.3
"""
function HHIConcentration(returns::TimeArray) # returns series

    columnName(x) = colnames(x)[1] # get first column name

    returnsHHIPositive = HHI(returns[returns[columnName(returns)] .>= 0]) # get concentration of positive returns per bet
    returnsHHINegative = HHI(returns[returns[columnName(returns)] .< 0]) # get concentration of negative returns per bet

    returnsGroupBy = groupby(transform(DataFrame(returns), :timestamp => x->yearmonth.(x)), :timestamp_function) # group by month
    timeConcentratedHHI = HHI(combine(returnsGroupBy, Symbol(columnName(returns)) => length, renamecols = false)[:, columnName(returns)]) # get concentr. bets/month
    
    return (returnsHHIPositive, returnsHHINegative, timeConcentratedHHI)
end

function HHI(betReturns) # bet returns to do: argument type

    if length(betReturns) <= 2 # find returns length is less than 3
        return nothing
    end

    weight = values(betReturns) ./ sum(values(betReturns)) # Calculate weights
    hhi = sum(weight.^2) # sum of squared weights
    hhi = (hhi - length(betReturns) ^ -1) / (1.0 - length(betReturns) ^ -1) # calculate hhi with squared weights
    return hhi
end

"""
    function: computes series of drawdowns and the time under water associated with them
    refernce: De Prado, M (2018) Advances in financial machine learning
    methodology: page 201, snippet 14.4
"""
function computeDrawDownsTimeUnderWater(series::TimeArray, # series of returns or dollar performance
                                        dollars::Bool = false) # returns or dollar performance

    columnName(x) = colnames(x)[1] # get first column name

    seriesDf = DataFrame(TimeSeries.rename(series, Symbol(columnName(series)) => :PnL)) # convert to DataFrame
    seriesDf[!, :HWM] = [maximum(seriesDf[1:i, :PnL]) for i in 1:length(seriesDf.PnL)] # find max of expanding window
    drawdownAnalysis = DataFrame() # initiate dataframe    

    function processGroups(group)
        if nrow(group) <= 1 # check if there is a drawdown 
            return 
        end    

        result = DataFrame() 
        result[!, :Start] = [group[1, :timestamp]] # find drawdown beginning
        result[!, :Stop] = [group[end, :timestamp]] # find drawdown ending
        result[!, :HWM] = [group[1, :HWM]] # find drawdown high watermark
        result[!, :Min] = [minimum(group.PnL)] # find the maximum drawdown 
        result[!, "Min. Time"] = [group.timestamp[group.PnL .== minimum(group.PnL)][1]] # find the maximum drawdown time

        return result
    end

    for i in collect(groupby(seriesDf, :HWM)) # group by high water mark
        processed = processGroups(DataFrame(i)) # process drawdowns
        if !isnothing(processed) # check if it is not null
            drawdownAnalysis = vcat(drawdownAnalysis, processed) # aggregate drawdowns
        end    
    end  

    if dollars # find if the input is returns or dollar performance series
        drawDowns = TimeArray(DataFrame(drawDowns = drawdownAnalysis.HWM .- drawdownAnalysis.Min, ts = drawdownAnalysis.Start), timestamp = :ts) # calculate draw downs
    else
        drawDowns = TimeArray(DataFrame(drawDowns = 1 .- drawdownAnalysis.Min ./ drawdownAnalysis.HWM , ts = drawdownAnalysis.Start), timestamp = :ts) # calculate draw downs
    end

    timeUnderWater = [yearfrac(drawdownAnalysis.Start[i], drawdownAnalysis.Stop[i], DayCounts.ActualActualISDA()) for i in range(1, nrow(drawdownAnalysis))] # convert time under water to years
    timeUnderWater = TimeArray(DataFrame(timeUnderWater = timeUnderWater, ts = drawdownAnalysis.Start), timestamp = :ts) # create TimeArray

    return (drawDowns, timeUnderWater, drawdownAnalysis)
end