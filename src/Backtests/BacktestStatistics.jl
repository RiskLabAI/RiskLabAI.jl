using Dates
using TimeSeries
using DayCounts

"""
Returns the timing of bets when positions flatten or flip.

:param target_positions: TimeArray of target positions.
:return: TimeArray containing timestamps of bet timings.
"""
function betTiming(targetPositions::TimeArray)
    colName(x) = colnames(x)[1]
    
    zeroPositions = timestamp(targetPositions[targetPositions[colName(targetPositions)] .== 0])
    
    laggedNonZeroPositions = lag(targetPositions, padding=true)
    laggedNonZeroPositions = timestamp(laggedNonZeroPositions[laggedNonZeroPositions[colName(laggedNonZeroPositions)] .!= 0])
    
    bets = intersect(zeroPositions, laggedNonZeroPositions)
    zeroPositions = targetPositions[2:end] .* targetPositions[1:end-1]
    bets = sort(union(bets, timestamp(zeroPositions[zeroPositions[colName(zeroPositions)] .< 0])))
    
    if timestamp(targetPositions)[end] ∉ bets
        append!(bets, [timestamp(targetPositions)[end]])
    end
    
    return bets
end

"""
Derives the average holding period (in days) using average entry time pairing algorithm.

:param target_positions: TimeArray of target positions.
:return: Tuple containing TimeArray of holding periods and average holding period.
"""
function holdingPeriod(targetPositions::TimeArray)
    holdPeriod, timeEntry = DataFrame(index=[], dT=[], w=[]), 0.0
    
    positionDifference, timeDifference = diff(targetPositions; padding=true), 
                                           Dates.value.(Day.(timestamp(targetPositions) .- timestamp(targetPositions)[1]))
    
    for i in 2:length(targetPositions)
        if values(positionDifference)[i] * values(targetPositions)[i-1] >= 0
            if values(targetPositions)[i] != 0
                timeEntry = (timeEntry * values(targetPositions)[i-1] + 
                              timeDifference[i] * values(positionDifference)[i]) / values(targetPositions)[i]
            end
        else
            if values(targetPositions)[i] * values(targetPositions)[i-1] < 0
                append!(holdPeriod, DataFrame(index=[timestamp(targetPositions)[i]], 
                                                dT=[timeDifference[i] - timeEntry], w=[abs(values(targetPositions)[i-1])]))
                timeEntry = timeDifference[i]
            else
                append!(holdPeriod, DataFrame(index=[timestamp(targetPositions)[i]], 
                                                dT=[timeDifference[i] - timeEntry], w=[abs(values(positionDifference)[i])]))
            end
        end
    end
    
    if sum(holdPeriod[:, :w]) > 0
        meanHold = sum(holdPeriod[:, :dT] .* holdPeriod[:, :w]) / sum(holdPeriod[:, :w])
    else
        return nothing
    end
    
    return (TimeArray(holdPeriod; timestamp=:index), meanHold)
end

"""
Derives the algorithm for calculating HHI concentration.

:param returns: TimeArray of returns series.
:return: Tuple of positive returns HHI, negative returns HHI, and concentrated HHI over time.
"""
function hhiConcentration(returns::TimeArray)
    colName(x) = colnames(x)[1]
    
    returnsHHIPositive = HHI(returns[returns[colName(returns)] .>= 0])
    returnsHHINegative = HHI(returns[returns[colName(returns)] .< 0])
    
    returnsGrouped = groupby(transform(DataFrame(returns), :timestamp => x->yearmonth.(x)), :timestampFunction)
    timeConcentratedHHI = HHI(combine(returnsGrouped, Symbol(colName(returns)) => length, renamecols=false)[:, colName(returns)])
    
    return (returnsHHIPositive, returnsHHINegative, timeConcentratedHHI)
end

"""
Calculates the Herfindahl-Hirschman Index (HHI).

:param bet_returns: Bet returns series.
:return: HHI value.
"""
function hhi(betReturns)
    if length(betReturns) <= 2
        return nothing
    end
    
    weight = values(betReturns) ./ sum(values(betReturns))
    hhiValue = sum(weight.^2)
    hhiValue = (hhiValue - length(betReturns) ^ -1) / (1.0 - length(betReturns) ^ -1)
    
    return hhiValue
end

"""
Computes series of drawdowns and the time under water associated with them.

:param series: TimeArray of returns or dollar performance.
:param dollars: Boolean indicating whether returns or dollar performance.
:return: Tuple containing TimeArray of drawdowns, TimeArray of time under water, and drawdown analysis DataFrame.
"""
function computeDrawdownsTimeUnderWater(series::TimeArray, dollars::Bool = false)
    colName(x) = colnames(x)[1]
    
    seriesDF = DataFrame(TimeSeries.rename(series, Symbol(colName(series)) => :PnL))
    seriesDF[!, :HWM] = [maximum(seriesDF[1:i, :PnL]) for i in 1:length(seriesDF.PnL)]
    drawdownAnalysis = DataFrame()
    
    function processGroups(group)
        if nrow(group) <= 1
            return
        end
        
        result = DataFrame()
        result[!, :Start] = [group[1, :timestamp]]
        result[!, :Stop] = [group[end, :timestamp]]
        result[!, :HWM] = [group[1, :HWM]]
        result[!, :Min] = [minimum(group.PnL)]
        result[!, "Min. Time"] = [group.timestamp[group.PnL .== minimum(group.PnL)][1]]
        
        return result
    end
    
    for i in collect(groupby(seriesDF, :HWM))
        processed = processGroups(DataFrame(i))
        if !isnothing(processed)
            drawdownAnalysis = vcat(drawdownAnalysis, processed)
        end
    end
    
    if dollars
        drawdowns = TimeArray(DataFrame(drawDowns = drawdownAnalysis.HWM .- drawdownAnalysis.Min, ts = drawdownAnalysis.Start), timestamp = :ts)
    else
        drawdowns = TimeArray(DataFrame(drawDowns = 1 .- drawdownAnalysis.Min ./ drawdownAnalysis.HWM , ts = drawdownAnalysis.Start), timestamp = :ts)
    end
    
    timeUnderWater = [yearfrac(drawdownAnalysis.Start[i], drawdownAnalysis.Stop[i], DayCounts.ActualActualISDA()) for i in 1:nrow(drawdownAnalysis)]
    timeUnderWater = TimeArray(DataFrame(timeUnderWater = timeUnderWater, ts = drawdownAnalysis.Start), timestamp = :ts)
    
    return (drawdowns, timeUnderWater, drawdownAnalysis)
end
