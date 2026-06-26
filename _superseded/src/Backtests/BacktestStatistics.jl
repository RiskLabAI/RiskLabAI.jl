using Dates
using TimeSeries
using DayCounts
using DataFrames

"""
    betTiming(targetPositions::TimeArray)::Vector{DateTime}

Returns the timing of bets when positions flatten or flip.

# Arguments
- `targetPositions::TimeArray`: TimeArray of target positions.

# Returns
- `Vector{DateTime}`: Vector containing timestamps of bet timings.
"""
function betTiming(targetPositions::TimeArray)::Vector{DateTime}
    colName = names(targetPositions)[1]

    zeroPositions = timestamp(targetPositions[targetPositions[colName] .== 0])
    laggedNonZeroPositions = lag(targetPositions, padding=true)
    laggedNonZeroPositions = timestamp(laggedNonZeroPositions[laggedNonZeroPositions[colName] .!= 0])

    bets = intersect(zeroPositions, laggedNonZeroPositions)
    zeroPositions = targetPositions[2:end] .* targetPositions[1:end-1]
    bets = sort(union(bets, timestamp(zeroPositions[zeroPositions[colName] .< 0])))

    if timestamp(targetPositions)[end] ∉ bets
        append!(bets, [timestamp(targetPositions)[end]])
    end

    return bets
end

"""
    holdingPeriod(targetPositions::TimeArray)::Tuple{TimeArray, Float64}

Derives the average holding period (in days) using average entry time pairing algorithm.

# Arguments
- `targetPositions::TimeArray`: TimeArray of target positions.

# Returns
- `Tuple{TimeArray, Float64}`: Tuple containing TimeArray of holding periods and average holding period.

# Methodology
1. Compute the position difference and time difference arrays.
2. Iterate through the target positions.
3. Update the entry time based on position changes.
4. Record holding periods and associated weights.
5. Calculate the weighted average holding period.
"""
function holdingPeriod(targetPositions::TimeArray)::Tuple{TimeArray, Float64}
    holdPeriod = DataFrame(index=DateTime[], dT=Float64[], w=Float64[])
    timeEntry = 0.0

    positionDifference = diff(targetPositions; padding=true)
    timeDifference = Dates.value.(Day.(timestamp(targetPositions) .- timestamp(targetPositions)[1]))

    for i in 2:length(targetPositions)
        if values(positionDifference)[i] * values(targetPositions)[i-1] >= 0
            if values(targetPositions)[i] != 0
                timeEntry = (timeEntry * values(targetPositions)[i-1] +
                             timeDifference[i] * values(positionDifference)[i]) / values(targetPositions)[i]
            end
        else
            if values(targetPositions)[i] * values(targetPositions)[i-1] < 0
                push!(holdPeriod, [timestamp(targetPositions)[i], timeDifference[i] - timeEntry, abs(values(targetPositions)[i-1])])
                timeEntry = timeDifference[i]
            else
                push!(holdPeriod, [timestamp(targetPositions)[i], timeDifference[i] - timeEntry, abs(values(positionDifference)[i])])
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
    hhiConcentration(returns::TimeArray)::Tuple{Float64, Float64, Float64}

Derives the algorithm for calculating HHI concentration.

# Arguments
- `returns::TimeArray`: TimeArray of returns series.

# Returns
- `Tuple{Float64, Float64, Float64}`: Tuple of positive returns HHI, negative returns HHI, and concentrated HHI over time.
"""
function hhiConcentration(returns::TimeArray)::Tuple{Float64, Float64, Float64}
    colName = names(returns)[1]

    returnsHHIPositive = hhi(returns[returns[colName] .>= 0])
    returnsHHINegative = hhi(returns[returns[colName] .< 0])

    returnsGrouped = groupby(transform(DataFrame(returns), :timestamp => x -> yearmonth.(x)), :timestampFunction)
    timeConcentratedHHI = hhi(combine(returnsGrouped, colName => length, renamecols=false)[:, colName])

    return (returnsHHIPositive, returnsHHINegative, timeConcentratedHHI)
end

"""
    hhi(betReturns::TimeArray)::Union{Float64, Nothing}

Calculates the Herfindahl-Hirschman Index (HHI).

# Arguments
- `betReturns::TimeArray`: Bet returns series.

# Returns
- `Union{Float64, Nothing}`: HHI value.

# Mathematical formula
The Herfindahl-Hirschman Index (HHI) is calculated as follows:

\[ \text{HHI} = \left( \sum_{i=1}^{N} w_i^2 - \frac{1}{N} \right) / \left( 1 - \frac{1}{N} \right) \]

where \(N\) is the total number of returns, and \(w_i\) is the weight of the i-th return, defined as:

\[ w_i = \frac{\text{value of i-th return}}{\text{sum of all return values}} \]
"""
function hhi(betReturns::TimeArray)::Union{Float64, Nothing}
    if length(betReturns) <= 2
        return nothing
    end

    weight = values(betReturns) ./ sum(values(betReturns))
    hhiValue = sum(weight.^2)
    hhiValue = (hhiValue - 1/length(betReturns)) / (1 - 1/length(betReturns))

    return hhiValue
end

"""
    computeDrawdownsTimeUnderWater(series::TimeArray, dollars::Bool = false)
        -> Tuple{TimeArray, TimeArray, DataFrame}

Computes series of drawdowns and the time under water associated with them.

# Arguments
- `series::TimeArray`: TimeArray of returns or dollar performance.
- `dollars::Bool = false`: Boolean indicating whether returns or dollar performance.

# Returns
- `Tuple{TimeArray, TimeArray, DataFrame}`: Tuple containing TimeArray of drawdowns, TimeArray of time under water, and drawdown analysis DataFrame.
"""
function computeDrawdownsTimeUnderWater(series::TimeArray, dollars::Bool = false)
    -> Tuple{TimeArray, TimeArray, DataFrame}
    
    seriesDF = DataFrame(TimeSeries.rename(series, names(series)[1] => :PnL))
    seriesDF[:, :HWM] = [maximum(seriesDF[1:i, :PnL]) for i in 1:length(seriesDF.PnL)]
    drawdownAnalysis = DataFrame()

    function processGroups(group::DataFrame)::Union{DataFrame, Nothing}
        if nrow(group) <= 1
            return nothing
        end
        
        result = DataFrame()
        result[:, :Start] = [group[1, :timestamp]]
        result[:, :Stop] = [group[end, :timestamp]]
        result[:, :HWM] = [group[1, :HWM]]
        result[:, :Min] = [minimum(group.PnL)]
        result[:, "Min. Time"] = [group.timestamp[group.PnL .== minimum(group.PnL)][1]]
        
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
