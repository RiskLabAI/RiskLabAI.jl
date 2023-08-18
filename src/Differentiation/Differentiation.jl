using DataFrames
using Plots
using Statistics
using HypothesisTests

"""
    calculateOhlcv(tickDataGrouped::GroupedDataFrame) -> DataFrame

Calculates grouped data's OHLCV data.

Computes the Open, High, Low, Close, Volume, ValueOfTrades, PriceMean, TickCount,
and PriceMeanLogReturn from grouped dataframes.

# Arguments
- `tickDataGrouped::GroupedDataFrame`: Grouped dataframes.

# Returns
- `DataFrame`: OHLCV data.
"""
function calculateOhlcv(
        tickDataGrouped::GroupedDataFrame
    )::DataFrame
    ohlcvDataframe = combine(
        tickDataGrouped, :price => first => :open,
        :price => maximum => :high,
        :price => minimum => :low,
        :price => last => :close,
        :size => sum => :volume,
        AsTable([:price, :size]) => x -> sum(x.price .* x.size) / sum(x.size),
        :price => mean => :priceMean,
        :price => length => :tickCount
    )
    rename!(ohlcvDataframe, :price_size_function => :valueOfTrades)
    ohlcvDataframe.priceMeanLogReturn = log.(ohlcvDataframe.priceMean) - log.(circshift(ohlcvDataframe.priceMean, 1))
    ohlcvDataframe.priceMeanLogReturn[1] = NaN
    return ohlcvDataframe
end

"""
    generateTimeBar(
        tickData::DataFrame,
        frequency::Int = 5
    ) -> DataFrame

Generates a time bar dataframe from a dataframe.

This function takes a dataframe and generates a time bar dataframe.

# Arguments
- `tickData::DataFrame`: Input dataframe.
- `frequency::Int=5`: Time frequency in minutes. Default is 5.

# Returns
- `DataFrame`: Time bar dataframe.
"""
function generateTimeBar(
        tickData::DataFrame,
        frequency::Int = 5
    )::DataFrame
    dates = tickData.dates
    datesCopy = copy(dates)
    tickData.dates = floor.(datesCopy, Dates.Minute(frequency))
    tickDataGrouped = groupby(tickData, :dates)
    ohlcvDataframe = calculateOhlcv(tickDataGrouped)
    tickData.dates = dates
    return ohlcvDataframe
end

"""
    generateWeights(
        degree::Float64,
        size::Int
    ) -> Vector{Float64}

Generates the sequence of weights for fractionally differentiated series.

This function generates the sequence of weights used to compute each value of the fractionally differentiated series.

# Arguments
- `degree::Float64`: The degree of differentiation.
- `size::Int`: Size of the weights sequence.

# Returns
- `Vector{Float64}`: Sequence of weights.
"""
function generateWeights(
        degree::Float64,
        size::Int
    )::Vector{Float64}
    ω = [1.]
    for k ∈ 2:size
        thisω = -ω[end] / (k - 1) * (degree - k + 2)
        append!(ω, thisω)
    end
    return reverse(ω)
end

"""
    plotWeights(
        degreeRange::Tuple{Float64, Float64},
        nDegrees::Int,
        numberWeights::Int
    )

Plots the weights for fractionally differentiated series.

This function plots the weights for fractionally differentiated series.

# Arguments
- `degreeRange::Tuple{Float64, Float64}`: Range of degrees.
- `nDegrees::Int`: Number of degrees.
- `numberWeights::Int`: Number of weights.
"""
function plotWeights(
        degreeRange::Tuple{Float64, Float64},
        nDegrees::Int,
        numberWeights::Int
    )
    ω = DataFrame(index = collect(numberWeights - 1:-1:0))
    for degree ∈ range(degreeRange[1], degreeRange[2], length = nDegrees)
        degree = round(degree; digits = 2)
        thisω = generateWeights(degree, numberWeights)
        thisω = DataFrame(index = collect(numberWeights - 1:-1:0), ω = thisω)
        ω = outerjoin(ω, thisω, on = :index, makeunique = true)
    end
    rename!(ω, names(ω)[2:end] .=> string.(range(degreeRange[1], degreeRange[2], length = nDegrees)))
    plot(ω[:, 1], Matrix(ω[:, 2:end]), label = reshape(names(ω)[2:end], (1, nDegrees)), background = :transparent)
end

"""
    fractionalDifferentiation(
        series::DataFrame,
        degree::Float64,
        threshold::Float64=0.01
    )::DataFrame

Applies the standard fractionally differentiated method to a series.

# Parameters
- `series`: DataFrame, input series with a column named `:dates`.
- `degree`: Float64, the degree of differentiation.
- `threshold`: Float64, threshold value, default is 0.01.

# Returns
- DataFrame: Fractionally differentiated series.

"""
function fractionalDifferentiation(
        series::DataFrame,
        degree::Float64,
        threshold::Float64=0.01
    )::DataFrame

    weights = computeWeights(degree, nrow(series))
    weightsNormalized = cumsum(abs.(weights), dims=1)
    weightsNormalized ./= weightsNormalized[end]
    drop = length(filter(x -> x > threshold, weightsNormalized))
    dataframe = DataFrame(index=filter(!ismissing, series[:, :dates])[
                range(drop + 1, stop=nrow(filter(!ismissing, series[:, :dates])), step=1)])
    for name in names(series)[2:end]
        seriesFiltered = filter(!ismissing, series[:, [:dates, name]])
        thisRange = range(drop + 1, stop=nrow(seriesFiltered), step=1)
        dataframeFiltered = DataFrame(index=seriesFiltered[thisRange, :dates], value=zeros(length(thisRange)))
        data = []
        for i in thisRange
            date = seriesFiltered[i, :dates]
            price = series[series.dates .== date, name][1]
            if !isfinite(price)
                continue
            end
            try
                append!(data, dot(
                    weights[end-i+1:end],
                    filter(row -> row[:dates] in seriesFiltered[1, :dates]:Day(1):date, seriesFiltered)[:, name]))
            catch
                continue
            end
        end
        dataframeFiltered.value = data
        dataframe = innerjoin(dataframe, dataframeFiltered, on=:index)
    end
    return dataframe
end

"""
    computeWeights(
        degree::Float64,
        threshold::Float64
    )::Vector{Float64}

Generates weights for the fixed-width window method.

# Parameters
- `degree`: Float64, the degree of differentiation.
- `threshold`: Float64, threshold value.

# Returns
- Vector{Float64}: Sequence of weights.

"""
function computeWeights(
        degree::Float64,
        threshold::Float64
    )::Vector{Float64}

    ω = [1.]
    k = 1
    while abs(ω[end]) >= threshold
        thisω = -ω[end] / k * (degree - k + 1)
        append!(ω, thisω)
        k += 1
    end
    return reverse(ω)[2:end]
end

"""
    fractionalDifferentiationFixed(
        series::DataFrame,
        degree::Float64,
        threshold::Float64=1e-5
    )::DataFrame

Applies the fixed-width window fractionally differentiated method to a series.

# Parameters
- `series`: DataFrame, input series with a column named `:dates`.
- `degree`: Float64, the degree of differentiation.
- `threshold`: Float64, threshold value, default is 1e-5.

# Returns
- DataFrame: Fractionally differentiated series.

"""
function fractionalDifferentiationFixed(
        series::DataFrame,
        degree::Float64,
        threshold::Float64=1e-5
    )::DataFrame
    weights = computeWeights(degree, threshold)
    width = length(weights) - 1

    dataframe = DataFrame(index=series[range(width + 1, stop=nrow(series), step=1), :dates])
    for name in names(series)[2:end]
        seriesFiltered = filter(!ismissing, series[:, [:dates, name]])
        thisRange = range(width + 1, stop=nrow(seriesFiltered), step=1)
        dataframeFiltered = DataFrame(index=seriesFiltered[thisRange, :dates], value=zeros(length(thisRange)))
        data = []
        for i in thisRange
            day1 = seriesFiltered[i - width, :dates]
            day2 = seriesFiltered[i, :dates]
            if !isfinite(series[series.dates .== day2, name][1])
                continue
            end
            append!(data, dot(
                    weights,
                    filter(row -> row[:dates] in day1:Day(1):day2, seriesFiltered)[:, name]))
        end
        dataframeFiltered.value = data
        dataframe = innerjoin(dataframe, dataframeFiltered, on=:index)
    end
    return dataframe
end

"""
    findMinimumDegree(
        input::DataFrame
    )::DataFrame

Finds the minimum degree value that passes the Augmented Dickey-Fuller (ADF) test.

# Parameters
- `input`: DataFrame, input data with columns `:dates` and `:close`.

# Returns
- DataFrame: ADF test results with columns for degree, ADF statistic, p-value, lags, number of observations, 95% confidence level, and correlation.

"""
function findMinimumDegree(input::DataFrame)::DataFrame
    out = DataFrame(d=[], adfStat=[], pVal=[], lags=[], nObs=[], nintyfiveperconf=[], corr=[])
    for d in range(0, 1, length=11)
        dataframe = DataFrame(dates=Date.(input[:, :dates]), pricelog=log.(input[:, :close]))
        differentiated = fractionalDifferentiationFixed(dataframe, d, .01)
        corr = cor(filter(row -> row[:dates] in differentiated[:, 1],dataframe)[:, :pricelog],
                   differentiated[:, 2])
        adfTest = ADFTest(Float64.(differentiated[:, 2]), :constant, 1)
        push!(out, [d, adfTest.stat, pvalue(adfTest), adfTest.lag,
                    adfTest.n, adfTest.cv[2], corr])
    end
    return out
end
