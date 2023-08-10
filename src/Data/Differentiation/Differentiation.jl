using GLM
using DataFrames
using TimeSeries

"""
Function: Combines grouped dataframe to create a new one with information about prices and volume.

Combines grouped dataframe to create a new one with information about prices and volume.

:param tickDataGrouped: Grouped dataframe containing tick data.
:return: ohlcvDataframe::DataFrame: Combined dataframe with OHLCV information.
"""
function ohlcv(tickDataGrouped)
    ohlcvDataframe = combine(tickDataGrouped,
        :price => first => :open,
        :price => maximum => :high,
        :price => minimum => :low,
        :price => last => :close,
        :size => sum => :volume,
        AsTable([:price, :size]) => x -> sum(x.price .* x.size) / sum(x.size),
        :price => mean => :priceMean,
        :price => length => :tickCount
    )
    DataFrames.rename!(ohlcvDataframe, :priceSizeFunction => :valueOfTrades)
    ohlcvDataframe.priceMeanLogReturn = log.(ohlcvDataframe.priceMean) - log.(circshift(ohlcvDataframe.priceMean, 1))
    ohlcvDataframe.priceMeanLogReturn[1] = NaN
    return ohlcvDataframe
end

"""
Function: Takes a dataframe and generates a time bar dataframe.

Takes a dataframe and generates a time bar dataframe with specified frequency.

:param tickData: Input dataframe containing tick data.
:param frequency: Frequency for time bars (default = 5).
:return: ohlcvDataframe::DataFrame: Time bar dataframe with OHLCV information.
"""
function timeBar(tickData, frequency = 5)
    dates = tickData.dates
    datesCopy = copy(dates)
    tickData.dates = floor.(datesCopy, Dates.Minute(frequency))
    tickDataGrouped = DataFrames.groupby(tickData, :dates)
    ohlcvDataframe = ohlcv(tickDataGrouped)
    tickData.dates = dates
    return ohlcvDataframe
end

"""
Function: The sequence of weights used to compute each value of the fractionally differentiated series.

Generates the sequence of weights used to compute each value of the fractionally differentiated series.

:param degree: Degree of differentiation.
:param size: Size of the weights sequence.
:return: ω::Vector{Float64}: Sequence of weights.
"""
function weighting(degree, size)
    ω = [1.]
    for k in 2:size
        thisω = -ω[end] / (k - 1) * (degree - k + 2)
        push!(ω, thisω)
    end
    return reverse(ω)
end

"""
Function: Plot weights.

Plots the weights used for fractionally differentiated series.

:param degreeRange: Range of degree values.
:param numberDegrees: Number of degree values to consider.
:param numberWeights: Number of weights to plot.
"""
function plotWeights(degreeRange, numberDegrees, numberWeights)
    ω = DataFrames.DataFrame(index = collect(numberWeights - 1:-1:0))
    for degree in range(degreeRange[1], degreeRange[2], length = numberDegrees)
        degree = round(degree; digits = 2)
        thisω = weighting(degree, numberWeights)
        thisω = DataFrames.DataFrame(index = collect(numberWeights - 1:-1:0), ω = thisω)
        ω = outerjoin(ω, thisω, on = :index, makeunique = true)
    end
    DataFrames.rename!(ω, names(ω)[2:end] .=> string.(range(degreeRange[1], degreeRange[2], length = numberDegrees)))
    plot(ω[:, 1], Matrix(ω[:, 2:end]), label = reshape(names(ω)[2:end], (1, numberDegrees)), background = :transparent)
end

"""
Function: Standard fractionally differentiated.

Performs standard fractionally differentiation on the given series.

:param series: Input time series data.
:param degree: Degree of differentiation.
:param threshold: Threshold for drop in weights.
:return: dataframe::DataFrame: Fractionally differentiated series.
"""
function fracDiff(series, degree, threshold = 0.01)
    weights = weighting(degree, size(series)[1])
    weightsNormalized = cumsum(broadcast(abs, weights), dims = 1)
    weightsNormalized /= weightsNormalized[end]
    drop = length(filter(x -> x > threshold, weightsNormalized))
    dataframe = DataFrames.DataFrame(index = filter(!ismissing, series[:, [:dates]])[
        range(drop + 1, stop = size(filter(!ismissing, series[:, [:dates]]))[1], step = 1), 1])
    for name in Symbol.(names(series))[2:end]
        seriesFiltered = filter(!ismissing, series[:, [:dates, name]])
        thisRange = range(drop + 1, stop = size(seriesFiltered)[1], step = 1)
        dataframeFiltered = DataFrames.DataFrame(index = seriesFiltered[thisRange, 1], value = repeat([0.], length(thisRange)))
        data = []
        for i in range(drop + 1, stop = size(seriesFiltered)[1], step = 1)
            date = seriesFiltered[i, 1]
            price = series[series.dates .== date, name][1]
            if !isfinite(price)
                continue
            end
            try
                append!(data, Statistics.dot(
                    weights[length(weights) - i + 1:end, :],
                    filter(row -> row[:dates] in collect(seriesFiltered[1, 1]:Day(1):date), seriesFiltered)[:, name]))
            catch
                continue
            end
        end
        dataframeFiltered.value = data
        dataframe = DataFrames.innerjoin(dataframe, dataframeFiltered, on = :index)
    end
    return dataframe
end

"""
Function: Weights for fixed-width window method.

Calculates the weights for the fixed-width window method of fractionally differentiation.

:param degree: Degree of differentiation.
:param threshold: Threshold for drop in weights.
:return: ω::Vector{Float64}: Sequence of weights.
"""
function weightingFfd(degree, threshold)
    ω = [1.]
    k = 1
    while abs(ω[end]) >= threshold
        thisω = -ω[end] / k * (degree - k + 1)
        push!(ω, thisω)
        k += 1
    end
    return reverse(ω)[2:end]
end

"""
Function: Fixed-width window fractionally differentiated method.

Applies the fixed-width window method of fractionally differentiation to the given series.

:param series: Input time series data.
:param degree: Degree of differentiation.
:param threshold: Threshold for drop in weights.
:return: dataframe::DataFrame: Fractionally differentiated series.
"""
function fracDiffFixed(series, degree, threshold = 1e-5)
    weights = weightingFfd(degree, threshold)
    width = length(weights) - 1
    dataframe = DataFrames.DataFrame(index = series[
        range(width + 1, stop = size(series)[1], step = 1), 1])
    for name in Symbol.(names(series))[2:end]
        seriesFiltered = filter(!ismissing, series[:, [:dates, name]])
        thisRange = range(width + 1, stop = size(seriesFiltered)[1], step = 1)
        dataframeFiltered = DataFrames.DataFrame(index = seriesFiltered[thisRange, 1])
        data = []
        for i in range(width + 1, stop = size(seriesFiltered)[1], step = 1)
            day1 = seriesFiltered[i - width, 1]
            day2 = seriesFiltered[i, 1]
            if !isfinite(series[series.dates .== day2, name][1])
                continue
            end
            append!(data, Statistics.dot(
                    weights,
                    filter(row -> row[:dates] in collect(day1:Day(1):day2), seriesFiltered)[:, name]))
        end
        dataframeFiltered.value = data
        dataframe = DataFrames.innerjoin(dataframe, dataframeFiltered, on = :index)
    end
    return dataframe
end

"""
Function: Find the minimum degree value that passes the ADF test.

Finds the minimum degree value that passes the ADF test for fractionally differentiated series.

:param input: Input time series data.
:return: out::DataFrame: Results dataframe with ADF statistics.
"""
function minFFD(input)
    out = DataFrames.DataFrame(d = [], adfStat = [], pVal = [], lags = [], nObs = [], ninetyFivePerConf = [], corr = [])
    for d in range(0, 1, length = 11)
        dataframe = DataFrames.DataFrame(dates = Date.(input[:, 1]), priceLog = log.(input[:, :close]))
        differentiated = fracDiffFixed(dataframe, d, .01)
        corr = cor(filter(row -> row[:dates] in differentiated[:, 1], dataframe)[:, :priceLog], differentiated[:, 2])
        differentiated = HypothesisTests.ADFTest(Float64.(differentiated[:, 2]), :constant, 1)
        push!(out, [d, differentiated.stat, pvalue(differentiated), differentiated.lag, differentiated.n, differentiated.cv[2], corr])
    end
    return out
end
