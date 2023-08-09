"""
Calculates grouped data's OHLCV data.

This function calculates the Open, High, Low, Close, Volume, ValueOfTrades, PriceMean, TickCount, 
and PriceMeanLogReturn from grouped dataframes.

Parameters:
- tickDataGrouped: Grouped dataframes.

Returns:
- DataFrame: OHLCV data.
"""
function ohlcv(tickDataGrouped) 
    ohlcvDataframe = combine(tickDataGrouped, :price => first => :open, 
                             :price => maximum => :high,
                             :price => minimum => :low,
                             :price => last => :close,
                             :size => sum => :volume,
                             AsTable([:price, :size]) => x -> sum(x.price .* x.size) / sum(x.size), 
                             :price => mean => :priceMean,
                             :price => length => :tickCount)
    DataFrames.rename!(ohlcvDataframe, :price_size_function => :valueOfTrades)
    ohlcvDataframe.priceMeanLogReturn = log.(ohlcvDataframe.priceMean) - log.(circshift(ohlcvDataframe.priceMean, 1))
    ohlcvDataframe.priceMeanLogReturn[1] = NaN
    return ohlcvDataframe
end

"""
Generates a time bar dataframe from a dataframe.

This function takes a dataframe and generates a time bar dataframe.

Parameters:
- tickData: Input dataframe.
- frequency: Time frequency in minutes. Default is 5.

Returns:
- DataFrame: Time bar dataframe.
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
Generates the sequence of weights for fractionally differentiated series.

This function generates the sequence of weights used to compute each value of the fractionally differentiated series.

Parameters:
- degree: The degree of differentiation.
- size: Size of the weights sequence.

Returns:
- Vector: Sequence of weights.
"""
function weighting(degree, size)
    ω = [1.]
    for k ∈ 2:size
        thisω = -ω[end] / (k - 1) * (degree - k + 2)
        append!(ω, thisω)
    end
    return reverse(ω)
end

"""
Plots the weights for fractionally differentiated series.

This function plots the weights for fractionally differentiated series.

Parameters:
- degreeRange: Range of degrees.
- nDegrees: Number of degrees.
- numberWeights: Number of weights.
"""
function plotWeights(degreeRange, nDegrees, numberWeights)
    ω = DataFrames.DataFrame(index = collect(numberWeights - 1:-1:0))
    for degree ∈ range(degreeRange[1], degreeRange[2], length = nDegrees)
        degree = round(degree; digits = 2)
        thisω = weighting(degree, numberWeights)
        thisω = DataFrames.DataFrame(index = collect(numberWeights - 1:-1:0), ω = thisω)
        ω = outerjoin(ω, thisω, on = :index, makeunique = true)
    end
    DataFrames.rename!(ω, names(ω)[2:end] .=> string.(range(degreeRange[1], degreeRange[2], length = nDegrees)))
    plot(ω[:, 1], Matrix(ω[:, 2:end]), label = reshape(names(ω)[2:end], (1, nDegrees)), background = :transparent)
end

"""
Applies standard fractionally differentiated method.

This function applies the standard fractionally differentiated method to a series.

Parameters:
- series: Input series.
- degree: The degree of differentiation.
- threshold: Threshold value. Default is 0.01.

Returns:
- DataFrame: Fractionally differentiated series.
"""
function fracDiff(series, degree, threshold = 0.01)
    weights = weighting(degree, size(series)[1])
    weightsNormalized = cumsum(broadcast(abs, weights), dims = 1)
    weightsNormalized /= weightsNormalized[end]
    drop = length(filter(x -> x > threshold, weightsNormalized))
    dataframe = DataFrames.DataFrame(index = filter(!ismissing, series[:, [:dates]])[
                range(drop + 1, stop = size(filter(!ismissing, series[:, [:dates]]))[1], step = 1), 1])
    for name ∈ Symbol.(names(series))[2:end]
        seriesFiltered = filter(!ismissing, series[:, [:dates, name]])
        thisRange = range(drop + 1, stop = size(seriesFiltered)[1], step = 1)
        dataframeFiltered = DataFrames.DataFrame(index = seriesFiltered[thisRange, 1], value = repeat([0.], length(thisRange)))
        data = []
        for i ∈ range(drop + 1, stop = size(seriesFiltered)[1], step = 1)
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
Generates weights for fixed-width window method.

This function generates weights for the fixed-width window method.

Parameters:
- degree: The degree of differentiation.
- threshold: Threshold value.

Returns:
- Vector: Sequence of weights.
"""
function weightingFFD(degree, threshold)
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
Applies the fixed-width window fractionally differentiated method.

This function applies the fixed-width window fractionally differentiated method to a series.

Parameters:
- series: Input series.
- degree: The degree of differentiation.
- threshold: Threshold value. Default is 1e-5.

Returns:
- DataFrame: Fractionally differentiated series.
"""
function fracDiffFixed(series, degree, threshold = 1e-5)
    weights = weightingFFD(degree, threshold)
    width = length(weights) - 1
    
    dataframe = DataFrames.DataFrame(index = series[
                range(width + 1, stop = size(series)[1], step = 1), 1])
                
    for name ∈ Symbol.(names(series))[2:end]
        seriesFiltered = filter(!ismissing, series[:, [:dates, name]])
        thisRange = range(width + 1, stop = size(seriesFiltered)[1], step = 1)
        dataframeFiltered = DataFrames.DataFrame(index = seriesFiltered[thisRange, 1])
        data = []
        for i ∈ range(width + 1, stop = size(seriesFiltered)[1], step = 1)
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
Finds the minimum degree value that passes the ADF test.

This function finds the minimum degree value that passes the Augmented Dickey-Fuller (ADF) test.

Parameters:
- input: Input data.

Returns:
- DataFrame: ADF test results.
"""
function minFFD(input)
    out = DataFrames.DataFrame(d = [], adfStat = [], pVal = [], lags = [], nObs = [], 
                               nintyfiveperconf = [], corr = [])
    for d in range(0, 1, length = 11)
        dataframe = DataFrames.DataFrame(dates = Date.(input[:, 1]), 
                                         pricelog = log.(input[:, :close]))
        differentiated = fracDiffFixed(dataframe, d, .01)
        corr = cor(filter(row -> row[:dates] in differentiated[:, 1],dataframe)[:, :pricelog], 
                   differentiated[:, 2])
        differentiated = HypothesisTests.ADFTest(Float64.(differentiated[:, 2]), :constant, 1)
        push!(out, [d, differentiated.stat, pvalue(differentiated), differentiated.lag, 
                    differentiated.n, differentiated.cv[2], corr])
    end
    return out
end
