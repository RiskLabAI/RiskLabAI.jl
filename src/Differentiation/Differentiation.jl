"""
    function: Takes grouped dataframe, combining and creating the new one with info. about prices and volume.
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: n/a
"""
function ohlcv(tickDataGrouped) # grouped dataframes
    ohlcvDataframe = combine(tickDataGrouped, :price => first => :Open, 
                             :price => maximum => :High,
                             :price => minimum => :Low,
                             :price => last => :Close,
                             :size => sum => :Volume,
                             AsTable([:price, :size]) => x -> sum(x.price .* x.size) / sum(x.size), 
                             :price => mean => :PriceMean,
                             :price => length => :TickCount)
    DataFrames.rename!(ohlcvDataframe, :price_size_function => :ValueOfTrades)
    ohlcvDataframe.PriceMeanLogReturn = log.(ohlcvDataframe.PriceMean) - log.(circshift(ohlcvDataframe.PriceMean, 1))
    ohlcvDataframe.PriceMeanLogReturn[1] = NaN
    return ohlcvDataframe
end

"""
    function: Takes dataframe and generates time bar dataframe
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: n/a
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
    function: The sequence of weights used to compute each value of the fractionally differentiated series.
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 79
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
    function: plot weights
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 79
"""
function plotWeights(degreeRange, nDegrees, numberWeights)
    ω = DataFrames.DataFrame(index = collect(numberWeights - 1:-1:0))
    for degree ∈ range(degreeRange[1], degreeRange[2], length = nDegrees)
        degree = round(degree; digits = 2)
        thisω = weighting(degree, numberWeights)
        thisω = DataFrames.DataFrame(index = collect(numberWeights - 1:-1:0), ω = thisω)
        ω = outerjoin(ω, thisω, on = :index, makeunique = true)
    end
    DataFrames.rename!(ω, names(ω)[2:end] .=> string.(range(degreeRange[1], degreeRange[2], length = numberDegrees)))
    plot(ω[:, 1], Matrix(ω[:, 2:end]), label = reshape(names(ω)[2:end], (1, nDegrees)), background = :transparent)
end

"""
    function: standard fractionally differentiated
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 82
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
    function: weights for fixed-width window method
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 83
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
    function: Fixed-width window fractionally differentiated method
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 83
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
    function: Find the minimum degree value that passes the ADF test
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 85
"""
function minFFD(input)
    out = DataFrames.DataFrame(d = [], adfStat = [], pVal = [], lags = [], nObs = [], 
                               nintyfiveperconf = [], corr = [])
    for d in range(0, 1, length = 11)
        dataframe = DataFrames.DataFrame(dates = Date.(input[:, 1]), 
                                         pricelog = log.(input[:, :Close]))
        differentiated = fracDiffFixed(dataframe, d, .01)
        corr = cor(filter(row -> row[:dates] in differentiated[:, 1],dataframe)[:, :pricelog], 
                   differentiated[:, 2])
        differentiated = HypothesisTests.ADFTest(Float64.(differentiated[:, 2]), :constant, 1)
        push!(out, [d, differentiated.stat, pvalue(differentiated), differentiated.lag, 
                    differentiated.n, differentiated.cv[2], corr])
    end
    return out
end
