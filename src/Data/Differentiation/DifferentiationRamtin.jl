

"""
function: Takes grouped dataframe, combining and creating the new one with info. about prices and volume.
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
"""
function ohlcv(tickDataGrouped) #grouped dataframes
    # combining groups
    ohlcvDataframe = combine(tickDataGrouped, :price => first => :Open, # find open price
               :price => maximum => :High, # find highest price
               :price => minimum => :Low,  # find lowest price
               :price => last => :Close, # find close price
               :size => sum => :Volume, # find volume traded
               AsTable([:price, :size]) =>x -> sum(x.price.*x.size)/sum(x.size), # find value of trades
               :price => mean => :PriceMean, # mean of price
               :price => length => :TickCount) # number of ticks
    DataFrames.rename!(ohlcvDataframe, :price_size_function => :ValueOfTrades) # rename to value of trades
    ohlcvDataframe.PriceMeanLogReturn = log.(ohlcvDataframe.PriceMean) - log.(circshift(ohlcvDataframe.PriceMean, 1)) # calculate log return
    ohlcvDataframe.PriceMeanLogReturn[1] = NaN # set first return to NaN
    return ohlcvDataframe
end

"""
function: Takes dataframe and generating time bar dataframe
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: n/a
"""
function timeBar(tickData, # dataframe of tick data
                 frequency = 5) # frequency for rounding date time
    dates = tickData.dates #date time column
    datesCopy = copy(dates)
    tickData.dates = floor.(datesCopy, Dates.Minute(frequency)) # round down date times with frequency freq
    tickDataGrouped = DataFrames.groupby(tickData, :dates) # group date times based on rounding
    ohlcvDataframe = ohlcv(tickDataGrouped) # create a dataframe based on time bars with frequency freq
    tickData.dates = dates # recovery date times
    return ohlcvDataframe
end

"""
function: The sequence of weights used to compute each value of the fractionally differentiated series.
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: 79
"""
function weighting(degree, # degree of binomial series
                   size) # number of weights
    ω = [1.] # array of weights
    for k ∈ 2:size
        thisω = -ω[end]/(k - 1)*(degree - k + 2) # calculate each weight
        append!(ω, thisω) # append weight into array
    end
    return reverse(ω)
end

"""
function: plot weights
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: 79
"""
function plotWeights(degreeRange, # range for degree
                     numberDegrees, # number of degrees
                     numberWeights) # number of weights
    ω = DataFrames.DataFrame(index = collect(numberWeights - 1:-1:0)) # dataframe of weights
    for degree ∈ range(degreeRange[1], degreeRange[2], length = numberDegrees)
        degree = round(degree; digits = 2) # round degree with digits = 2
        thisω = weighting(degree, numberWeights) # calculate weights for each degree
        thisω = DataFrames.DataFrame(index = collect(numberWeights - 1:-1:0), ω = thisω) # dataframe of weights for each degree
        ω = outerjoin(ω, thisω, on = :index, makeunique = true) # append into ω
    end
    # rename columns
    DataFrames.rename!(ω, names(ω)[2:end] .=> string.(range(degreeRange[1], degreeRange[2], length = numberDegrees)))
    plot(ω[:,1], Matrix(ω[:,2:end]), label = reshape(names(ω)[2:end],
            (1, numberDegrees))) # plot weights
end

"""
function: standard fractionally differentiated
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: 82
"""
function fracDiff(series, # dataframe of dates and prices
                  degree, # degree of binomial series
                  threshold = 0.01) # threshold for weight-loss
    weights = weighting(degree, size(series)[1]) # calculate weights
    weightsNormalized = cumsum(broadcast(abs, weights), dims = 1) # cumulate weights
    weightsNormalized/=weightsNormalized[end] # normalize weight
    drop = length(filter(x -> x > threshold, weightsNormalized)) # number of droping observations
    # dataframe of output
    dataframe = DataFrames.DataFrame(index = filter(!ismissing, series[:, [:dates]])[
                range(drop +1, stop = size(filter(!ismissing, series[:, [:dates]]))[1], step = 1), 1])
    for name ∈ Symbol.(names(series))[2:end] # column names of series
        seriesFiltered = filter(!ismissing, series[:, [:dates, name]]) # filter for missing data
        thisRange = range(drop + 1, stop = size(seriesFiltered)[1], step = 1) # range of dates
        # dataframe for each name
        dataframeFiltered = DataFrames.DataFrame(index = seriesFiltered[thisRange, 1], value = repeat([0.], length(thisRange)))
        data = [] # output for each name
        for i ∈ range(drop + 1, stop = size(seriesFiltered)[1], step = 1)
            date = seriesFiltered[i, 1] # date
            price = series[series.dates .== date, name][1] # price for that date
            # exclude NAs
            if !isfinite(price) # check for being finite
                continue
            end
            try
                # calculate values and append into data
                append!(data, LinearAlgebra.dot(
                    weights[length(weights) - i + 1:end, :], 
                    filter(row -> row[:dates] in collect(seriesFiltered[1, 1]:Day(1):date), seriesFiltered)[:, name]))
            catch
                continue
            end
        end
        dataframeFiltered.value = data # replace data into dataframeFiltered
        dataframe = DataFrames.innerjoin(dataframe, dataframeFiltered, on = :index) # join dataframes
    end
    return dataframe
end

"""
function: weights for fixed-width window method
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: 83
"""
function weightingFFD(degree, # degree of binomial series
                      threshold) # threshold
    ω = [1.]  # array of weights
    k = 1 # initial value of k
    while abs(ω[end]) >= threshold 
        thisω = -ω[end]/k*(degree - k + 1) # calculate each weight
        append!(ω, thisω) # append into array
        k += 1 # update k
    end
    return reverse(ω)[2:end]
end

"""
function: Fixed-width window fractionally differentiated method
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: 83
"""
function fracDiffFixed(series, # dataframe of dates and prices
                       degree, # degree of binomial series
                       threshold = 1e-5) # threshold
    weights = weightingFFD(degree, threshold) # compute weights for the longest series
    width = length(weights) - 1 # length of weights
    
    dataframe = DataFrames.DataFrame(index = series[
                range(width + 1, stop = size(series)[1], step = 1), 1])
                # dataframe of output
    for name ∈ Symbol.(names(series))[2:end]
        seriesFiltered = filter(!ismissing, series[:, [:dates, name]]) # filter for missing data
        thisRange = range(width + 1, stop = size(seriesFiltered)[1], step = 1) # range of dates
        dataframeFiltered = DataFrames.DataFrame(index = seriesFiltered[thisRange, 1]) # dataframe for each name
        data = [] # output for each name
        for i ∈ range(width + 1, stop = size(seriesFiltered)[1], step = 1)
            day1 = seriesFiltered[i - width, 1] # first day
            day2 = seriesFiltered[i, 1] # last day
            if !isfinite(series[series.dates .== day2, name][1]) # check for being finite
                continue
            end
            # calculate value and append into data
            append!(data, LinearAlgebra.dot(
                    weights, 
                    filter(row -> row[:dates] in collect(day1:Day(1):day2), seriesFiltered)[:, name]))
        end
        dataframeFiltered.value = data # replace data into dataframeFiltered
        dataframe = DataFrames.innerjoin(dataframe, dataframeFiltered, on = :index) # join dataframes
    end
    return dataframe
end

"""
function: Find the minimum degree value that passes the ADF test
reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
methodology: 85
"""
function minFFD(input) # input dataframe
    # output dataframe
    out = DataFrames.DataFrame(d=[], adfStat=[], pVal=[], lags=[], nObs=[], nintyfiveperconf=[], corr=[]) # dataframe of output
    for d in range(0, 1, length = 11)
        dataframe = DataFrames.DataFrame(dates = Date.(input[:, 1]), pricelog = log.(input[:, :Close])) # dataframe of price and dates
        differentiated = fracDiffFixed(dataframe, d, .01) # call fixed-width frac diff method
        corr = cor(filter(row -> row[:dates] in differentiated[:, 1],dataframe)[:, :pricelog], differentiated[:, 2]) # correlation 
        differentiated = HypothesisTests.ADFTest(Float64.(differentiated[:, 2]),:constant, 1) # ADF test
        push!(out, [d,differentiated.stat, pvalue(differentiated), differentiated.lag, differentiated.n, differentiated.cv[2], corr]) # push new observation
    end
    return out
end
