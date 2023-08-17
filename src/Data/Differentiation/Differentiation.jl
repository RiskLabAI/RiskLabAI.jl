using DataFrames, Dates, Statistics, HypothesisTests, Plots

"""
    ohlcv(tickDataGrouped::GroupedDataFrame)

Combine grouped tick data to create a new dataframe with Open, High, Low, Close, Volume (OHLCV) information.

# Arguments
- `tickDataGrouped::GroupedDataFrame`: Grouped dataframe containing tick data.

# Returns
- `ohlcvDataframe::DataFrame`: Combined dataframe with OHLCV information.

# Mathematical formulae
- Open price: First price within each group.
- High price: Maximum price within each group.
- Low price: Minimum price within each group.
- Close price: Last price within each group.
- Volume: Sum of size within each group.
- Value of trades: Sum(price * size) / Sum(size).
- Mean price log return: Log(priceMean) - Log(circshift(priceMean, 1)).
"""
function ohlcv(tickDataGrouped::GroupedDataFrame)
    ohlcvDataframe = combine(
        tickDataGrouped,
        :price => first => :open,
        :price => maximum => :high,
        :price => minimum => :low,
        :price => last => :close,
        :size => sum => :volume,
        AsTable([:price, :size]) => x -> sum(x.price .* x.size) / sum(x.size) => :valueOfTrades,
        :price => mean => :priceMean,
        :price => length => :tickCount
    )
    ohlcvDataframe.priceMeanLogReturn = log.(ohlcvDataframe.priceMean) - log.(circshift(ohlcvDataframe.priceMean, 1))
    ohlcvDataframe.priceMeanLogReturn[1] = NaN
    return ohlcvDataframe
end

"""
    timeBar(tickData::DataFrame, frequency::Int=5)

Generates a time bar dataframe with specified frequency.

# Arguments
- `tickData::DataFrame`: Input dataframe containing tick data.
- `frequency::Int=5`: Frequency for time bars (default = 5).

# Returns
- `ohlcvDataframe::DataFrame`: Time bar dataframe with OHLCV information.
"""
function timeBar(tickData::DataFrame, frequency::Int=5)
    datesCopy = copy(tickData.dates)
    tickData.dates = floor.(datesCopy, Dates.Minute(frequency))
    tickDataGrouped = groupby(tickData, :dates)
    ohlcvDataframe = ohlcv(tickDataGrouped)
    tickData.dates = datesCopy
    return ohlcvDataframe
end

"""
    weighting(degree::Float64, size::Int)

Generates the sequence of weights used to compute each value of the fractionally differentiated series.

# Arguments
- `degree::Float64`: Degree of differentiation.
- `size::Int`: Size of the weights sequence.

# Returns
- `ω::Vector{Float64}`: Sequence of weights.
"""
function weighting(degree::Float64, size::Int)
    ω = [1.0]
    for k in 2:size
        thisω = -ω[end] / (k - 1) * (degree - k + 2)
        push!(ω, thisω)
    end
    return reverse(ω)
end

"""
    plotWeights(degreeRange::Tuple{Float64, Float64}, numberDegrees::Int, numberWeights::Int)

Plots the weights used for fractionally differentiated series.

# Arguments
- `degreeRange::Tuple{Float64, Float64}`: Range of degree values.
- `numberDegrees::Int`: Number of degree values to consider.
- `numberWeights::Int`: Number of weights to plot.
"""
function plotWeights(degreeRange::Tuple{Float64, Float64}, numberDegrees::Int, numberWeights::Int)
    ω = DataFrame(index = collect(numberWeights - 1:-1:0))
    for degree in range(degreeRange[1], degreeRange[2], length = numberDegrees)
        degree = round(degree; digits = 2)
        thisω = weighting(degree, numberWeights)
        thisω = DataFrame(index = collect(numberWeights - 1:-1:0), ω = thisω)
        ω = outerjoin(ω, thisω, on = :index, makeunique = true)
    end
    rename!(ω, names(ω)[2:end] .=> string.(range(degreeRange[1], degreeRange[2], length = numberDegrees)))
    plot(ω[:, 1], Matrix(ω[:, 2:end]), label = reshape(names(ω)[2:end], (1, numberDegrees)), background = :transparent)
end

using Statistics

"""
    weightingFfd(degree::Float64, threshold::Float64)

Calculates the weights for the fixed-width window method of fractionally differentiation.

# Arguments
- `degree::Float64`: Degree of differentiation.
- `threshold::Float64`: Threshold for drop in weights.

# Returns
- `Vector{Float64}`: Sequence of weights.

# Mathematical Formula
The weights for the fixed-width window method are given by the formula:

.. math::
    ω_{i} = -ω_{i-1} / k * (degree - k + 1)

where `ω_{i}` is the weight at index `i`, `k` is the index, and `degree` is the differentiation degree.
"""
function weightingFfd(degree::Float64, threshold::Float64)
    ω = Float64[1.0]
    k = 1
    while abs(ω[end]) >= threshold
        thisω = -ω[end] / k * (degree - k + 1)
        push!(ω, thisω)
        k += 1
    end
    return reverse(ω)[2:end]
end

"""
    fractionalDifferentiation(series::DataFrame, degree::Float64, threshold::Float64 = 0.01)

Performs standard fractional differentiation on the given series.

# Arguments
- `series::DataFrame`: Input time series data.
- `degree::Float64`: Degree of differentiation.
- `threshold::Float64`: Threshold for drop in weights. Defaults to 0.01.

# Returns
- `DataFrame`: Fractionally differentiated series.
"""
function fractionalDifferentiation(series::DataFrame, degree::Float64, threshold::Float64 = 0.01)
    weights = weightingFfd(degree, size(series, 1))
    weightsNormalized = cumsum(abs.(weights))
    weightsNormalized ./= weightsNormalized[end]
    drop = length(filter(x -> x > threshold, weightsNormalized))
    dataframe = DataFrame(index = filter(!ismissing, series.dates)[drop + 1:end])
    
    for name in names(series)[2:end]
        seriesFiltered = filter(!ismissing, series[:, [:dates, name]])
        thisRange = drop + 1:size(seriesFiltered, 1)
        dataframeFiltered = DataFrame(index = seriesFiltered.dates[thisRange], value = zeros(length(thisRange)))
        data = []
        
        for i in thisRange
            date = seriesFiltered.dates[i]
            price = series[series.dates .== date, name][1]
            
            if !isfinite(price)
                continue
            end
            
            try
                append!(data, dot(weights[end-i+1:end], 
                        filter(row -> row[:dates] in Date(seriesFiltered.dates[1]):Day(1):date, seriesFiltered)[!, name]))
            catch
                continue
            end
        end
        
        dataframeFiltered.value .= data
        dataframe = innerjoin(dataframe, dataframeFiltered, on = :index)
    end
    
    return dataframe
end

"""
    fractionalDifferentiationFixed(series::DataFrame, degree::Float64, threshold::Float64 = 1e-5)

Applies the fixed-width window method of fractional differentiation to the given series.

# Arguments
- `series::DataFrame`: Input time series data.
- `degree::Float64`: Degree of differentiation.
- `threshold::Float64`: Threshold for drop in weights. Defaults to 1e-5.

# Returns
- `DataFrame`: Fractionally differentiated series.
"""
function fractionalDifferentiationFixed(series::DataFrame, degree::Float64, threshold::Float64 = 1e-5)
    weights = weightingFfd(degree, threshold)
    width = length(weights) - 1
    dataframe = DataFrame(index = series.dates[width + 1:end])
    
    for name in names(series)[2:end]
        seriesFiltered = filter(!ismissing, series[:, [:dates, name]])
        thisRange = width + 1:size(seriesFiltered, 1)
        dataframeFiltered = DataFrame(index = seriesFiltered.dates[thisRange], value = Float64[])
        data = []
        
        for i in thisRange
            day1 = seriesFiltered.dates[i - width]
            day2 = seriesFiltered.dates[i]
            
            if !isfinite(series[series.dates .== day2, name][1])
                continue
            end
            
            append!(data, dot(weights, 
                    filter(row -> row[:dates] in Date(day1):Day(1):day2, seriesFiltered)[!, name]))
        end
        
        dataframeFiltered.value .= data
        dataframe = innerjoin(dataframe, dataframeFiltered, on = :index)
    end
    
    return dataframe
end

"""
    minimumDegreeFFD(input::DataFrame)

Finds the minimum degree value that passes the ADF test for fractionally differentiated series.

# Arguments
- `input::DataFrame`: Input time series data.

# Returns
- `DataFrame`: Results dataframe with ADF statistics.
"""
function minimumDegreeFfd(input::DataFrame)
    out = DataFrame(d = Float64[], adfStat = Float64[], pVal = Float64[], lags = Int[], nObs = Int[], nintyfiveperconf = Float64[], corr = Float64[])
    
    for d in range(0.0, 1.0, length = 11)
        dataframe = DataFrame(dates = Date.(input[:, 1]), priceLog = log.(input[:, :close]))
        differentiated = fractionalDifferentiationFixed(dataframe, d, .01)
        corr = cor(filter(row -> row[:dates] in differentiated[:, 1], dataframe)[:, :priceLog], differentiated[:, 2])
        differentiated = ADFTest(Float64.(differentiated[:, 2]), :constant, 1)
        push!(out, [d, differentiated.stat, pvalue(differentiated), differentiated.lag, differentiated.n, differentiated.cv[2], corr])
    end
    
    return out
end
