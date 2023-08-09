"""
    Function: Combines grouped dataframe to create a new one with information about prices and volume.
    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: n/a
"""
function ohlcv(tick_data_grouped)
    ohlcv_dataframe = combine(tick_data_grouped, :price => first => :open,
               :price => maximum => :high,
               :price => minimum => :low,
               :price => last => :close,
               :size => sum => :volume,
               AsTable([:price, :size]) => x -> sum(x.price .* x.size) / sum(x.size),
               :price => mean => :price_mean,
               :price => length => :tick_count)
    DataFrames.rename!(ohlcv_dataframe, :price_size_function => :value_of_trades)
    ohlcv_dataframe.price_mean_log_return = log.(ohlcv_dataframe.price_mean) - log.(circshift(ohlcv_dataframe.price_mean, 1))
    ohlcv_dataframe.price_mean_log_return[1] = NaN
    return ohlcv_dataframe
end

"""
    Function: Takes a dataframe and generates a time bar dataframe.
    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: n/a
"""
function time_bar(tick_data, frequency = 5)
    dates = tick_data.dates
    dates_copy = copy(dates)
    tick_data.dates = floor.(dates_copy, Dates.Minute(frequency))
    tick_data_grouped = DataFrames.groupby(tick_data, :dates)
    ohlcv_dataframe = ohlcv(tick_data_grouped)
    tick_data.dates = dates
    return ohlcv_dataframe
end

"""
    Function: The sequence of weights used to compute each value of the fractionally differentiated series.
    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: 79
"""
function weighting(degree, size)
    ω = [1.]
    for k ∈ 2:size
        this_ω = -ω[end] / (k - 1) * (degree - k + 2)
        append!(ω, this_ω)
    end
    return reverse(ω)
end

"""
    Function: Plot weights.
    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: 79
"""
function plot_weights(degree_range, number_degrees, number_weights)
    ω = DataFrames.DataFrame(index = collect(number_weights - 1:-1:0))
    for degree ∈ range(degree_range[1], degree_range[2], length = number_degrees)
        degree = round(degree; digits = 2)
        this_ω = weighting(degree, number_weights)
        this_ω = DataFrames.DataFrame(index = collect(number_weights - 1:-1:0), ω = this_ω)
        ω = outerjoin(ω, this_ω, on = :index, makeunique = true)
    end
    DataFrames.rename!(ω, names(ω)[2:end] .=> string.(range(degree_range[1], degree_range[2], length = number_degrees)))
    plot(ω[:, 1], Matrix(ω[:, 2:end]), label = reshape(names(ω)[2:end], (1, number_degrees)), background = :transparent)
end

"""
    Function: Standard fractionally differentiated.
    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: 82
"""
function frac_diff(series, degree, threshold = 0.01)
    weights = weighting(degree, size(series)[1])
    weights_normalized = cumsum(broadcast(abs, weights), dims = 1)
    weights_normalized /= weights_normalized[end]
    drop = length(filter(x -> x > threshold, weights_normalized))
    dataframe = DataFrames.DataFrame(index = filter(!ismissing, series[:, [:dates]])[
                range(drop + 1, stop = size(filter(!ismissing, series[:, [:dates]]))[1], step = 1), 1])
    for name ∈ Symbol.(names(series))[2:end]
        series_filtered = filter(!ismissing, series[:, [:dates, name]])
        this_range = range(drop + 1, stop = size(series_filtered)[1], step = 1)
        dataframe_filtered = DataFrames.DataFrame(index = series_filtered[this_range, 1], value = repeat([0.], length(this_range)))
        data = []
        for i ∈ range(drop + 1, stop = size(series_filtered)[1], step = 1)
            date = series_filtered[i, 1]
            price = series[series.dates .== date, name][1]
            if !isfinite(price)
                continue
            end
            try
                append!(data, Statistics.dot(
                    weights[length(weights) - i + 1:end, :], 
                    filter(row -> row[:dates] in collect(series_filtered[1, 1]:Day(1):date), series_filtered)[:, name]))
            catch
                continue
            end
        end
        dataframe_filtered.value = data
        dataframe = DataFrames.innerjoin(dataframe, dataframe_filtered, on = :index)
    end
    return dataframe
end

"""
    Function: Weights for fixed-width window method.
    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: 83
"""
function weighting_ffd(degree, threshold)
    ω = [1.]
    k = 1
    while abs(ω[end]) >= threshold 
        this_ω = -ω[end] / k * (degree - k + 1)
        append!(ω, this_ω)
        k += 1
    end
    return reverse(ω)[2:end]
end

"""
    Function: Fixed-width window fractionally differentiated method.
    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: 83
"""
function frac_diff_fixed(series, degree, threshold = 1e-5)
    weights = weighting_ffd(degree, threshold)
    width = length(weights) - 1
    dataframe = DataFrames.DataFrame(index = series[
                range(width + 1, stop = size(series)[1], step = 1), 1])
    for name ∈ Symbol.(names(series))[2:end]
        series_filtered = filter(!ismissing, series[:, [:dates, name]])
        this_range = range(width + 1, stop = size(series_filtered)[1], step = 1)
        dataframe_filtered = DataFrames.DataFrame(index = series_filtered[this_range, 1])
        data = []
        for i ∈ range(width + 1, stop = size(series_filtered)[1], step = 1)
            day1 = series_filtered[i - width, 1]
            day2 = series_filtered[i, 1]
            if !isfinite(series[series.dates .== day2, name][1])
                continue
            end
            append!(data, Statistics.dot(
                    weights, 
                    filter(row -> row[:dates] in collect(day1:Day(1):day2), series_filtered)[:, name]))
        end
        dataframe_filtered.value = data
        dataframe = DataFrames.innerjoin(dataframe, dataframe_filtered, on = :index)
    end
    return dataframe
end

"""
    Function: Find the minimum degree value that passes the ADF test.
    Reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    Methodology: 85
"""
function min_ffd(input)
    out = DataFrames.DataFrame(d = [], adf_stat = [], p_val = [], lags = [], n_obs = [], nintyfive_per_conf = [], corr = [])
    for d in range(0, 1, length = 11)
        dataframe = DataFrames.DataFrame(dates = Date.(input[:, 1]), pricelog = log.(input[:, :close]))
        differentiated = frac_diff_fixed(dataframe, d, .01)
        corr = cor(filter(row -> row[:dates] in differentiated[:, 1],dataframe)[:, :pricelog], differentiated[:, 2])
        differentiated = HypothesisTests.ADFTest(Float64.(differentiated[:, 2]),:constant, 1)
        push!(out, [d,differentiated.stat, pvalue(differentiated), differentiated.lag, differentiated.n, differentiated.cv[2], corr])
    end
    return out
end
