using Dates
using DataFrames

"""
    showProgressBar(value::Int, endValue::Int, startTime::Int, barLength::Int=20)

Show a progress bar indicating the completion percentage and remaining time.

# Arguments
- `value::Int`: Current progress value.
- `endValue::Int`: Maximum progress value.
- `startTime::Int`: Start time of the event.
- `barLength::Int=20`: Length of the progress bar.

# Returns
- None. Prints the progress bar to the console.
"""
function showProgressBar(
        value::Int,
        endValue::Int,
        startTime::Int,
        barLength::Int=20
    )

    percent = Float64(value) / endValue
    progressValue = max(0, trunc(Int, round(percent * barLength) - 1))
    arrow = string(repeat("-", progressValue), ">")
    spaces = repeat(" ", barLength - length(arrow))
    remaining = trunc(Int, ((Dates.value(Dates.now() - Dates.DateTime(1970, 1, 1, 0, 0, 0))) /
        1000 - startTime) / value) * (endValue - value) / 60)
    message = string(arrow, spaces)
    println("Completed: [$message] $(trunc(Int, round(percent * 100)))% - $remaining minutes remaining.")
end

"""
    computeEwma(data::Array{Float64}, windowLength::Int=100)

Compute the Exponential Weighted Moving Average (EWMA), EWMA variance, and EWMA standard deviations.

# Arguments
- `data::Array{Float64}`: Array of data.
- `windowLength::Int=100`: Window length for EWMA.

# Returns
- `(ewmaMean, ewmaVariance, ewmaStd)`: Tuple of EWMA mean, variance, and standard deviation.

# Reference
- [Pandas ewm std calculation](https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation)
"""
function computeEwma(
        data::Array{Float64},
        windowLength::Int=100
    )

    N = length(data)
    ewmaMean = Float64[]
    ewmaVariance = Float64[]
    ewmaStd = Float64[]
    α = 2 / Float64(windowLength + 1)

    for i in 1:N
        window = data[1:i]
        m = length(window)
        ω = (1 - α) .^ range(m - 1, step = -1, stop = 0)
        ewma = sum(ω .* window) / sum(ω)
        bias = sum(ω)^2 / (sum(ω)^2 - sum(ω .^ 2))
        var = bias * sum(ω .* (window .- ewma).^2) / sum(ω)
        std = sqrt(var)
        push!(ewmaMean, ewma)
        push!(ewmaVariance, var)
        push!(ewmaStd, std)
    end

    return ewmaMean, ewmaVariance, ewmaStd
end

"""
    groupAndCalculateThresholds(targetCol::Array{Float64}, tickExpectedInit::Int, barSize::Float64)

Group a dataframe based on a feature and then calculate thresholds.

# Arguments
- `targetCol::Array{Float64}`: Target column of tick dataframe.
- `tickExpectedInit::Int`: Initial expected ticks.
- `barSize::Float64`: Initial expected size in each tick.

# Returns
- `(ΔTimes, θAbsolute, thresholds, times, θs, groupingId)`: Tuple of various statistics and grouping information.

# Reference
- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
"""
function groupAndCalculateThresholds(
        targetCol::Array{Float64},
        tickExpectedInit::Int,
        barSize::Float64
    )

    ΔTimes, times = Float64[], Int[]
    timePrev, tickExpected, barExpectedValue = 0, tickExpectedInit, barSize
    N = length(targetCol)
    θAbsolute = zeros(N)
    thresholds = zeros(N)
    θs = zeros(N)
    groupingId = zeros(Int, N)
    θAbsolute[1], θCurrent = abs(targetCol[1]), targetCol[1]
    time = Dates.value(Dates.now() - Dates.DateTime(1970, 1, 1, 0, 0, 0)) / 1000
    groupingIdCurrent = 0

    for i in 2:N
        θCurrent += targetCol[i]
        θs[i] = θCurrent
        thisθAbsolute = abs(θCurrent)
        θAbsolute[i] = thisθAbsolute
        threshold = tickExpected * barExpectedValue
        thresholds[i] = threshold
        groupingId[i] = groupingIdCurrent

        if thisθAbsolute >= threshold
            push!(ΔTimes, time - timePrev)
            push!(times, time)
            tickExpected = max(tickExpectedInit, trunc(Int, i / length(ΔTimes)))
            barExpectedValue = barSize * tickExpected / tickExpectedInit
            θCurrent = 0
            timePrev = time
            groupingIdCurrent += 1
            groupingId[i] = groupingIdCurrent
        end

        time += Dates.value(Dates.now() - Dates.DateTime(1970, 1, 1, 0, 0, 0)) / 1000
    end

    return ΔTimes, θAbsolute, thresholds, times, θs, groupingId
end

using DataFrames
using Statistics
using Combinatorics
using Dates

"""
    generateInfoDrivenBars(tickData::DataFrame, barType::String="volume", tickExpectedInit::Int=2000)

Generate information-driven bars, which can be of type "tick", "volume", or "dollar".

# Arguments

- `tickData::DataFrame`: A DataFrame of tick data, which should include columns: "volumelabeled", "label", and "dollars".
- `barType::String`: The type of bar to generate. Choose "tick", "volume" or "dollar". Default is "volume".
- `tickExpectedInit::Int`: The value of the expected tick. Default is 2000.

# Returns
- `DataFrame`: A DataFrame of the generated bars with columns: "dates", "open", "high", "low", "close", "volume".

# Related Mathematical Formulae

- The bar expected value is calculated as the absolute mean of the input data.

# References

- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
- Page number: 29

# Methodology

- The methodology is described in De Prado's book, page number 29.
"""
function generateInfoDrivenBars(
        tickData::DataFrame,
        barType::String="volume",
        tickExpectedInit::Int=2000
    )::DataFrame

    if barType == "volume"
        inputData = tickData.volumelabeled
    elseif barType == "tick"
        inputData = tickData.label
    elseif barType == "dollar"
        inputData = tickData.dollars
    else
        throw(ArgumentError("Error: unknown barType"))
    end

    barExpectedValue = abs(mean(inputData))
    ΔTimes, θAbsolute, thresholds, times, θs, groupingId = groupAndCalculateThresholds(inputData, tickExpectedInit, barExpectedValue)
    tickData[!, :groupingID] = groupingId
    dates = combine(groupby(tickData, :groupingID), :dates => first => :dates)
    tickDataGrouped = groupby(tickData, :groupingID)
    ohlcvDataframe = ohlcv(tickDataGrouped)
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates)
    colDrop = [:groupingID]
    select!(ohlcvDataframe, Not(colDrop))
    return ohlcvDataframe, θAbsolute, thresholds
end

using DataFrames
using Statistics

"""
    generateOhlcvData(tickDataGrouped::GroupedDataFrame) -> DataFrame

Generate OHLCV (Open, High, Low, Close, Volume) data from a grouped DataFrame of tick data.
This function also calculates the value of trades, mean price, and the mean log return.

# Arguments

- `tickDataGrouped::GroupedDataFrame`: Grouped dataframes with tick data, which should include columns: "price" and "size".

# Returns

- `DataFrame`: A DataFrame of the generated OHLCV data with columns: "Open", "High", "Low", "Close", "Volume", "ValueOfTrades", "PriceMean", "TickCount", and "PriceMeanLogReturn".

# Related Mathematical Formulae

- The value of trades is calculated as the sum of the product of price and size divided by the sum of the size.
- The mean log return is calculated as the logarithm of the mean price minus the logarithm of the mean price of the previous row.

# References

- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
"""
function generateOhlcvData(
        tickDataGrouped::GroupedDataFrame
    )::DataFrame

    ohlcvDataframe = combine(
        tickDataGrouped, :price => first => :Open,
        :price => maximum => :High,
        :price => minimum => :Low,
        :price => last => :Close,
        :size => sum => :Volume,
        AsTable([:price, :size]) => x -> sum(x.price .* x.size) / sum(x.size) => :ValueOfTrades,
        :price => mean => :PriceMean,
        :price => length => :TickCount
    )

    ohlcvDataframe[!, :PriceMeanLogReturn] = log.(ohlcvDataframe.PriceMean) - log.(circshift(ohlcvDataframe.PriceMean, 1))
    ohlcvDataframe[1, :PriceMeanLogReturn] = NaN
    return ohlcvDataframe
end

using DataFrames
using Dates

"""
    generateTimeBar(tickData::DataFrame, frequency::Int=5) -> DataFrame

Generate a time bar DataFrame from tick data by rounding the datetime of each tick to the nearest specified frequency (in minutes).

# Arguments

- `tickData::DataFrame`: A DataFrame of tick data, which should include a column: "dates".
- `frequency::Int`: Frequency for rounding date time (in minutes). Default is 5.

# Returns

- `DataFrame`: A DataFrame of the generated time bars with columns: "Open", "High", "Low", "Close", "Volume", "ValueOfTrades", "PriceMean", "TickCount", and "PriceMeanLogReturn".

# Related Mathematical Formulae

- The datetime of each tick is rounded to the nearest specified frequency (in minutes).

# References

- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
"""
function generateTimeBar(
        tickData::DataFrame,
        frequency::Int=5
    )::DataFrame

    originalDates = tickData.dates
    roundedDates = floor.(originalDates, Dates.Minute(frequency))
    tickData[!, :roundedDates] = roundedDates
    tickDataGrouped = groupby(tickData, :roundedDates)
    ohlcvDataframe = generateOhlcvData(tickDataGrouped)
    return ohlcvDataframe
end

using DataFrames
using Statistics

"""
    generateTickBar(tickData::DataFrame, ticksPerBar::Int=10, numberOfBars::Int=nothing) -> DataFrame

Generate a tick bar DataFrame from tick data by grouping the ticks into bars based on a specified number of ticks per bar or a specified total number of bars.

# Arguments

- `tickData::DataFrame`: A DataFrame of tick data, which should include a column: "dates".
- `ticksPerBar::Int`: Number of ticks in each bar. Default is 10.
- `numberOfBars::Int`: Total number of bars. Default is nothing.

# Returns

- `DataFrame`: A DataFrame of the generated tick bars with columns: "Open", "High", "Low", "Close", "Volume", "ValueOfTrades", "PriceMean", "TickCount", and "PriceMeanLogReturn".

# Related Mathematical Formulae

- The tick data is divided into bars by grouping the ticks based on a specified number of ticks per bar or a specified total number of bars.

# References

- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
"""
function generateTickBar(
        tickData::DataFrame,
        ticksPerBar::Int=10,
        numberOfBars::Int=nothing
    )::DataFrame

    if numberOfBars !== nothing
        ticksPerBar = floor(Int, size(tickData, 1) / numberOfBars)
    end
    
    groupingId = [div(x, ticksPerBar) for x in 0:size(tickData, 1) - 1]
    tickData[!, :groupingId] = groupingId
    tickDataGrouped = groupby(tickData, :groupingId)
    ohlcvDataframe = generateOhlcvData(tickDataGrouped)
    dates = combine(tickDataGrouped, :dates => first => :dates)
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates)
    select!(ohlcvDataframe, Not(:groupingId))
    return ohlcvDataframe
end

using DataFrames
using Statistics

"""
    generateVolumeBar(tickData::DataFrame, volumePerBar::Int=10000, numberOfBars::Int=nothing) -> DataFrame

Generate a volume bar DataFrame from tick data by grouping the ticks into bars based on a specified volume per bar or a specified total number of bars.

# Arguments

- `tickData::DataFrame`: A DataFrame of tick data, which should include columns: "dates" and "size".
- `volumePerBar::Int`: Volume in each bar. Default is 10000.
- `numberOfBars::Int`: Total number of bars. Default is nothing.

# Returns

- `DataFrame`: A DataFrame of the generated volume bars with columns: "Open", "High", "Low", "Close", "Volume", "ValueOfTrades", "PriceMean", "TickCount", and "PriceMeanLogReturn".

# References

- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
"""
function generateVolumeBar(
        tickData::DataFrame,
        volumePerBar::Int=10000,
        numberOfBars::Int=nothing
    )::DataFrame

    tickData[!, :volumeCumulated] = cumsum(tickData.size)
    if numberOfBars !== nothing
        totalVolume = tickData.volumeCumulated[end]
        volumePerBar = round(totalVolume / numberOfBars; digits = 2)
    end
    tickData[!, :groupingId] = div.(tickData.volumeCumulated, volumePerBar)
    tickDataGrouped = groupby(tickData, :groupingId)
    ohlcvDataframe = generateOhlcvData(tickDataGrouped)
    dates = combine(tickDataGrouped, :dates => first => :dates)
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates)
    select!(ohlcvDataframe, Not(:groupingId))
    return ohlcvDataframe
end

"""
    generateDollarBar(tickData::DataFrame, dollarPerBar::Int=100000, numberOfBars::Int=nothing) -> DataFrame

Generate a dollar bar DataFrame from tick data by grouping the ticks into bars based on a specified dollar amount per bar or a specified total number of bars.

# Arguments

- `tickData::DataFrame`: A DataFrame of tick data, which should include columns: "dates", "price", and "size".
- `dollarPerBar::Int`: Dollar amount in each bar. Default is 100000.
- `numberOfBars::Int`: Total number of bars. Default is nothing.

# Returns

- `DataFrame`: A DataFrame of the generated dollar bars with columns: "Open", "High", "Low", "Close", "Volume", "ValueOfTrades", "PriceMean", "TickCount", and "PriceMeanLogReturn".

# References

- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
"""
function generateDollarBar(
        tickData::DataFrame,
        dollarPerBar::Int=100000,
        numberOfBars::Int=nothing
    )::DataFrame

    tickData[!, :dollar] = tickData.price .* tickData.size
    tickData[!, :cumulativeDollars] = cumsum(tickData.dollar)
    if numberOfBars !== nothing
        dollarsTotal = tickData.cumulativeDollars[end]
        dollarPerBar = round(dollarsTotal / numberOfBars; digits = 2)
    end
    tickData[!, :groupingId] = div.(tickData.cumulativeDollars, dollarPerBar)
    tickDataGrouped = groupby(tickData, :groupingId)
    ohlcvDataframe = generateOhlcvData(tickDataGrouped)
    dates = combine(tickDataGrouped, :dates => first => :dates)
    insertcols!(ohlcvDataframe, 1, :dates => dates.dates)
    select!(ohlcvDataframe, Not(:groupingId))
    return ohlcvDataframe
end

using LinearAlgebra
using DataFrames

"""
    pcaWeights(covarianceMatrix::Matrix, riskDistribution::Vector{Float64}=ones(size(covarianceMatrix, 1)), riskTarget::Float64=1.0) -> Vector{Float64}

Calculate hedging weights using covariance matrix, risk distribution (riskDistribution), and risk target (riskTarget).

# Arguments

- `covarianceMatrix::Matrix`: Covariance matrix.
- `riskDistribution::Vector{Float64}`: Risk distribution. Default is a vector of ones with the same length as the number of assets in the covariance matrix.
- `riskTarget::Float64`: Risk target. Default is 1.0.

# Returns

- `Vector{Float64}`: The vector of hedging weights.

# Related Mathematical Formulae

Given a covariance matrix `Σ`, the eigendecomposition of `Σ` is `Σ = VΛV'`, where `V` is the matrix of eigenvectors, and `Λ` is the diagonal matrix of eigenvalues. The hedging weights are given by `weights = V * loads`, where `loads = σ * (riskDistribution ./ Λ) .^ 0.5`.

# References

- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Methodology: Page 36
"""
function pcaWeights(
        covarianceMatrix::Matrix,
        riskDistribution::Vector{Float64}=ones(size(covarianceMatrix, 1)),
        riskTarget::Float64=1.0
    )::Vector{Float64}

    Λ, V = eigen(covarianceMatrix)
    indices = reverse(sortperm(Λ.values))
    Λ = Diagonal(Λ.values[indices])
    V = V[:, indices]

    loads = riskTarget * (riskDistribution ./ diagm(Λ)) .^ 0.5
    weights = V * loads
    return weights
end

"""
    symmetricCusumFilter(input::DataFrame, threshold::Float64) -> Vector{Any}

Implement the symmetric CUSUM filter.

# Arguments

- `input::DataFrame`: DataFrame of prices and dates.
- `threshold::Float64`: Threshold.

# Returns

- `Vector{Any}`: The vector of timestamps when the filter triggers.

# References

- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons. Methodology: Page 39
"""
function symmetricCusumFilter(
        input::DataFrame,
        threshold::Float64
    )::Vector{Any}

    timeEvents = []
    shiftPositive, shiftNegative = 0.0, 0.0
    Δprice = diff(input[:, 2])

    for i in 1:length(Δprice)
        shiftPositive = max(0.0, shiftPositive + Δprice[i])
        shiftNegative = min(0.0, shiftNegative + Δprice[i])

        if shiftNegative < -threshold
            shiftNegative = 0.0
            push!(timeEvents, input[i+1, 1])
        elseif shiftPositive > threshold
            shiftPositive = 0.0
            push!(timeEvents, input[i+1, 1])
        end
    end
    
    return timeEvents
end
