using DataFrames
using Distributions
using Statistics

"""
    numberConcurrentEvents(events::DataFrame, returnedLabel::DataFrame, molecule::Vector)

Compute the number of concurrent events (both long and short) for each event in a given molecule.

# Parameters:
- `events::DataFrame`: DataFrame containing events data.
- `returnedLabel::DataFrame`: DataFrame containing return and label for each period.
- `molecule::Vector`: Index on which the function must apply.

# Returns:
- `DataFrame`: DataFrame containing the number of concurrent long and short events for each event.

# References:
- De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
- Methodology 51
"""
function numberConcurrentEvents(
        events::DataFrame,
        returnedLabel::DataFrame,
        molecule::Vector
    )
    eventsFiltered = filter(row -> row[:date] in molecule, events)
    
    concurrency = DataFrame(
        Dates = events.date,
        activeLong = zeros(Int, size(eventsFiltered, 1)),
        activeShort = zeros(Int, size(eventsFiltered, 1))
    )
    
    for (i, idx) in enumerate(eventsFiltered.date)
        idxFilter = events.date .<= idx
        activeFilter = events.timestamp .> idx
        positiveReturn = returnedLabel.ret .>= 0
        
        maxlen = maximum([length(positiveReturn), length(activeFilter), length(idxFilter)])
        
        positiveReturn = vcat(positiveReturn, falses(maxlen - length(positiveReturn)))
        idxFilter = vcat(idxFilter, falses(maxlen - length(idxFilter)))
        activeFilter = vcat(activeFilter, falses(maxlen - length(activeFilter)))
        
        condition = idxFilter .& activeFilter .& positiveReturn
        activeLongSet = Set(eventsFiltered.date[condition])
        concurrency.activeLong[i] = length(activeLongSet)
        
        negativeReturn = returnedLabel.ret .< 0
        negativeReturn = vcat(negativeReturn, falses(maxlen - length(negativeReturn)))
        condition = idxFilter .& activeFilter .& negativeReturn
        activeShortSet = Set(eventsFiltered.date[condition])
        concurrency.activeShort[i] = length(activeShortSet)
    end

    concurrency.ct = concurrency.activeLong - concurrency.activeShort
    return concurrency
end

"""
    mixtureNormalCdf(x::AbstractVector; weights=[0.647, 0.353], means=[2.79, -3.03], std=[9.62, 12.0])

Compute the cumulative distribution function (CDF) of a mixture model.

# Parameters:
- `x::AbstractVector`: Input data for calculating the CDF.
- `weights::AbstractVector`: Weights for the mixture model (default: [0.647, 0.353]).
- `means::AbstractVector`: Means of each Gaussian distribution in the model (default: [2.79, -3.03]).
- `std::AbstractVector`: Standard deviations of each Gaussian distribution in the model (default: [9.62, 12.0]).

# Returns:
- `Float64`: Calculated CDF value.
"""
function mixtureNormalCdf(
        x::AbstractVector;
        weights::AbstractVector=[0.647, 0.353],
        means::AbstractVector=[2.79, -3.03],
        std::AbstractVector=[9.62, 12.0]
    )
    mix = MixtureModel(Normal[Normal(means[1], std[1]), Normal(means[2], std[2])], weights)
    mcdf = cdf(mix, x)
    return float(mcdf)
end

using DataFrames, Statistics, Distributions

"""
    Calculate the bet size using a Gaussian mixture model.

    This function calculates the bet size based on the given concurrency and Gaussian mixture model parameters.

    Parameters
    ----------
    concurrency : Float64
        Concurrency.
    weights : AbstractVector
        Weights of the normal distribution for the mixture model.
    means : AbstractVector
        Means of the normal distribution.
    standardDeviations : AbstractVector
        Standard deviation of the normal distribution.

    Returns
    -------
    Float64
        Calculated bet size.

    References
    ----------
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    - Methodology p.142

"""
function calculateGaussianBet(
    concurrency::Float64,
    weights::AbstractVector,
    means::AbstractVector,
    standardDeviations::AbstractVector
) :: Float64

    mixCDF = x -> sum(w * cdf(Normal(m, s), x) for (w, m, s) in zip(weights, means, standardDeviations))

    numerator = mixCDF(concurrency) - mixCDF(0)
    denominator = mixCDF(0) * (concurrency >= 0.0 ? -1.0 + 1.0 : 1.0)

    return numerator / denominator
end

"""
    Calculate bet size statistics.

    This function calculates statistics related to bet sizes based on the given probability and number of label classes.

    Parameters
    ----------
    probability : AbstractVector
        Probability that the label takes place.
    numberOfClasses : Int
        Number of label classes.

    Returns
    -------
    AbstractVector
        Calculated bet size statistics.

    References
    ----------
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    - Methodology p.142

"""
function betSizeStatistics(
    probability::AbstractVector,
    numberOfClasses::Int
) :: AbstractVector

    statistic = (probability .- (1 / numberOfClasses)) ./ sqrt.(probability .* (1 .- probability))
    model = Normal(0, 1)

    return 2 * cdf(model, statistic) .- 1
end

"""
    Calculate bet size statistics for a range of label classes.

    This function calculates bet size statistics for a range of label class probabilities.

    Parameters
    ----------
    numberOfClasses : Int
        Number of label classes.

    Returns
    -------
    DataFrame
        DataFrame with calculated bet size statistics for the given range of probabilities.

"""
function betSizeForRange(
    numberOfClasses::Int
) :: DataFrame

    prob = range(0, stop=1, length=10000)
    stats = betSizeStatistics(prob, numberOfClasses)

    return DataFrame(probability = prob, statistics = stats)
end

"""
    Calculate average active signals.

    This function calculates the average active signals based on the DataFrame of signals.

    Parameters
    ----------
    dataFrameOfSignals : DataFrame
        DataFrame that has signals.

    Returns
    -------
    DataFrame
        DataFrame with calculated average active signals.

    References
    ----------
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    - Methodology p.144

"""
function averageActiveSignals(
    dataFrameOfSignals::DataFrame
) :: DataFrame

    barrierSet = union(Set(dropmissing(dataFrameOfSignals).t1), Set(dataFrameOfSignals.Dates))
    index = sort(collect(barrierSet))
    out = averageActiveSignalsMultiProcessing(dataFrameOfSignals, index)

    return out
end

"""
    Calculate average active signals for a molecule.

    This function calculates the average active signals for a given molecule based on the DataFrame of signals.

    Parameters
    ----------
    dataFrameOfSignals : DataFrame
        DataFrame that has signals.
    molecule : AbstractVector
        Index that the function must apply on.

    Returns
    -------
    DataFrame
        DataFrame with calculated average active signals for the given molecule.

    References
    ----------
    - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    - Methodology p.144

"""
function averageActiveSignalsMultiProcessing(
    dataFrameOfSignals::DataFrame,
    molecule::AbstractVector
) :: DataFrame

    signals = copy(dataFrameOfSignals)
    output = DataFrame(date = molecule, signal = zeros(length(molecule)))

    for (i, loc) in enumerate(molecule)
        activeEvents = filter(row -> row[:Dates] <= loc && (ismissing(row[:t1]) || row[:t1] > loc), signals)
        output[i, :signal] = isempty(activeEvents) ? 0.0 : mean(activeEvents[:signal])
    end

    return sort!(output, [:date])
end

"""
    Discretize signals.

    This function discretizes the given signal DataFrame based on the provided step size.

    Args:
        signal (AbstractVector): DataFrame that contains signals.
        stepSize (Float64): Step size.

    Returns:
        AbstractVector: Discretized signal.

"""
function discreteSignal(
    signal::AbstractVector,
    stepSize::Float64
)::AbstractVector

    signal = round.(signal ./ stepSize) * stepSize
    signal[signal .> 1] .= 1
    signal[signal .< -1] .= -1
    return signal
end

"""
    Generate signals.

    This function generates signals based on the provided events, step size, estimation results, and number of classes.

    Args:
        events (DataFrame): DataFrame for events.
        stepSize (Float64): Step size for discretization.
        estimationResult (DataFrame): DataFrame that contains probability and prediction.
        nClasses (Int): Number of classes.

    Returns:
        DataFrame: Generated signals.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology p.145

"""
function generateSignal(
    events::DataFrame,
    stepSize::Float64,
    estimationResult::DataFrame,
    nClasses::Int
)::DataFrame
    if size(prob)[1] == 0
        return
    end
    probability = estimationResult.probability
    predictions = estimationResult.prediction
    signal = (probability .- 1.0/nClasses) ./ sqrt.(probability .* (1 .- probability))
    model = Normal(0, 1)
    signal = predictions .* (2 .* cdf.(model, signal) .- 1)

    if "side" in names(events)
        signal = signal .* filter(row -> row[:Dates] in estimationResult.date, events).side
    end

    finalSignal = DataFrame(Date = estimationResult.date, signal = signal)
    finalSignal.t1 = events.t1

    finalSignal = averageActiveSignals(finalSignal)
    finalSignal = discreteSignal(finalSignal, stepSize)

    return finalSignal
end

"""
    Calculate the coefficient ω for dynamic position size and limit price.

    This function calculates the coefficient ω used in dynamic position size and limit price calculations.

    Args:
        x (Float64): Divergence between the current market price and the forecast.
        m (Float64): Bet size.

    Returns:
        Float64: Coefficient ω.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology p.145 SNIPPET 10.4

"""
function ω(
    x::Float64,
    m::Float64
)::Float64

    return x^2 * (1 / m^2 - 1)
end

"""
    Calculate the size of bet for dynamic position size.

    This function calculates the size of bet based on the coefficient ω and the divergence between the current market price and the forecast.

    Args:
        ω (Float64): Coefficient that regulates the width of the sigmoid function.
        x (Float64): Divergence between the current market price and the forecast.

    Returns:
        Float64: Size of bet.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology p.145 SNIPPET 10.4

"""
function calculateSizeOfBet(
    ω::Float64,
    x::Float64
)::Float64

    return x * (ω + x^2)^(-0.5)
end

"""
    Calculate the target position size.

    This function calculates the target position size based on the coefficient ω, predicted price, actual price, and maximum absolute position size.

    Args:
        ω (Float64): Coefficient that regulates the width of the sigmoid function.
        f (Float64): Predicted price.
        actualPrice (Float64): Actual price.
        maximumPositionSize (Int): Maximum absolute position size.

    Returns:
        Int: Target position size.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology p.145 SNIPPET 10.4

"""
function calculateTargetPosition(
    ω::Float64,
    f::Float64,
    actualPrice::Float64,
    maximumPositionSize::Int
)::Int

    return trunc(Int, (calculateSizeOfBet(ω, f - actualPrice) * maximumPositionSize))
end

"""
    Calculate the inverse price for dynamic position size.

    This function calculates the inverse price based on the predicted price, coefficient ω, and bet size.

    Args:
        f (Float64): Predicted price.
        ω (Float64): Coefficient that regulates the width of the sigmoid function.
        m (Float64): Bet size.

    Returns:
        Float64: Inverse price.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology p.145 SNIPPET 10.4

"""
function calculateInversePrice(
    f::Float64,
    ω::Float64,
    m::Float64
)::Float64

    return f - m * (ω / (1 - m^2))^(0.5)
end

"""
    Calculate the limit price for dynamic position size.

    This function calculates the limit price based on the target position size, current position, predicted price, coefficient ω, and maximum absolute position size.

    Args:
        targetPositionSize (Int): Target position size.
        cPosition (Int): Current position.
        f (Float64): Predicted price.
        ω (Float64): Coefficient that regulates the width of the sigmoid function.
        maximumPositionSize (Int): Maximum absolute position size.

    Returns:
        Float64: Limit price.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology p.145 SNIPPET 10.4

"""
function calculateLimitPrice(
    targetPositionSize::Int,
    cPosition::Int,
    f::Float64,
    ω::Float64,
    maximumPositionSize::Int
)::Float64

    if targetPositionSize >= cPosition
        sgn = 1
    else
        sgn = -1
    end
    lP = 0
    for i in abs(cPosition + sgn):abs(targetPositionSize)
        lP += calculateInversePrice(f, ω, i / float(maximumPositionSize))
    end
    lP /= targetPositionSize - cPosition
    return lP
end
