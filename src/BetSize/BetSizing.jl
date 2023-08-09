
#include("labeling.jl")
#include("hpc.jl")

using DataFrames
using Distributions


"""
    Calculate the number of concurrent events for each event in a given molecule.

    This function computes the number of concurrent events (both long and short) for each event in a given molecule, 
    based on the provided events and their corresponding returns and labels.

    Args:
        events (DataFrame): DataFrame that contains events data.
        returnandlable (DataFrame): DataFrame that contains return and label of each period.
        molecule (Vector): The index on which the function must apply.

    Returns:
        DataFrame: A DataFrame containing the number of concurrent long and short events for each event.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology 51

"""
function n_concurrency_events(events::DataFrame, return_and_label::DataFrame, molecule::Vector)

    events_filtered = filter(row -> row[:date] in molecule, events)
    
    concurrency = DataFrame(Dates = events.date, active_long = zeros(Int, size(events_filtered)[1]), active_short = zeros(Int, size(events_filtered)[1]))
    
    for (i, idx) in enumerate(events_filtered.date)
        index_less_than_idx = events.date .<= idx
        events_more_than_idx = events.timestamp .> idx
        out_is_positive = return_and_label.ret .>= 0
        
        maximum_length = maximum([length(out_is_positive), length(events_more_than_idx), length(index_less_than_idx)])
        
        out_is_positive = vcat(out_is_positive, falses(maximum_length - length(out_is_positive)))
        index_less_than_idx = vcat(index_less_than_idx, falses(maximum_length - length(index_less_than_idx)))
        events_more_than_idx = vcat(events_more_than_idx, falses(maximum_length - length(events_more_than_idx)))

        condition = index_less_than_idx .& events_more_than_idx .& out_is_positive
        my_set = events_filtered.date[condition]

        df_long_active_idx = Set(my_set)
        concurrency.active_long[i] = length(df_long_active_idx)

        out_is_negative = return_and_label.ret .< 0
        
        out_is_negative = vcat(out_is_negative, falses(maximum_length - length(out_is_negative)))
        condition = index_less_than_idx .& events_more_than_idx .& out_is_negative
        my_set = events_filtered.date[condition]

        df_short_active_index = Set(my_set)
        concurrency.active_short[i] = length(df_short_active_index)
    end

    concurrency.ct = concurrency.active_long - concurrency.active_short
    return concurrency
end


"""
    Calculate the cumulative distribution function (CDF) of a mixture model.

    This function computes the CDF of a mixture model given input data.

    Args:
        x (AbstractVector): Input data for calculating the CDF on it.
        weights (AbstractVector): Weights for the mixture model.
        means (AbstractVector): Means of each Gaussian distribution in the model.
        std (AbstractVector): Standard deviation of each Gaussian distribution in the model.

    Returns:
        Float64: The calculated CDF value.

"""
function mixtureNormalCdf(x::AbstractVector; weights=[0.647, 0.353], means=[2.79, -3.03], std=[9.62, 12.0])
    mix = MixtureModel(Normal[
        Normal(means[1], std[1]),
        Normal(means[2], std[2])],
        [weights[1], weights[2]])

    mcdf = cdf(mix, x)
    return float(mcdf)
end

"""
    Calculate the bet size using a Gaussian mixture model.

    This function calculates the bet size based on the given concurrency and Gaussian mixture model parameters.

    Args:
        c (Float64): Concurrency.
        weights (AbstractVector): Weights of the normal distribution for the mixture model.
        means (AbstractVector): Means of the normal distribution.
        sd (AbstractVector): Standard deviation of the normal distribution.

    Returns:
        Float64: Calculated bet size.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology p.142

"""
function gaussianBet(c::Float64, weights::AbstractVector, means::AbstractVector, sd::AbstractVector)
    if c >= 0.0
        return (mixtureNormalCdf(c; weights=weights, means=means, std=sd) - 
                mixtureNormalCdf(0; weights=weights, means=means, std=sd)) /
               (-mixtureNormalCdf(0; weights=weights, means=means, std=sd) + 1)
    else
        return (mixtureNormalCdf(c; weights=weights, means=means, std=sd) - 
                mixtureNormalCdf(0; weights=weights, means=means, std=sd)) /
               mixtureNormalCdf(0; weights=weights, means=means, std=sd)
    end
end

"""
    Calculate bet size statistics.

    This function calculates statistics related to bet sizes based on the given probability and number of label classes.

    Args:
        probability (AbstractVector): Probability that the label takes place.
        nClasses (Int): Number of label classes.

    Returns:
        AbstractVector: Calculated bet size statistics.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology p.142

"""
function betSizeProbability(probability::AbstractVector, nClasses::Int)
    statistic = (probability .- (1/nClasses)) ./ sqrt.(probability .* (1 .- probability))
    model = Normal(0, 1)
    return 2*cdf(model, statistic) .- 1
end

"""
    Calculate bet size statistics for a range of label classes.

    This function calculates bet size statistics for a range of label class probabilities.

    Args:
        nClasses (Int): Number of label classes.

    Returns:
        DataFrame: DataFrame with calculated bet size statistics for the given range of probabilities.

"""
function betSizeForRange(nClasses::Int)
    prob = range(0, stop=1, length=10000)
    return DataFrame(probability = betSizeProbability(prob, nClasses), index=prob)
end

"""
    Calculate average active signals.

    This function calculates the average active signals based on the DataFrame of signals.

    Args:
        DataFrameofSignals (DataFrame): DataFrame that has signals.

    Returns:
        DataFrame: DataFrame with calculated average active signals.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology p.144

"""
function averageActiveSignals(DataFrameofSignals::DataFrame)
    setofbarrier = Set((dropmissing(DataFrameofSignals)).t1)
    setofbarrier = union(setofbarrier, DataFrameofSignals.Dates)
    index = [i for i in setofbarrier]
    sort!(index)
    out = mpDataFrameObj(averageActiveSignalsMultiProcessing, :molecule, index; DataFrameofSignals=DataFrameofSignals)
    return out
end

using DataFrames

"""
    Calculate average active signals for a molecule.

    This function calculates the average active signals for a given molecule based on the DataFrame of signals.

    Args:
        DataFrameofSignals (DataFrame): DataFrame that has signals.
        molecule (AbstractVector): Index that the function must apply on.

    Returns:
        DataFrame: DataFrame with calculated average active signals for the given molecule.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology p.144

"""
function averageActiveSignalsMultiProcessing(;DataFrameofSignals, molecule::AbstractVector)
    signals = copy(DataFrameofSignals) # Filter by molecule
    output = DataFrame(date = molecule, signal = zeros(length(molecule))) # Create DataFrame 

    for (i, loc) in enumerate(molecule)
        sig_lessthan_loc = signals.Dates .<= loc # Select events starting before loc
        loc_lessthan_t1 = signals.t1 .> loc # Select events ending after loc

        is_missing = ismissing.(signals.t1) # Select missing barriers
     
        df0 = sig_lessthan_loc .& (loc_lessthan_t1 .| is_missing) # Select events indices that contain loc 
        act = signals.Dates[df0]     # Select events that contain loc 
        if length(act) > 0
            output[i, :signal] = mean(signals.signal[df0])
        else
            output[i, :signal]  = 0
        end
    end
    output = sort!(output, [:date])  # Sort DataFrame by date 

    return output
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
function discreteSignal(signal::AbstractVector, stepSize::Float64)
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
        EstimationResult (DataFrame): DataFrame that contains probability and prediction.
        nClasses (Int): Number of classes.

    Returns:
        DataFrame: Generated signals.

    References:
        - De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
        - Methodology p.145

"""
function generateSignal(events::DataFrame, stepSize::Float64, EstimationResult::DataFrame, nClasses::Int)
    if size(prob)[1] == 0
        return
    end
    probability = EstimationResult.probability
    predictions = EstimationResult.prediction
    signal = (probability .- 1.0/nClasses) ./ sqrt.(probability .* (1 .- probability))
    model = Normal(0, 1)
    signal = predictions .* (2 .* cdf.(model, signal) .- 1)

    if "side" in names(events)
        signal = signal .* filter(row -> row[:Dates] in EstimationResult.date, events).side
    end

    finalSignal = DataFrame(Date = EstimationResult.date, signal = signal)
    finalSignal.t1 = events.t1

    finalSignal = averageActiveSignals(finalSignal)
    finalSignal = discreteSignal(finalSignal, stepSize)

    return finalSignal
end

"""----------------------------------------------------------------------
    function: DYNAMIC POSITION SIZE AND LIMIT PRICE 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.145 SNIPPET 10.4 
----------------------------------------------------------------------"""

function ω(x, # divergence between the current market price and the forecast
           m) # bet size 
    return x^2*(1/m^2-1)
end


function SizeOfBet(ω ,# coefficient that regulates the width of the sigmoid function
                   x) # divergence between the current market price and the forecast               
    return x*(ω+x^2)^(-0.5)    
end


function getTPos(ω, # coefficient that regulates the width of the sigmoid function
                 f, # predicted price
                 acctualprice , # acctual price 
                 maximumpositionsize) #  maximum absolute position size
    return trunc(Int, (SizeOfBet(ω, f-acctualprice)*maximumpositionsize))   
end


function InversePrice(f, # predicted price
                      ω, #coefficient that regulates the width of the sigmoid function
                      m) # betsize 
    return f-m*(ω/(1-m^2))^(.5)
end


function limitPrice(TargetPositionSize, #  target position size
                    cPosition, # current position
                    f, #  predicted price
                    ω, # coefficient that regulates the width of the sigmoid function
                    maximumpositionsize)  # maximum absolute position size
    if TargetPositionSize >=  cPosition
        sgn = 1
    else
        sgn = -1
    end
    lP = 0
    for i in abs(cPosition+sgn):abs(TargetPositionSize)
        lP += invPrice(f,ω,i/float(maximumpositionsize))
    end
    lP /= TargetPositionSize-cPosition
    return lP    
end




