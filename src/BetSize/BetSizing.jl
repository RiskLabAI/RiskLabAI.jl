
#include("labeling.jl")
#include("hpc.jl")

"""----------------------------------------------------------------------
    function: expand label tO incorporate meta-labeling
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 51
----------------------------------------------------------------------"""
function nConcurrencyEvents( events, # DataFrame that has events
                        returnandlable, # DateFrame that has return and label of each period
                        molecule) # index that function must apply on it

    eventsfiltered = filter(row -> row[:date] in molecule, events) # filter events respect to molecule

    concurrency = DataFrame(Dates=events.date,active_long = zeros(size(eventsfiltered)[1]), active_short =  zeros(size(eventsfiltered)[1])) 
    # create dataframe that contain number of concurrent label for each events 

    for (i, idx) in enumerate(eventsfiltered.date)
         
        indexlessthanidx = events.date .<= idx # store events that before idx
        eventsmorethanidx = events.timestamp .> idx # store events that end after idx
        outispositive = returnandlable.ret .>= 0  # store events have positive returns

        maximumlength = maximum([length(outispositive), length(eventsmorethanidx), length(indexlessthanidx)])  
        
        # extend each array to maximum length

        if length(outispositive) < maximumlength
            append!(outispositive, Bool.(zeros(maximumlength-length(outispositive))))
        end
        if length(indexlessthanidx) < maximumlength
            append!(indexlessthanidx, Bool.(zeros(maximumlength-length(indexlessthanidx))))
        end
        if length(eventsmorethanidx) < maximumlength
            append!(eventsmorethanidx, Bool.(zeros(maximumlength-length(eventsmorethanidx))))
        end

        cond = indexlessthanidx .& eventsmorethanidx .& outispositive # compute intersection of arrays for selecting concurrent lebeled period that contains idx

        mySet = eventsfiltered.date[cond] # select data has our condition 

        dflongactiveidx = Set(mySet) #convert myset to set for eliminate redundant elements

        concurrency.active_long[i] = length(dflongactiveidx) #set active_long of idx to length of dflongactiveidx


        # we repeate this procedure for negative return same as postive returns
        outisnegative = returnandlable.ret .< 0
        maximumlength = maximum([length(outisnegative), length(eventsmorethanidx), length(indexlessthanidx)])
        if length(outisnegative) < maximumlength
            append!(outisnegative, Bool.(zeros(maximumlength-length(outisnegative))))
        end
        cond = indexlessthanidx .& eventsmorethanidx .& outisnegative
        mySet = eventsfiltered.date[cond]
        dfshortactiveindex = Set(mySet)
        concurrency.active_short[i] = length(dfshortactiveindex)

    end

    concurrency.ct = concurrency.active_long - concurrency.active_short # compute differences of active_long and active_short for each events
    return concurrency
end


"""----------------------------------------------------------------------
    function: Mixture Model CDF
    reference: -
    methodology: -
----------------------------------------------------------------------"""
function MixtureNormalCDF(x; # input data for calculating cdf on it 
                          weights=[0.647, 0.353], # weights for mixture model
                          means=[2.79, -3.03],  # means of each gaussian distribution on model
                          std=[9.62, 12.0]) # standard deviation of each gaussian distribution
    # define mixture model
    mix = MixtureModel(Normal[
        Normal(means[1], std[1]),
        Normal(means[2], std[2])],
        [weights[1], weights[2]])

    mcdf = cdf(mix, x) # calculate cdf
    return float(mcdf)
end

"""----------------------------------------------------------------------
    function: Calculation of bet size with gaussian mixture model
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.142
----------------------------------------------------------------------"""
function GaussianBet(c; # concurrency
                     weights, #weights of Normal distribution for mixture model
                     means,  # means of normal distribution
                     sd) # standard deviation of normal distribution
    if c >= 0.0
        return (MixtureNormalCDF(c;weights = weights,means = means , std = sd) - 
                MixtureNormalCDF(0;weights = weights,means = means , std = sd)) /
                (-MixtureNormalCDF(0;weights = weights,means = means , std = sd) + 1)
    else
        return (MixtureNormalCDF(c;weights = weights,means = means , std = sd) - 
                MixtureNormalCDF(0;weights = weights,means = means , std = sd)) /
                MixtureNormalCDF(0;weights = weights,means = means , std = sd)
    end
end

"""----------------------------------------------------------------------
    function: Calculation of bet size statistice
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.142
----------------------------------------------------------------------"""
function betSizeProbability(probability, # probability that label take place 
                            nClasses) # number of label classes 
    # compute statistic
    statistic = (probability .- (1/nClasses)) ./ sqrt.(probability .* (1 .- probability))
    model = Normal(0, 1) # create Normal distribution instanse 
    return 2*cdf(model, statistic) .- 1
end

function betSizeForRange(nClasses)
    prob = range(0, stop=1, length=10000)
    return DataFrame(probability = betSizeProbability(probability, nClasses), index=prob)
end
"""----------------------------------------------------------------------
    function: Calculation of Average Active Signals 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.144
----------------------------------------------------------------------"""
function AverageActiveSignals(DataFrameofSignals # DataFrame that has signal
                                ) 

    setofbarrier = Set((dropmissing(DataFrameofSignals)).t1) # select ending point of each events 
    
    setofbarrier = union(setofbarrier, (DataFrameofSignals.Dates)) # select opening point of each events 
    index = []
    for i in setofbarrier
        append!(index, [i])
    end
  
    sort!(index) # sort date  
    out = mpDataFrameObj(AverageActiveSignalsMultiProcessing, :molecule, index; DataFrameofSignals=DataFrameofSignals) 
    # create dataframe object for parallel computing
    return out
end

"""----------------------------------------------------------------------
    function: Calculation of Average Active Signals for molecule
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.144
----------------------------------------------------------------------"""
function AverageActiveSignalsMultiProcessing(;DataFrameofSignals, # DataFrame that has signal
                             molecule) 
    signals  = copy(DataFrameofSignals) # filter by molecule
    output = DataFrame(date = molecule , signal = zeros(length(molecule))) # create dataframe 

    for (i,loc) in enumerate(molecule)
        sig_lessthan_loc = signals.Dates .<= loc # select events start before loc
        loc_lessthan_t1 = signals.t1 .> loc # select events ends after loc

        is_missing = ismissing.(signals.t1) # select missing barirer
     
        df0 = sig_lessthan_loc .& (loc_lessthan_t1 .| is_missing) # select events index that contain loc 
        act = signals.Dates[df0]     # select events that contain loc 
        if length(act) > 0
            output[i,:signal] = mean(signals.signal[df0])
        else
            output[i,:signal]  = 0
        end
    end
    output =sort!(output,[:date])  # sort dataframe by date 

    return output
end

"""----------------------------------------------------------------------
    function: Discretize Signals 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.145
----------------------------------------------------------------------"""

function DiscreteSignal(signal,  # dataframe that contain signals 
                        stepSize) # stepSize 
    signal = round.(signal ./ stepSize)*stepSize # Discretize 
    signal[signal .> 1] .= 1 # set all signal above 1 to 1
    signal[signal .< -1] .= -1 # set all signal below -1 to -1 
    return signal
end

"""
    function: Genetrate Signal 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.145
"""
function GenerateSignal(events, # DataFrame for events 
                        stepSize, # stepsize for Discretize 
                        EstimationResult, # DataFrame that contain probability and prediction
                        nClasses) # number of classes
    if size(prob)[1] == 0
        return 
    end
    probability = EstimationResult.probability # extract probability from dataframe
    predictions  = EstimationResult.prediction # extract prediction from dataframe
    signal = (probability .- 1.0./nClasses) ./sqrt.(probability*(1-probability)) # compute signal from probability
    model = Normal(0, 1)
    signal = predictions .* (2*cdf.(model, signal) .- 1) # recompute signal 

    # if events have side then multiply signal by events side 
    if "side" in names(events)
        signal = signal .* filter(row -> row[:Dates] in EstimationResult.date, events).side
    end

    FinalSignal = DataFrame(Date = EstimationResult.date, signal= signal) # create dataframe for final signal
    FinalSignal.t1  = events.t1 # set first columns of dataframe to t1 

    FinalSignal = AverageActiveSignals(FinalSignal) # compute average signal

    FinalSignal = DiscreteSignal(FinalSignal, stepSize) #Discretize signal 

    return FinalSignal
    
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




