"""
    function: expand label tO incorporate meta-labeling
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 51
"""
function concurrencyEvents(
    closeIndex, # DataFrame that has events
    timestamp, # DateFrame that has return and label of each period
    molecule
) # index that function must apply on it

    eventsfiltered = filter(row -> row[:date] in molecule, timestamp) # filter events respect to molecule
    starttime = eventsfiltered.date[1]
    endtime = maximum(eventsfiltered.timestamp)
    concurrencyindex = closeIndex[(closeIndex .>= starttime) .& (closeIndex .<= endtime)]
    concurrency = DataFrame(date=concurrencyindex , concurrency = zeros(size(concurrencyindex)[1])) 
    # create dataframe that contain number of concurrent label for each events 

    for (i, idx) in enumerate(eventsfiltered.date)
         
        startIndex  = concurrency.date .>= idx # store events that before idx
        endIndex = concurrency.date .<= eventsfiltered.timestamp[i] # store events that end after idx
        selectedIndex = startIndex .& endIndex
        concurrency.concurrency[selectedIndex] .+= 1
    end
    return concurrency
end

"""
    function: sampleWeight with triple barrier
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 51
"""
function sampleWeight(
    timestamp, # DataFrame of events start and end for labelling 
    concurrencyEvents, # Data frame of concurrent events for each events 
    molecule
) # index that function must apply on it

    eventsfiltered = filter(row -> row[:date] in molecule,timestamp) # filter timestamp by molecule
    weight = DataFrame(date = molecule,weight = zeros(length(molecule))) # create DataFrame for result 
    for i in 1:size(weight)[1]
        starttime, endtime = eventsfiltered.date[i] , eventsfiltered.timestamp[i] #initial starting Time and ending time 
        concurrencyEventsForSpecificTime = concurrencyEvents[(concurrencyEvents.date .>= starttime) .& (concurrencyEvents.date .<= endtime),"concurrency"] # compute concurrency for this time horizon
        weight[i,"weight"] = mean(1 ./ concurrencyEventsForSpecificTime) #compute weight
    end
    return weight
end
"""
    function: Creating Index matrix 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 51
"""
function indexMatrix(
    barIndex, #index of all data 
    timestamp
) # times of events contain starting and ending time 

    indexMatrix = zeros((size(barIndex)[1] , size(timestamp)[1])) #creat evetnt matrix that show index is time horizon or not 
    for (j, (t0,t1)) in enumerate(timestamp)
        Indicator = [(i<=t1 && i>= t0 ) ? 1 : 0 for i in barIndex] #if index is in events put 1 in its entry 
        indexMatrix[!,j] = Indicator
    end
    return indexMatrix
end

"""
    function: compute average uniqueness
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 51
"""
function averageUniqueness(indexMatrix) #matrix that Indicator for events 
    concurrency = sum(indexMatrix,dims = 2) #compute concurrency for each evnents 
    uniqueness = copy(indexMatrix) 
    for i in 1:size(indexMatrix)[1]
        if concurrency[i] > 0
            uniqueness[i,:] = uniqueness[i,:] /concurrency[i] #comput uniqueness
        end
    end 
    averageUniqueness_ = zeros(size(indexMatrix)[2])
    for i in 1:size(indexMatrix)[2]
        averageUniqueness_[i] = sum(uniqueness[:,i]) / sum(indexMatrix[:,i]) # compute average uniqueness
    end
    return averageUniqueness_
end

"""
    function:  sequentialBootstrap implementation 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 51
"""
function sequentialBootstrap(
    indexMatrix, #matrix that Indicator for events 
                            SampleLength) # number of sample 
    if isnan(SampleLength) # check SampleLength is nan or no 
        SampleLength = size(indexMatrix)[2] # if SampleLength is nan initial it with number of columns of indexMatrix(number of events )
    end
    ϕ = []
    while length(ϕ) < SampleLength # do this loop untill number of sample is less than sample length
        averageUniqueness_ = zeros(size(indexMatrix)[2]) 
        for i in 1:size(indexMatrix)[2]
            tempIndexMatrix = indexMatrix[:,vcat(ϕ,i)]
            averageUniqueness_[i] = averageUniqueness(tempIndexMatrix)[end] # compute averageUniqueness for indexMatrix with selected evnets 
        end
        probability = averageUniqueness_ / sum(averageUniqueness_) # compute probability of each events for sampleing 
        append!(ϕ , sample(1:size(indexMatrix)[2] , Weights(probability))) #sample one evnets with computed probability 
    end
    return ϕ
end

"""
    function:  sample weight with returns 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 51
"""
function sampleWeight(timestamp, # dataframe for events 
                      concurrencyEvents,# dataframe that contain number of concurrent evnets for each evnetst 
                      returns,#data frame that contains returns
                      molecule) # molecule
    eventsfiltered = filter(row -> row[:date] in molecule,timestamp) #filter timestamp with molecule
    weight = DataFrame(date = molecule,weight = zeros(length(molecule))) #create weight dataframe for results 
    priceReturn = copy(returns)
    priceReturn.returns = log.(returns.returns .+ 1) # compute log returns
    for i in 1:size(weight)[1]
        starttime, endtime = eventsfiltered.date[i] , eventsfiltered.timestamp[i] #initial statring time and endtime time of event 
        concurrencyEventsForSpecificTime = concurrencyEvents[(concurrencyEvents.date .>= starttime) .& (concurrencyEvents.date .<= endtime),:] # compute concurrency for event
        returnForSpecificTime = priceReturn[(priceReturn.date .>= starttime) .& (priceReturn.date .<= endtime),:returns] #select subrow of return that occure in this event 
        weight.weight[i] = sum(returnForSpecificTime ./ concurrencyEventsForSpecificTime.concurrency) #comput weight
    end
    weight.weight = abs.(weight.weight) #weight must be positive !
    return weight
end

"""
    function:  compute timeDecay
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 51
"""
function timeDecay(weight; #weight taht compute for each event 
                   clfLastW = 1.0) #weight of oldest observation 
    timedecay = sort(weight,[:date]) # sort weight with date index 
    timedecay[!, :timedecay] = cumsum(timedecay[!, :weight]) # cumpute cumulitive sum 
    slope = 0.0 
    if clfLastW >= 0 #compute slope based on oldest observation weight
        slope  = (1-clfLastW) / timedecay.timedecay[end] 
    else
        slope = 1.0 /((clfLastW + 1)*timedecay.timedecay[end])
    end
    constant = 1.0 - slope*timedecay.timedecay[end] # compute b in y =ax + b 
    timedecay.timedecay =  slope .* timedecay.timedecay .+ constant # y = ax + b 
    timedecay.timedecay[timedecay.timedecay .< 0] .= 0 #set all timedecay below zero to zero
    timedecay = select!(timedecay, Not(:weight)) #select date and timedecay columns 
    return timedecay
end



