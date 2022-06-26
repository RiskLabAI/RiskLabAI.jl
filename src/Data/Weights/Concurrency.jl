"""----------------------------------------------------------------------
    function: expand label tO incorporate meta-labeling
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: 51
----------------------------------------------------------------------"""
function nConcurrentEventsMultiProcess(events, # time events
                                       output, #
                                       molecule) # 

                                       eventsfiltered = filter(row -> row[:date] in molecule, events) # filter events respect to molecule

    concurrency = DataFrame(Dates=events.date,active_long = zeros(size(eventsfiltered)[1]), active_short =  zeros(size(eventsfiltered)[1])) 
      # create dataframe that contain number of concurrent label for each events 

    for (i, idx) in enumerate(eventsfiltered.date)
         
        indexlessthanidx = events.date .<= idx # store events that before idx
        eventsmorethanidx = events.timestamp .> idx # store events that end after idx
        outispositive = out.ret .>= 0  # store events have positive returns

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

        cond = indexlessthanidx .& eventsmorethanidx .& outispositive 
          # compute intersection of arrays for selecting concurrent lebeled period that contains idx

        mySet = eventsfiltered.date[cond] # select data has our condition 

        dflongactiveidx = Set(mySet) #convert myset to set for eliminate redundant elements

        concurrency.active_long[i] = length(dflongactiveidx) #set active_long of idx to length of dflongactiveidx


        # we repeate this procedure for negative return same as postive returns
        outisnegative = out.ret .< 0
        maximumlength = maximum([length(outisnegative), length(eventsmorethanidx), length(indexlessthanidx)])
        if length(outisnegative) < maximumlength
            append!(outisnegative, Bool.(zeros(maximumlength-length(outisnegative))))
        end
        cond = indexlessthanidx .& eventsmorethanidx .& outisnegative
        mySet = eventsfiltered.date[cond]
        dfshortactiveidx = Set(mySet)
        concurrency.active_short[i] = length(dfshortactiveidx)

    end

    concurrency.ct = concurrency.active_long - concurrency.active_short 
          # compute differences of active_long and active_short for each events
    return concurrency
end
