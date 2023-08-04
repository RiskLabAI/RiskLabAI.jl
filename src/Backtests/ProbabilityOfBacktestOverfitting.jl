"""
    function: Selecting row from given prartion
    reference: -
    methodology: -
"""
function selectedRow(
        partition, # index of submatrix 
        subMatrixSize # sub matrix size 
)    

    array = [] # create array for storing indexes 
    for p in partition
        start = (p-1)*subMatrixSize + 1 # inital starting point of partition
        ends = p*subMatrixSize # initial ending point of partition
        append!(array,[start:ends]) # append this indexes to array 
    end

    return (vcat(array...))

end

"""
    function: Computing Probability Of BacktestOverfitting
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.156
"""
function ProbabilityOfBacktestOverfitting(
        matrixData, # matrix of T×N for T observation on N strategy 
        nPartions, # Number of Partition (must be even)
        metric;    # metric function for evaluate strategy
        RiskFreeReturn  = 0 # RiskFreeReturn for calculate sharp ratio
) 

    if nPartions % 2 == 1 #check number of partition is even or not 
        print("Number of Partition must be even") # if S is odd then stop algorithm and ask number of partition must be even 
        return
    end
    
    nObservation,nStrategy = size(matrixData) # inital number of observation and number of strategy
    exceed = nObservation % nPartions 
    nObservation  -= exceed # minus exceed from number of observation because we want nPartions|nObservation
    Matrix = matrixData[1:nObservation,:] #select first nObservation  of data
    subMatrixSize = Int(nObservation // nPartions ) # calculate size of each partition size 
    partition = collect(combinations(1:nPartions,Int(nPartions//2))) # compute all combinations of size nPartions/2
    pboNum = Threads.Atomic{Int}(0) # initial Atomic pboNum with zero
    ResultInThreads = [[] for i in 1:Threads.nthreads()]
    
    Threads.@threads for p in partition
        index = [i for i in 1:nObservation] # create array of all index 
        selectedrow = SelectedRow(p,subMatrixSize) # select row of this partition
        trainData = Matrix[selectedrow,:] #inital train data 
        evaluate = [metric(trainData[:,i],RiskFreeReturn) for i in 1:nStrategy] # evaluate strategy on train data 
        bestStrategy = sortperm(evaluate)[nStrategy] # calculate best strategy in train data and save its index 
        testindex = index .∉ Ref(selectedrow) #inital test data index
        testData = Matrix[testindex,:] # inital test data 
        evaluate = [metric(testData[:,i],RiskFreeReturn) for i in 1:nStrategy] # evaluate strategy on test data 
        testindex = findall(x -> x == bestStrategy,sortperm(evaluate))[1] # find plcae of bestStrategy in sorted performance on test data 
        wbar = Float64(testindex) / (nStrategy+1) # calculate wbar
        logit = log(wbar/(1-wbar)) # calculate logit 
        if logit <= 0.0 # check if logit is less than zero 
            Threads.atomic_add!(pboNum, 1) # if logit is less than zero it means that it is overfitting case 
        end
        append!(ResultInThreads[Threads.threadid()],logit)
    end

    return pboNum[] / length(partition),vcat(ResultInThreads...) # calculate  Probability Of Backtest Overfitting

end