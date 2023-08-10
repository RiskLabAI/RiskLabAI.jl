"""
Selects rows from a given partition.

:param partition: Index of submatrix.
:param subMatrixSize: Submatrix size.
:return: Array of selected indexes.
"""
function selectedRow(partition, subMatrixSize)
    array = []
    
    for p in partition
        start = (p - 1) * subMatrixSize + 1
        ends = p * subMatrixSize
        append!(array, [start:ends])
    end
    
    return vcat(array...)
end

"""
Computes the Probability Of Backtest Overfitting.

:param matrixData: Matrix of T×N for T observations on N strategies.
:param nPartitions: Number of partitions (must be even).
:param metric: Metric function for evaluating strategy.
:param riskFreeReturn: Risk-free return for calculating Sharpe ratio.
:return: Tuple containing Probability Of Backtest Overfitting and an array of logit values.
"""
function probabilityOfBacktestOverfitting(
        matrixData,
        nPartitions,
        metric;
        riskFreeReturn=0.0
    )
    if nPartitions % 2 == 1
        println("Number of partitions must be even")
        return
    end
    
    nObservation, nStrategy = size(matrixData)
    exceed = nObservation % nPartitions
    nObservation -= exceed
    matrix = matrixData[1:nObservation, :]
    subMatrixSize = Int(nObservation // nPartitions)
    partition = collect(combinations(1:nPartitions, Int(nPartitions / 2)))
    pboNum = Threads.Atomic{Int}(0)
    resultInThreads = [[] for _ in 1:Threads.nthreads()]
    
    Threads.@threads for p in partition
        index = 1:nObservation
        selectedRowIndices = selectedRow(p, subMatrixSize)
        trainData = matrix[selectedRowIndices, :]
        evaluate = [metric(trainData[:, i], riskFreeReturn) for i in 1:nStrategy]
        bestStrategy = argmax(evaluate)
        testIndex = .! in(selectedRowIndices)
        testData = matrix[testIndex, :]
        evaluate = [metric(testData[:, i], riskFreeReturn) for i in 1:nStrategy]
        testIndex = findfirst(x -> x == bestStrategy, argmax(evaluate))
        wbar = Float64(testIndex) / (nStrategy + 1)
        logit = log(wbar / (1 - wbar))
        
        if logit <= 0.0
            Threads.atomic_add!(pboNum, 1)
        end
        
        push!(resultInThreads[Threads.threadid()], logit)
    end
    
    return pboNum[] / length(partition), vcat(resultInThreads...)
end
