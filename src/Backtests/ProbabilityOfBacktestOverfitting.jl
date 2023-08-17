using Combinatorics
using Base.Threads
using Statistics

"""
    selectRows(partition::Vector{Int}, subMatrixSize::Int) -> Vector{Int}

Selects rows from a given partition.

# Arguments
- `partition::Vector{Int}`: Index of submatrix.
- `subMatrixSize::Int`: Submatrix size.

# Returns
- `Vector{Int}`: Array of selected indexes.

"""
function selectRows(partition::Vector{Int}, subMatrixSize::Int) -> Vector{Int}
    array = [((p - 1) * subMatrixSize + 1):(p * subMatrixSize) for p in partition]
    return vcat(array...)
end

"""
    probabilityOfBacktestOverfitting(
        matrixData::Matrix{Float64},
        nPartitions::Int,
        metric::Function;
        riskFreeReturn::Float64 = 0.0
    ) -> Tuple{Float64, Vector{Float64}}

Computes the Probability Of Backtest Overfitting.

# Arguments
- `matrixData::Matrix{Float64}`: Matrix of T×N for T observations on N strategies.
- `nPartitions::Int`: Number of partitions (must be even).
- `metric::Function`: Metric function for evaluating strategy.
- `riskFreeReturn::Float64`: Risk-free return for calculating Sharpe ratio.

# Returns
- `Tuple{Float64, Vector{Float64}}`: Tuple containing Probability Of Backtest Overfitting and an array of logit values.

"""
function probabilityOfBacktestOverfitting(
    matrixData::Matrix{Float64},
    nPartitions::Int,
    metric::Function;
    riskFreeReturn::Float64 = 0.0
) -> Tuple{Float64, Vector{Float64}}
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
        selectedRowIndices = selectRows(p, subMatrixSize)
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
