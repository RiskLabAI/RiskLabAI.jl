using DataFrames

"""
    linearPartitions(numAtoms::Int, numThreads::Int)::Vector{Int}

Perform linear partitions for parallel processing.

# Arguments
- `numAtoms::Int`: Number of atoms in array.
- `numThreads::Int`: Number of threads.

# Returns
- `Vector{Int}`: The array of partition indices.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.306
"""
function linearPartitions(
    numAtoms::Int,
    numThreads::Int
)::Vector{Int}
    len = min(numThreads, numAtoms) + 1
    parts = range(0, stop=numAtoms, length=len) |> ceil |> trunc.(Int)
    return parts
end

"""
    nestedPartitions(numAtoms::Int, numThreads::Int, upperTriangular::Bool = false)::Vector{Int}

Perform nested partitions for parallel processing.

# Arguments
- `numAtoms::Int`: Number of atoms in array.
- `numThreads::Int`: Number of threads.
- `upperTriangular::Bool`: If true, reverse and redesign partitions.

# Returns
- `Vector{Int}`: The array of partition indices.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.308
"""
function nestedPartitions(
    numAtoms::Int,
    numThreads::Int,
    upperTriangular::Bool = false
)::Vector{Int}
    parts = [0]
    numThreads_ = min(numThreads, numAtoms)
    for num in 1:numThreads_
        part = 1 + 4 * (parts[end]^2 + parts[end] + numAtoms * (numAtoms + 1.0) / numThreads_)
        part = (-1 + sqrt(part)) / 2
        push!(parts, part)
    end
    parts = round.(Int, parts)
    if upperTriangular
        parts = cumsum(reverse(diff(parts)))
        pushfirst!(parts, 0)
    end
    return parts
end

"""
    mpDataFrameObj(func, dfObj1, dfObj2, numThreads=4, mpBatches=1, linMols=true, kwargs...)

Perform multi-processed dataframe objects.

# Arguments
- `func`: Function to be parallelized.
- `dfObj1`: Name of the molecule.
- `dfObj2`: Dataframe that we like the result of func(dfObj2; kargs...).
- `numThreads::Int`: Number of threads that set before running code.
- `mpBatches::Int`: Number of batches.
- `linMols::Bool`: Boolean for using linear partitions or nested partitions.
- `kwargs...`: Other input arguments of func.

# Returns
- `DataFrame`: Processed dataframe.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.310
"""
function mpDataFrameObj(
    func,
    dfObj1,
    dfObj2,
    numThreads::Int = 4,
    mpBatches::Int = 1,
    linMols::Bool = true;
    kwargs...
)
    if linMols
        parts = linearPartitions(size(dfObj2, 1), numThreads * mpBatches)
    else
        parts = nestedPartitions(size(dfObj2, 1), numThreads * mpBatches)
    end

    jobs = [merge(Dict(dfObj1 => dfObj2[parts[i] + 1:parts[i + 1]], "func" => func, "index" => parts[i] + 1:parts[i + 1]), Dict(kwargs...)) for i in 1:length(parts) - 1]

    return processJobs(jobs)
end

"""
    processJobs(jobs::Vector{Dict})

Process jobs in parallel.

# Arguments
- `jobs::Vector{Dict}`: Array of jobs.

# Returns
- `DataFrame`: Processed dataframe.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.312
"""
function processJobs(jobs::Vector{Dict})
    dataframeOutput = [DataFrame() for _ in 1:nthreads()]
    @threads for job in jobs
        out, index = expandCall(job)
        out.index = index
        dataframeOutput[threadid()] = append!(dataframeOutput[threadid()], out)
    end

    finalOutput = DataFrame()
    for i in 1:nthreads()
        finalOutput = append!(finalOutput, dataframeOutput[i])
    end

    sort!(finalOutput, [:index])
    return select!(finalOutput, Not(:index))
end

"""
    expandCall(kargs::Dict)

Expand call for a specific job.

# Arguments
- `kargs::Dict`: Arguments of job.

# Returns
- `DataFrame`: Processed dataframe.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.312
"""
function expandCall(kargs::Dict)
    func = kargs["func"]
    index = kargs["index"]
    delete!(kargs, "func")
    delete!(kargs, "index")
    return func(; kargs...), index
end
