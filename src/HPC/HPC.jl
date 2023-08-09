include("PurgedKFoldCV.jl")

"""
Perform linear partitions for parallel processing.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.306
"""
function linearPartitions(
    numAtoms::Int,  # number of atoms in array
    numThreads::Int  # number of threads
)::Vector{Int}
    len = min(numThreads, numAtoms) + 1  # find minimum number of threads and atoms
    parts = collect(range(0, stop = numAtoms, length = len))  # find first & end elements of partitions
    parts = trunc.(Int, ceil.(parts))  # cascade first & end elements of partitions to Int
    return parts
end

"""
Perform nested partitions for parallel processing.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.308
"""
function nestedPartitions(
    numAtoms::Int,  # number of atoms in array
    numThreads::Int,  # number of threads
    upperTriangular::Bool = false
)::Vector{Int}
    parts = [0]  # set beginning of first part to zero
    numThreads_ = min(numThreads, numAtoms)  # number of threads can't be higher than number of atoms
    for num in 1:numThreads_
        # calculate other parts beginning points
        part = 1 + 4 * (parts[end]^2 + parts[end] + numAtoms * (numAtoms + 1.0) / numThreads_)
        part = (-1 + sqrt(part)) / 2
        append!(parts, [part])  # add new beginning point to parts
    end
    parts = int(round.(parts))  # cascade beginning point to integer
    if upperTriangular  # check for upper triangular and if true reverse and redesign partitions
        parts = cumsum(reverse(diff(parts)))
        parts = append!([0], parts)
    end
    return parts
end

"""
Perform multi-processed dataframe objects.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.310
"""
function mpDataFrameObj(
    func,  # function that we like to parallelize its tasks
    dfObj1,  # name of molecule
    dfObj2,  # dataframe that we like result of func(dfObj2; kargs...)
    numThreads = 4,  # number of threads that set before running code
    mpBatches = 1,  # number of batches
    linMols = true,  # boolean for using linear partitions or nested partitions
    kwargs...  # other input arguments of func
)
    if linMols  # check for linear partitions or nested partitions
        parts = linearPartitions(size(dfObj2, 1), numThreads * mpBatches)
    else
        parts = nestedPartitions(size(dfObj2, 1), numThreads * mpBatches)
    end

    jobs = []  # array of job for multi-threading

    for i in 1:length(parts) - 1
        job = Dict(dfObj1 => dfObj2[parts[i] + 1:parts[i + 1]], "func" => func, "index" => [j for j in parts[i] + 1:parts[i + 1]])  # create new job
        job = merge(job, Dict(kwargs...))  # add input arguments of func to job
        append!(jobs, [job])  # add this new job to array of jobs
    end

    out = processJobs(jobs)  # process jobs with multi-threading

    return out
end

"""
Process jobs in parallel.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.312
"""
function processJobs(jobs) # array of jobs
    dataframeOutput = [DataFrame() for _ in 1:nthreads()]  # create arrays of dataframe that contain one dataframe for each thread
    @threads for job in jobs  # create loop for processing jobs
        out, index = expandCall(job)  # call expandCall function for processing job
        out.index = index  # create column with name out that containing index that this thread is processing
        dataframeOutput[threadid()] = append!(dataframeOutput[threadid()], out)  # append created dataframe to thread dataframe
    end

    finalOutput = DataFrame()  # create dataframe that our final result must be in
    for i in 1:nthreads()  # create loop for combining result of each thread
        finalOutput = append!(finalOutput, dataframeOutput[i])  # append each thread's output to finalOutput
    end

    finalOutput = sort!(finalOutput, [:index])  # sort dataframe with respect to index column
    return select!(finalOutput, Not(:index))  # return dataframe without index column
end

"""
Expand call for a specific job.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.312
"""
function expandCall(kargs)  # arguments of job
    func = kargs["func"]  # get function
    index = kargs["index"]  # get index that thread is running on it

    # Deleting func, index, and molecule for kargs
    delete!(kargs, "func")
    delete!(kargs, "index")

    return func(; kargs...), index  # return output of function and index
end
