"""
Perform linear partitions for parallel processing.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.306
"""
function linear_partitions(
    num_atoms::Int,  # number of atoms in array
    num_threads::Int  # number of threads
)::Vector{Int}
    len = min(num_threads, num_atoms) + 1  # find minimum number of threads and atoms
    parts = collect(range(0, stop = num_atoms, length = len))  # find first & end elements of partitions
    parts = trunc.(Int, ceil.(parts))  # cascade first & end elements of partitions to Int
    return parts
end

"""
Perform nested partitions for parallel processing.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.308
"""
function nested_partitions(
    num_atoms::Int,  # number of atoms in array
    num_threads::Int,  # number of threads
    upper_triangular::Bool = false
)::Vector{Int}
    parts = [0]  # set beginning of first part to zero
    num_threads_ = min(num_threads, num_atoms)  # number of threads can't be higher than number of atoms
    for num in 1:num_threads_
        # calculate other parts beginning points
        part = 1 + 4 * (parts[end]^2 + parts[end] + num_atoms * (num_atoms + 1.0) / num_threads_)
        part = (-1 + sqrt(part)) / 2
        append!(parts, [part])  # add new beginning point to parts
    end
    parts = int(round.(parts))  # cascade beginning point to integer
    if upper_triangular  # check for upper triangular and if true reverse and redesign partitions
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
function mp_dataframe_obj(
    func,  # function that we like to parallelize its tasks
    df_obj1,  # name of molecule
    df_obj2,  # dataframe that we like result of func(df_obj2; kargs...)
    num_threads = 4,  # number of threads that set before running code
    mp_batches = 1,  # number of batches
    lin_mols = true,  # boolean for using linear partitions or nested partitions
    kwargs...  # other input arguments of func
)
    if lin_mols  # check for linear partitions or nested partitions
        parts = linear_partitions(size(df_obj2, 1), num_threads * mp_batches)
    else
        parts = nested_partitions(size(df_obj2, 1), num_threads * mp_batches)
    end

    jobs = []  # array of job for multi-threading

    for i in 1:length(parts) - 1
        job = Dict(df_obj1 => df_obj2[parts[i] + 1:parts[i + 1]], "func" => func, "index" => [j for j in parts[i] + 1:parts[i + 1]])  # create new job
        job = merge(job, Dict(kwargs...))  # add input arguments of func to job
        append!(jobs, [job])  # add this new job to array of jobs
    end

    out = process_jobs(jobs)  # process jobs with multi-threading

    return out
end

"""
Process jobs in parallel.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.312
"""
function process_jobs(jobs) # array of jobs
    dataframe_output = [DataFrame() for _ in 1:nthreads()]  # create arrays of dataframe that contain one dataframe for each thread
    @threads for job in jobs  # create loop for processing jobs
        out, index = expand_call(job)  # call expand_call function for processing job
        out.index = index  # create column with name out that containing index that this thread is processing
        dataframe_output[threadid()] = append!(dataframe_output[threadid()], out)  # append created dataframe to thread dataframe
    end

    final_output = DataFrame()  # create dataframe that our final result must be in
    for i in 1:nthreads()  # create loop for combining result of each thread
        final_output = append!(final_output, dataframe_output[i])  # append each thread's output to final_output
    end

    final_output = sort!(final_output, [:index])  # sort dataframe with respect to index column
    return select!(final_output, Not(:index))  # return dataframe without index column
end

"""
Expand call for a specific job.

Reference: De Prado, M. (2018) Advances in Financial Machine Learning
Methodology: p.312
"""
function expand_call(kargs)  # arguments of job
    func = kargs["func"]  # get function
    index = kargs["index"]  # get index that thread is running on it

    # Deleting func, index, and molecule for kargs
    delete!(kargs, "func")
    delete!(kargs, "index")

    return func(; kargs...), index  # return output of function and index
end
