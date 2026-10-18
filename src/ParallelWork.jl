module ParallelWork

using DataFrames

"""Zero-based boundary counts for contiguous groups with at most `num_threads` parts."""
function linear_partitions(num_atoms::Integer, num_threads::Integer)
    num_atoms >= 0 || throw(ArgumentError("num_atoms must be nonnegative"))
    num_threads > 0 || throw(ArgumentError("num_threads must be positive"))
    parts = min(num_atoms, num_threads)
    return parts == 0 ? [0] : [ceil(Int, index*num_atoms/parts) for index in 0:parts]
end

"""Triangular-work boundaries, optionally reversing group sizes for an upper triangle."""
function nested_partitions(num_atoms::Integer, num_threads::Integer; upper_triangle::Bool = false)
    num_atoms >= 0 || throw(ArgumentError("num_atoms must be nonnegative"))
    num_threads > 0 || throw(ArgumentError("num_threads must be positive"))
    parts = min(num_atoms, num_threads)
    points = [0.]
    for _ in 1:parts
        previous = last(points)
        push!(points, (sqrt(1+4*(previous^2+previous+num_atoms*(num_atoms+1.)/parts))-1)/2)
    end
    boundaries = round.(Int, points)
    return upper_triangle ? vcat(0, cumsum(reverse(diff(boundaries)))) : boundaries
end

function _workers(count)
    count == -1 && return Threads.nthreads()
    count > 0 || throw(ArgumentError("worker count must be positive or -1"))
    return min(Int(count), Threads.nthreads())
end

function _work_results(func, items; workers, ordered::Bool)
    values = collect(items)
    isempty(values) && return Any[]
    workers == 1 && return map(func, values)
    next_index = Threads.Atomic{Int}(1)
    completed = Channel{Tuple{Int,Any}}(length(values))
    @sync for _ in 1:min(workers, length(values))
        Threads.@spawn begin
            while true
                index = Threads.atomic_add!(next_index, 1)
                index > length(values) && break
                result = func(values[index])
                put!(completed, (index, result))
            end
        end
    end
    results = [take!(completed) for _ in eachindex(values)]
    ordered && sort!(results; by = first)
    return last.(results)
end

"""
Run independent calls with bounded Julia threads and return submitted order.
With `lin_partition=true`, the callback receives one-based position groups,
including empty groups when requested; returned groups are flattened. Threads
share memory, so callbacks must avoid conflicting writes and shared mutable RNGs.
"""
function parallel_run(func, iterable; num_cpus::Integer = -1,
    lin_partition::Bool = false, kwargs...)
    workers = _workers(num_cpus)
    values = collect(iterable)
    if !lin_partition
        return _work_results(value -> func(value; kwargs...), values; workers = workers, ordered = true)
    end
    parts = num_cpus == -1 ? Threads.nthreads() : Int(num_cpus)
    quotient, remainder = divrem(length(values), parts)
    sizes = [quotient+(index <= remainder) for index in 1:parts]
    boundaries = vcat(0, cumsum(sizes))
    groups = [collect((boundaries[index]+1):boundaries[index+1]) for index in 1:parts]
    results = _work_results(group -> func(group; kwargs...), groups; workers = workers, ordered = true)
    return reduce(vcat, results; init = Any[])
end

function _expand_job(job)
    options = (; (Symbol(key) => value for (key, value) in pairs(job) if Symbol(key) != :func)...)
    function_value = job isa NamedTuple ? job.func : haskey(job, :func) ? job[:func] : job["func"]
    return function_value(; options...)
end

"""Evaluate native function/keyword jobs in input order without mutating the jobs."""
process_jobs_sequential(jobs) = map(_expand_job, jobs)

"""
Evaluate jobs with bounded threads. The default collects completion order for
parallel execution; `ordered=true` returns input order. Sequential execution
always follows input order. Failures propagate instead of certifying partial work.
"""
function process_jobs(jobs; num_threads::Integer = -1, ordered::Bool = false)
    return _work_results(_expand_job, jobs; workers = _workers(num_threads), ordered = ordered)
end

"""
Partition caller indices, invoke `func(; molecule=group, ...)`, and combine
DataFrame results sorted by the explicit `index_column`. Non-table results stay
grouped in input order. This native interface replaces implicit pandas indices.
"""
function parallel_indexed(func, indices; argument::Symbol = :molecule,
    num_threads::Integer = -1, mp_batches::Integer = 1,
    linear_partition::Bool = true, index_column::Symbol = :index, kwargs...)
    workers = _workers(num_threads)
    mp_batches > 0 || throw(ArgumentError("mp_batches must be positive"))
    values = collect(indices)
    parts = (num_threads == -1 ? Threads.nthreads() : Int(num_threads))*mp_batches
    boundaries = linear_partition ? linear_partitions(length(values), parts) : nested_partitions(length(values), parts)
    groups = [values[(boundaries[index]+1):boundaries[index+1]] for index in 1:length(boundaries)-1]
    call = group -> func(; merge((; kwargs...), NamedTuple{(argument,)}((group,)))...)
    results = _work_results(call, groups; workers = workers, ordered = true)
    isempty(results) && return DataFrame(index_column => Any[])
    if all(result -> result isa AbstractDataFrame, results)
        combined = vcat(results...; cols = :setequal)
        index_column in propertynames(combined) || throw(ArgumentError("table result is missing the explicit index column"))
        return sort!(combined, index_column)
    end
    return results
end

export linear_partitions, nested_partitions, parallel_run, process_jobs,
    process_jobs_sequential, parallel_indexed

end
