
"""
    function: Linear Partitions 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.306
"""
function linParts(numAtoms, # number of atoms in array
                  numThreads) # number of threads

    len = min(numThreads, numAtoms)+1  # find minimum number of threads and atoms 
    parts = collect(range(0, stop = numAtoms, length = len)) #find frist & end elements of Partitions
    parts = trunc.(Int, ceil.(parts)) # cascade first & end elements of Partitions to Int

    return parts
end

"""
    function: Nested Partition
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.308
"""
function nestedParts(numAtoms,  # number of atoms in array
                     numThreads, # number of threads
                     upperTriang = false)

        parts = [0] # set beginnig of first part to zero 
        numThreads_ = min(numThreads, numAtoms) # number of threads can't be higher than number of atoms 
        for num in 1:numThreads_
            # calculate other parts beginnig points 

            part = 1 + 4*(parts[length(parts)]^2 + parts[end] + numAtoms*(numAtoms+1.0)/numThreads_)
            part = (-1 + sqrt(part))/2
            append!(parts, [part]) # add new begining point to parts 
        end
        parts = int(round(parts)) # cascade beginig point to integer 
        if upperTriang # chech for upperTriang and if true reverse and redesign Partitions
            parts = cumsum(reverse(diff(parts)))
            parts = append!([0] , parts)
        end
        parts
end

"""
    function: multi processed dataframes objects 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.310
"""
function mpDataFrameObj(func,# function that we like parallaized it's tasks
                        dfObj1, # name of molecule
                        dfObj2; # dataframe that we like result of func(dfObj2;kargs...)
                        numThreads = 4, # number of threads that set before running code
                        mpBatches = 1, # number of batchs
                        linMols = true, #boolian for use linear Partitions or nested Partitons
                        kwargs...) #other input arguments of func
    
    if linMols #check for linear Partition of nested Partition
        parts = linParts(size(dfObj2)[1], numThreads*mpBatches)
    else
        parts = nestedParts(size(dfObj2)[1], numThreads*mpBatches)
    end

    jobs = [] #arrays of job for multi threading 

    for i in 1:length(parts) - 1
        job = Dict(dfObj1 => dfObj2[parts[i] + 1:parts[i + 1]], "func" => func,"index" => [j for j in parts[i] + 1:parts[i +  1]]) #creat new job 
        job = merge(job, Dict(kwargs...)) #add input arguments of func to job
        append!(jobs, [job]) # add this new job for array of jobs 
    end

    out = ProcessJobs(jobs) #process jobs whith muliti threading
      
    return out
end

"""
    function: Process jobs 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.312
"""
function ProcessJobs(jobs #array of jobs
                    )
    dataframeOutput = [DataFrame() for _ in 1:nthreads()] #creat arrays of dataframe that contain one dataframe for each threads
    @threads for job in jobs #creat loop for processing jobs
        out, index = expandCall(job) # call expandCall function for processing job
        out.index  = index #creat column with name out that containg index that this thread processing them 
        dataframeOutput[threadid()] = append!(dataframeOutput[threadid()], out)     #append created dataframe to thread dataframe
    end

    finalOutput = DataFrame() #creat dataframe that our final result must be in it 
    for i in 1:nthreads() #creat loop for combining result of each thread
        finalOutput = append!(finalOutput, dataframeOutput[i]) # append each thread's output to finalOutput
    end

    final_output = sort!(finalOutput, [:index]) # sort dataframe respect to index column 
    return  select!(final_output, Not(:index)) # return dataframe whithout index column
end
    
"""
    function: expanding call for specific job 
    reference: De Prado, M. (2018) Advances in financial machine learning. John Wiley & Sons.
    methodology: p.312
"""
function expandCall(kargs #arguments of job
                   ) 
    func = kargs["func"] # get function 
    index = kargs["index"] # get index that thread running on it 

    # deleting func,index & molecule for kargs 
    delete!(kargs, "func")
    delete!(kargs, "index")

    return func(;kargs...), index # return output of function and index 
end