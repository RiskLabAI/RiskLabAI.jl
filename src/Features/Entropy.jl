module Entropy

export shannonEntropy, lempleZivEntropy, probabilityMassFunction, plugInEntropyEstimator, kontoyiannisEntorpy

"""
    function: Shannon Entropy 
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 263 SHANNON’S ENTROPY section
"""
function shannonEntropy(
    message::String # input encoded message
)::Float64

    charToCount = Dict()
    entropy = 0
    for character ∈ message
        try
            charToCount[character] += 1
        catch
            charToCount[character] = 1
        end
    end

    messageLength = length(message)
    for count ∈ values(charToCount)
        frequency = count / messageLength
        entropy -= frequency * log2(frequency)
    end

    entropy
end

"""
    function: A LIBRARY BUILT USING THE LZ ALGORITHM
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 266 LEMPEL-ZIV ESTIMATORS section
"""
function lempleZivEntropy(
    message::String # input encoded message
)::Float64

    i, library = 1, Set{String}([message[1] |> string])
    messageLength = length(message)
    while i < messageLength
        lastJValue = messageLength - 1
        for j ∈ i:messageLength-1
            message_ = message[i+1:j+1]
            if message_ ∉ library
                push!(library, message_)
                lastJValue = j
                break
            end
        end
        i = lastJValue + 1
    end


    length(library) / length(message)
end

"""
    function: Calculate probability mass function (PMF)
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 265 THE PLUG-IN (OR MAXIMUM LIKELIHOOD) ESTIMATOR section
"""
function probabilityMassFunction(
    message::String, # input encoded message
    approximateWordLength::Int # approximation of word length
)::Dict

    library = Dict()

    messageLength = length(message)
    for index ∈ approximateWordLength:messageLength-1
        message_ = message[index-approximateWordLength+1:index] 

        if message_ ∉ keys(library)
            library[message_] = [index - approximateWordLength + 1]
        else
            append!(library[message_], index - approximateWordLength + 1)
        end
    end

    denominator = (messageLength - approximateWordLength) |> float
    probabilityMassFunction = Dict([
        key => length(library[key]) / denominator for key ∈ keys(library)
    ])

    probabilityMassFunction
end

"""
    function: Plug-in Entropy Estimator Implementation
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 265 THE PLUG-IN (OR MAXIMUM LIKELIHOOD) ESTIMATOR section
"""
function plugInEntropyEstimator(
    message::String; # input encoded message
    approximateWordLength::Int=1 # approximation of word length
)::Float64

    pmf = probabilityMassFunction(message, approximateWordLength)
    plugInEntropyEstimator = -sum([pmf[key] * log2(pmf[key]) for key ∈ keys(pmf)]) / approximateWordLength
    plugInEntropyEstimator
end

"""
    function: COMPUTES THE LENGTH OF THE LONGEST MATCH
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 267 LEMPEL-ZIV ESTIMATORS section
"""
function longestMatchLength(
    message::String,
    i::Int,
    n::Int
)::Tuple{Int, String}

    subString = ""
    for l ∈ 1:n
        message1 = message[i:i+l]
        for j ∈ i-n+1:i
            message0 = message[j:j+l]
            if message1 == message0
                subString = message1
                break # search for higher l.
            end
        end
    end

    return (length(subString) + 1, subString) # matched length + 1 
end

"""
    function: IMPLEMENTATION OF ALGORITHMS DISCUSSED IN GAO ET AL.
    reference: De Prado, M. (18) Advances in Financial Machine Learning
    methodology: page 268 LEMPEL-ZIV ESTIMATORS section
"""
function kontoyiannisEntorpy(
    message::String; # input encoded message
    window::Int=0, # length of expanding window 
)::Float64

    output = Dict("num" => 0, "sum" => 0, "subString" => [])
    if window === nothing
        points = 2:length(message)÷2+1
    else
        window = min(window, length(message) ÷ 2)
        points = window+1:length(message)-window+1
    end

    for i ∈ points
        if window === nothing
            (l, message_) = longestMatchLength(message, i, i)
            output["sum"] += log2(i) / l # to avoid Doeblin condition
        else
            (l, message_) = longestMatchLength(message, i, window)
            output["sum"] += log2(window) / l # to avoid Doeblin condition
        end

        append!(output["subString"], message_)
        output["num"] += 1
    end

    output["h"] = output["sum"] / output["num"]
    output["r"] = 1 - output["h"] / log2(length(message)) 

    output["h"] 
end

end
