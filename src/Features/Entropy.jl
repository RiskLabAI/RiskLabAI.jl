module Entropy

export shannonEntropy, lempleZivEntropy, probabilityMassFunction, plugInEntropyEstimator, kontoyiannisEntropy, longestMatchLength

"""
    Calculate Shannon Entropy

    Parameters:
    - message::String: Input encoded message.

    Returns:
    - entropy::Float64: Shannon Entropy score.
"""
function shannonEntropy(message::String)::Float64
    charToCount = Dict()
    entropy = 0.0
    messageLength = length(message)

    for character ∈ message
        try
            charToCount[character] += 1
        catch
            charToCount[character] = 1
        end
    end

    for count ∈ values(charToCount)
        frequency = count / messageLength
        entropy -= frequency * log2(frequency)
    end

    return entropy
end

"""
    Calculate Lemple-Ziv Entropy

    Parameters:
    - message::String: Input encoded message.

    Returns:
    - entropy::Float64: Lemple-Ziv Entropy score.
"""
function lempleZivEntropy(message::String)::Float64
    i, library = 1, Set{String}([message[1] |> string])
    messageLength = length(message)

    while i < messageLength
        lastJValue = messageLength - 1
        for j ∈ i:messageLength - 1
            message_ = message[i + 1:j + 1]
            if message_ ∉ library
                push!(library, message_)
                lastJValue = j
                break
            end
        end
        i = lastJValue + 1
    end

    return length(library) / messageLength
end

"""
    Calculate probability mass function (PMF)

    Parameters:
    - message::String: Input encoded message.
    - approximateWordLength::Int: Approximation of word length.

    Returns:
    - pmf::Dict: Probability mass function.
"""
function probabilityMassFunction(message::String, approximateWordLength::Int)::Dict
    library = Dict()
    messageLength = length(message)

    for index ∈ approximateWordLength:messageLength - 1
        message_ = message[index - approximateWordLength + 1:index]

        if message_ ∉ keys(library)
            library[message_] = [index - approximateWordLength + 1]
        else
            append!(library[message_], index - approximateWordLength + 1)
        end
    end

    denominator = (messageLength - approximateWordLength) |> float
    pmf = Dict([key => length(library[key]) / denominator for key ∈ keys(library)])
    return pmf
end

"""
    Calculate Plug-in Entropy Estimator

    Parameters:
    - message::String: Input encoded message.
    - approximateWordLength::Int: Approximation of word length.

    Returns:
    - entropy::Float64: Plug-in Entropy Estimator score.
"""
function plugInEntropyEstimator(message::String, approximateWordLength::Int=1)::Float64
    pmf = probabilityMassFunction(message, approximateWordLength)
    plugInEntropyEstimator = -sum([pmf[key] * log2(pmf[key]) for key ∈ keys(pmf)]) / approximateWordLength
    return plugInEntropyEstimator
end

"""
    Calculate the length of the longest match

    Parameters:
    - message::String: Input encoded message.
    - i::Int: Starting index for the search.
    - n::Int: Length of the expanding window.

    Returns:
    - matchLength::Tuple{Int, String}: Match length and matched substring.
"""
function longestMatchLength(message::String, i::Int, n::Int)::Tuple{Int, String}
    subString = ""
    for l ∈ 1:n
        message1 = message[i:i + l]
        for j ∈ i - n + 1:i
            message0 = message[j:j + l]
            if message1 == message0
                subString = message1
                break  # search for higher l.
            end
        end
    end

    return (length(subString) + 1, subString)  # matched length + 1
end

"""
    Calculate Kontoyiannis Entropy

    Parameters:
    - message::String: Input encoded message.
    - window::Int: Length of the expanding window.

    Returns:
    - entropy::Float64: Kontoyiannis Entropy score.
"""
function kontoyiannisEntropy(message::String, window::Int=0)::Float64
    output = Dict("num" => 0, "sum" => 0, "subString" => [])

    if window === nothing
        points = 2:length(message) ÷ 2 + 1
    else
        window = min(window, length(message) ÷ 2)
        points = window + 1:length(message) - window + 1
    end

    for i ∈ points
        if window === nothing
            (l, message_) = longestMatchLength(message, i, i)
            output["sum"] += log2(i) / l  # to avoid Doeblin condition
        else
            (l, message_) = longestMatchLength(message, i, window)
            output["sum"] += log2(window) / l  # to avoid Doeblin condition
        end

        append!(output["subString"], message_)
        output["num"] += 1
    end

    output["h"] = output["sum"] / output["num"]
    output["r"] = 1 - output["h"] / log2(length(message))
    return output["h"]
end

end
