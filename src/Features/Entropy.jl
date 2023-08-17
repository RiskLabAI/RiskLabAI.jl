module Entropy

using Base.Iterators: partition
using DataStructures: DefaultDict

export shannonEntropy, lempelZivEntropy, probabilityMassFunction, plugInEntropyEstimator, kontoyiannisEntropy, longestMatchLength

"""
    shannonEntropy(message::String)::Float64

Calculate Shannon Entropy.

# Arguments
- `message::String`: Input encoded message.

# Returns
- `::Float64`: Shannon Entropy score.
"""
function shannonEntropy(
    message::String
)::Float64

    charToCount = DefaultDict{Char, Int}(0)
    entropy = 0.0
    messageLength = length(message)

    for character in message
        charToCount[character] += 1
    end

    for count in values(charToCount)
        frequency = count / messageLength
        entropy -= frequency * log2(frequency)
    end

    return entropy
end

"""
    lempelZivEntropy(message::String)::Float64

Calculate Lempel-Ziv Entropy.

# Arguments
- `message::String`: Input encoded message.

# Returns
- `::Float64`: Lempel-Ziv Entropy score.
"""
function lempelZivEntropy(
    message::String
)::Float64

    i, library = 1, Set{String}([message[1] |> string])
    messageLength = length(message)

    while i < messageLength
        lastJValue = messageLength - 1
        for j in i:messageLength - 1
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
    probabilityMassFunction(message::String, approximateWordLength::Int)::Dict

Calculate probability mass function (PMF).

# Arguments
- `message::String`: Input encoded message.
- `approximateWordLength::Int`: Approximation of word length.

# Returns
- `::Dict`: Probability mass function.
"""
function probabilityMassFunction(
    message::String,
    approximateWordLength::Int
)::Dict

    library = DefaultDict{String, Vector{Int}}(Vector{Int})
    messageLength = length(message)

    for index in approximateWordLength:messageLength - 1
        message_ = message[index - approximateWordLength + 1:index]
        push!(library[message_], index - approximateWordLength + 1)
    end

    denominator = (messageLength - approximateWordLength) |> float
    pmf = Dict([key => length(library[key]) / denominator for key in keys(library)])
    return pmf
end

"""
    plugInEntropyEstimator(message::String, approximateWordLength::Int=1)::Float64

Calculate Plug-in Entropy Estimator.

# Arguments
- `message::String`: Input encoded message.
- `approximateWordLength::Int=1`: Approximation of word length.

# Returns
- `::Float64`: Plug-in Entropy Estimator score.
"""
function plugInEntropyEstimator(
    message::String,
    approximateWordLength::Int=1
)::Float64

    pmf = probabilityMassFunction(message, approximateWordLength)
    plugInEntropyEstimator = -sum([pmf[key] * log2(pmf[key]) for key in keys(pmf)]) / approximateWordLength
    return plugInEntropyEstimator
end

"""
    longestMatchLength(message::String, i::Int, n::Int)::Tuple{Int, String}

Calculate the length of the longest match.

# Arguments
- `message::String`: Input encoded message.
- `i::Int`: Starting index for the search.
- `n::Int`: Length of the expanding window.

# Returns
- `::Tuple{Int, String}`: Match length and matched substring.
"""
function longestMatchLength(
    message::String,
    i::Int,
    n::Int
)::Tuple{Int, String}

    subString = ""
    for l in 1:n
        message1 = message[i:i + l]
        for j in i - n + 1:i
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
    kontoyiannisEntropy(message::String, window::Int=0)::Float64

Calculate Kontoyiannis Entropy.

# Arguments
- `message::String`: Input encoded message.
- `window::Int=0`: Length of the expanding window.

# Returns
- `::Float64`: Kontoyiannis Entropy score.
"""
function kontoyiannisEntropy(
    message::String,
    window::Int=0
)::Float64

    output = Dict("num" => 0, "sum" => 0, "subString" => [])

    if window === 0
        points = 2:length(message) ÷ 2 + 1
    else
        window = min(window, length(message) ÷ 2)
        points = window + 1:length(message) - window + 1
    end

    for i in points
        if window === 0
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
