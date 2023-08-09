module Entropy

export shannon_entropy, lemple_ziv_entropy, probability_mass_function, plug_in_entropy_estimator, kontoyiannis_entropy

"""
    Calculate Shannon Entropy

    Parameters:
    - message::String: Input encoded message.

    Returns:
    - entropy::Float64: Shannon Entropy score.
"""
function shannon_entropy(message::String)::Float64
    char_to_count = Dict()
    entropy = 0.0
    message_length = length(message)

    for character ∈ message
        try
            char_to_count[character] += 1
        catch
            char_to_count[character] = 1
        end
    end

    for count ∈ values(char_to_count)
        frequency = count / message_length
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
function lemple_ziv_entropy(message::String)::Float64
    i, library = 1, Set{String}([message[1] |> string])
    message_length = length(message)

    while i < message_length
        last_j_value = message_length - 1
        for j ∈ i:message_length - 1
            message_ = message[i + 1:j + 1]
            if message_ ∉ library
                push!(library, message_)
                last_j_value = j
                break
            end
        end
        i = last_j_value + 1
    end

    return length(library) / message_length
end

"""
    Calculate probability mass function (PMF)

    Parameters:
    - message::String: Input encoded message.
    - approximate_word_length::Int: Approximation of word length.

    Returns:
    - pmf::Dict: Probability mass function.
"""
function probability_mass_function(message::String, approximate_word_length::Int)::Dict
    library = Dict()
    message_length = length(message)

    for index ∈ approximate_word_length:message_length - 1
        message_ = message[index - approximate_word_length + 1:index]

        if message_ ∉ keys(library)
            library[message_] = [index - approximate_word_length + 1]
        else
            append!(library[message_], index - approximate_word_length + 1)
        end
    end

    denominator = (message_length - approximate_word_length) |> float
    pmf = Dict([key => length(library[key]) / denominator for key ∈ keys(library)])
    return pmf
end

"""
    Calculate Plug-in Entropy Estimator

    Parameters:
    - message::String: Input encoded message.
    - approximate_word_length::Int: Approximation of word length.

    Returns:
    - entropy::Float64: Plug-in Entropy Estimator score.
"""
function plug_in_entropy_estimator(message::String, approximate_word_length::Int=1)::Float64
    pmf = probability_mass_function(message, approximate_word_length)
    plug_in_entropy_estimator = -sum([pmf[key] * log2(pmf[key]) for key ∈ keys(pmf)]) / approximate_word_length
    return plug_in_entropy_estimator
end

"""
    Calculate the length of the longest match

    Parameters:
    - message::String: Input encoded message.
    - i::Int: Starting index for the search.
    - n::Int: Length of the expanding window.

    Returns:
    - match_length::Tuple{Int, String}: Match length and matched substring.
"""
function longest_match_length(message::String, i::Int, n::Int)::Tuple{Int, String}
    sub_string = ""
    for l ∈ 1:n
        message1 = message[i:i + l]
        for j ∈ i - n + 1:i
            message0 = message[j:j + l]
            if message1 == message0
                sub_string = message1
                break  # search for higher l.
            end
        end
    end

    return (length(sub_string) + 1, sub_string)  # matched length + 1
end

"""
    Calculate Kontoyiannis Entropy

    Parameters:
    - message::String: Input encoded message.
    - window::Int: Length of the expanding window.

    Returns:
    - entropy::Float64: Kontoyiannis Entropy score.
"""
function kontoyiannis_entropy(message::String, window::Int=0)::Float64
    output = Dict("num" => 0, "sum" => 0, "subString" => [])

    if window === nothing
        points = 2:length(message) ÷ 2 + 1
    else
        window = min(window, length(message) ÷ 2)
        points = window + 1:length(message) - window + 1
    end

    for i ∈ points
        if window === nothing
            (l, message_) = longest_match_length(message, i, i)
            output["sum"] += log2(i) / l  # to avoid Doeblin condition
        else
            (l, message_) = longest_match_length(message, i, window)
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
