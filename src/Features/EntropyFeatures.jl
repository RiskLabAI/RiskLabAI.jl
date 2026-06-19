"""
Entropy features — native Julia port mirroring the Python
`RiskLabAI.features.entropy_features` API (López de Prado, AFML Ch. 18):
Shannon, plug-in, Lempel–Ziv and Kontoyiannis entropy estimators over a
discretised message string.

These are pure string/combinatorial estimators (entropy in bits, `log2`); the
values match the Python implementation exactly (verified in `test/runtests.jl`).

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 18.
"""

"""
    shannon_entropy(message) -> Float64

Shannon entropy (bits) of the character distribution of `message`
(`0.0` for an empty message). Mirrors Python's `shannon_entropy`.
"""
function shannon_entropy(message::AbstractString)
    isempty(message) && return 0.0
    counts = Dict{Char,Int}()
    for c in message
        counts[c] = get(counts, c, 0) + 1
    end
    n = length(message)
    return -sum((count / n) * log2(count / n) for count in values(counts))
end

"""
    probability_mass_function(message, approximate_word_length) -> Dict{String,Float64}

Probability mass function of the length-`approximate_word_length` n-grams of
`message`. Mirrors Python's `probability_mass_function`.
"""
function probability_mass_function(message::AbstractString, approximate_word_length::Integer)
    (isempty(message) || length(message) < approximate_word_length) &&
        return Dict{String,Float64}()
    chars = collect(message)
    n = length(chars)
    counts = Dict{String,Int}()
    for i = 1:(n-approximate_word_length+1)
        word = String(chars[i:(i+approximate_word_length-1)])
        counts[word] = get(counts, word, 0) + 1
    end
    num_windows = n - approximate_word_length + 1
    return Dict(word => count / num_windows for (word, count) in counts)
end

"""
    plug_in_entropy_estimator(message, approximate_word_length=1) -> Float64

Plug-in (maximum-likelihood) entropy: Shannon entropy of the n-gram PMF,
normalised by the word length. Mirrors Python's `plug_in_entropy_estimator`.
"""
function plug_in_entropy_estimator(message::AbstractString, approximate_word_length::Integer = 1)
    isempty(message) && return 0.0
    pmf = probability_mass_function(message, approximate_word_length)
    isempty(pmf) && return 0.0
    entropy = -sum(p * log2(p) for p in values(pmf) if p > 0)
    return entropy / approximate_word_length
end

"""
    lempel_ziv_entropy(message) -> Float64

Lempel–Ziv complexity (count of distinct substrings in a one-pass parse)
normalised by the message length. Mirrors Python's `lempel_ziv_entropy`.
"""
function lempel_ziv_entropy(message::AbstractString)
    isempty(message) && return 0.0
    chars = collect(message)
    n = length(chars)
    library = Set{String}()
    i = 0
    while i < n
        j = i
        while j < n && String(chars[(i+1):(j+1)]) in library
            j += 1
        end
        push!(library, String(chars[(i+1):min(j + 1, n)]))
        i = j + 1
    end
    return length(library) / n
end

"""
    longest_match_length(message, i, n) -> (Int, String)

Longest substring starting at 0-based index `i` that also occurs in the
preceding window `message[max(0, i-n):i]`. Returns the match length + 1 and the
matched substring. Mirrors Python's `longest_match_length`.
"""
function longest_match_length(message::AbstractString, i::Integer, n::Integer)
    chars = collect(message)
    msglen = length(chars)
    longest_match = ""
    for len = 1:n
        i + len > msglen && break
        pattern = String(chars[(i+1):(i+len)])
        found = false
        for j = max(0, i - n):(i-1)
            if pattern == String(chars[(j+1):(j+len)])
                longest_match = pattern
                found = true
                break
            end
        end
        found || break
    end
    return (length(longest_match) + 1, longest_match)
end

"""
    kontoyiannis_entropy(message; window=nothing) -> Float64

Kontoyiannis LZ-based entropy estimator. With `window === nothing` an expanding
look-back is used (`nᵢ = i`); otherwise a rolling window of size `window`.
Mirrors Python's `kontoyiannis_entropy`.
"""
function kontoyiannis_entropy(message::AbstractString; window::Union{Nothing,Integer} = nothing)
    msglen = length(message)
    sum_h = 0.0
    num_points = 0
    if window === nothing
        points = 2:(msglen-1)
    else
        window = min(window, msglen - 1)
        points = window:(msglen-1)
    end
    isempty(points) && return 0.0
    for i in points
        n = window === nothing ? i : window
        n == 0 && continue
        l_i, _ = longest_match_length(message, i, n)
        sum_h += log2(n) / l_i
        num_points += 1
    end
    num_points == 0 && return 0.0
    return sum_h / num_points
end
