"""
Entropy features — native Julia port mirroring the Python
`RiskLabAI.features.entropy_features` API (López de Prado, AFML Ch. 18):
Shannon, plug-in, Lempel–Ziv and Kontoyiannis entropy estimators over a
discretised message string.

These are pure string/combinatorial estimators (entropy in bits, `log2`); the
values match the Python implementation exactly (verified in `test/runtests.jl`).

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 18.
The bias-corrected family (Miller–Madow, Grassberger, NSB) is appended below.
"""

using SpecialFunctions: digamma, loggamma

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

# --------------------------------------------------------------------------- #
# Bias-corrected Shannon entropy: Miller–Madow, Grassberger, NSB.
#
# The plug-in / maximum-likelihood entropy underestimates entropy when the
# alphabet K (number of possible n-grams) is large relative to the sample N,
# because unobserved symbols contribute zero (bias ≈ −(K−1)/(2N)). These add the
# missing mass back: Miller–Madow to first order, Grassberger to higher order,
# NSB by integrating over a near-uniform-entropy Bayesian prior. All operate on
# the same overlapping n-gram counts as the plug-in and return bits / word length.
# Admitted in Appraisal 06 (`library_extension/appraisals/06_verdict.md`).
# Clean-room from the published math; numeric parity asserted in `test/runtests.jl`.
# --------------------------------------------------------------------------- #

const _LN2 = log(2.0)

# Positive overlapping n-gram counts of a message (empty if it is too short).
function _ngram_counts(message::AbstractString, approximate_word_length::Integer)
    (isempty(message) || length(message) < approximate_word_length) && return Float64[]
    chars = collect(message)
    n = length(chars)
    counts = Dict{String,Int}()
    for i = 1:(n-approximate_word_length+1)
        word = String(chars[i:(i+approximate_word_length-1)])
        counts[word] = get(counts, word, 0) + 1
    end
    return Float64.(collect(values(counts)))
end

# Plug-in entropy (bits) of a count vector: H = -Σ p log2 p.
function _plugin_bits(counts::AbstractVector{<:Real})
    n = sum(counts)
    n <= 0 && return 0.0
    p = counts ./ n
    return -sum(pi * log2(pi) for pi in p)
end

"""
    miller_madow_entropy(message, approximate_word_length=1) -> Float64

Miller–Madow bias-corrected Shannon entropy (bits per symbol): the plug-in
entropy plus the first-order analytic correction `(K̂-1)/(2N)`, with K̂ the number
of observed n-grams and N the count. The cheapest correction.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer a bias-corrected estimator over the plug-in whenever the symbol counts are
undersampled (a large effective alphabet relative to the sample: long words, fine
encodings, or short windows); Miller-Madow is the cheap first-order fix. It
converges to the plug-in when N is much larger than K, with no over-correction;
the gain is negligible for coarse encodings (e.g. binary). Reference: Miller, G.
(1955), Note on the bias of information estimates.
"""
function miller_madow_entropy(message::AbstractString, approximate_word_length::Integer = 1)
    counts = _ngram_counts(message, approximate_word_length)
    isempty(counts) && return 0.0
    n = sum(counts)
    k_hat = length(counts)
    corrected = _plugin_bits(counts) + (k_hat - 1.0) / (2.0 * n * _LN2)
    return corrected / approximate_word_length
end

# G(n) = ψ(n) + 0.5(-1)ⁿ[ψ((n+1)/2) - ψ(n/2)] (Grassberger 2008), in nats.
function _grassberger_g(counts::AbstractVector{<:Real})
    return [
        digamma(ni) +
        0.5 * (iseven(Int(round(ni))) ? 1.0 : -1.0) *
        (digamma((ni + 1.0) / 2.0) - digamma(ni / 2.0)) for ni in counts
    ]
end

"""
    grassberger_entropy(message, approximate_word_length=1) -> Float64

Grassberger (2008) bias-corrected Shannon entropy (bits per symbol):
`H = ln N - (1/N) Σ nᵢ G(nᵢ)` converted to bits. A higher-order correction with
no tuning, intermediate in cost between Miller–Madow and NSB.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer a bias-corrected estimator over the plug-in whenever the symbol counts are
undersampled; Grassberger is close to NSB at lower cost (the practical default
when undersampled). It converges to the plug-in when N is much larger than K, with
no over-correction; the gain is negligible for coarse encodings (e.g. binary).
Reference: Grassberger, P. (2008), Entropy estimates from insufficient samplings.
"""
function grassberger_entropy(message::AbstractString, approximate_word_length::Integer = 1)
    counts = _ngram_counts(message, approximate_word_length)
    isempty(counts) && return 0.0
    n = sum(counts)
    h_nats = log(n) - (1.0 / n) * sum(counts .* _grassberger_g(counts))
    return (h_nats / _LN2) / approximate_word_length
end

# --- NSB (Nemenman–Shafee–Bialek 2002) ------------------------------------- #

# A priori mean entropy (nats) implied by concentration β over K symbols.
_xi_of_beta(beta::Real, k::Integer) = digamma(k * beta + 1.0) - digamma(beta + 1.0)

# Invert a monotone-increasing function by bisection (β such that xi(β,k)=target).
function _invert_xi(target::Real, k::Integer; lo = 1e-8, hi = 1e8, xtol = 1e-12)
    a, b = float(lo), float(hi)
    fa = _xi_of_beta(a, k) - target
    for _ = 1:200
        mid = 0.5 * (a + b)
        fm = _xi_of_beta(mid, k) - target
        if fm == 0.0 || (b - a) < xtol
            return mid
        end
        if (fa < 0) == (fm < 0)
            a, fa = mid, fm
        else
            b = mid
        end
    end
    return 0.5 * (a + b)
end

# Quadrature nodes for the NSB ξ-integral over (0, ln K): (ξ_nodes, β_nodes).
function _nsb_beta_grid(k::Integer, n_points::Integer)
    xi_max = log(k)
    xi_nodes =
        range(xi_max / (n_points + 1), xi_max * n_points / (n_points + 1); length = n_points)
    beta_nodes = [_invert_xi(xi, k) for xi in xi_nodes]
    return (collect(xi_nodes), beta_nodes)
end

# Trapezoidal integral ∫ y dx for sorted nodes x.
function _trapz(y::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    s = 0.0
    @inbounds for i = 1:(length(x)-1)
        s += (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2.0
    end
    return s
end

# Unnormalised log marginal likelihood at a concentration β (nats).
function _log_evidence(beta::Real, k::Integer, n::Real, counts::AbstractVector{<:Real})
    obs = sum(loggamma(ci + beta) - loggamma(beta) for ci in counts)
    return loggamma(k * beta) - loggamma(n + k * beta) + obs
end

# Posterior mean entropy (nats) at a concentration β (Wolpert–Wolf 1995).
function _mean_entropy_given_beta(
    beta::Real,
    k::Integer,
    n::Real,
    counts::AbstractVector{<:Real},
    k_obs::Integer,
)
    denom = n + k * beta
    observed = sum((ci + beta) * digamma(ci + beta + 1.0) for ci in counts)
    unobserved = (k - k_obs) * beta * digamma(beta + 1.0)
    return digamma(denom + 1.0) - (observed + unobserved) / denom
end

# NSB posterior mean entropy (bits) for a count vector and true alphabet size K.
function _nsb_from_counts(counts::AbstractVector{<:Real}, k::Integer, n_points::Integer)
    n = sum(counts)
    (n <= 0 || k <= 1) && return 0.0
    k_obs = length(counts)
    xi_nodes, beta = _nsb_beta_grid(k, n_points)
    log_ev = [_log_evidence(b, k, n, counts) for b in beta]
    mean_h = [_mean_entropy_given_beta(b, k, n, counts, k_obs) for b in beta]
    log_ev = log_ev .- maximum(log_ev)         # stabilise before exponentiating
    weight = exp.(log_ev)
    numerator = _trapz(weight .* mean_h, xi_nodes)
    normaliser = _trapz(weight, xi_nodes)
    (normaliser <= 0 || !isfinite(normaliser)) && return NaN
    return (numerator / normaliser) / _LN2
end

"""
    nsb_entropy(message, approximate_word_length=1, alphabet_size=nothing; n_points=80) -> Float64

NSB (Nemenman–Shafee–Bialek 2002) Bayesian Shannon entropy (bits per symbol): the
Wolpert–Wolf posterior mean entropy integrated over the Dirichlet concentration
against a prior chosen so the implied prior on entropy is near-uniform. Needs the
true alphabet size (`alphabet_size`, the number of distinct base symbols; the
effective alphabet of words is `alphabet_size ^ approximate_word_length`, defaulting
to the number of distinct symbols observed). The most accurate estimator in deep
undersampling, and the most expensive.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer a bias-corrected estimator over the plug-in whenever the symbol counts are
undersampled; NSB is most accurate in deep undersampling. It converges to the
plug-in when N is much larger than K, with no over-correction; the gain is
negligible for coarse encodings (e.g. binary), and a gap caused by
non-stationarity (e.g. sigma encoding) is not an entropy-bias problem the
correction can fix. Reference: Nemenman, Shafee & Bialek (2002), Entropy and
inference, revisited.
"""
function nsb_entropy(
    message::AbstractString,
    approximate_word_length::Integer = 1,
    alphabet_size::Union{Integer,Nothing} = nothing;
    n_points::Integer = 80,
)
    counts = _ngram_counts(message, approximate_word_length)
    isempty(counts) && return 0.0
    base = alphabet_size === nothing ? length(Set(collect(message))) : alphabet_size
    base = max(Int(base), 1)
    k_eff = base^approximate_word_length
    # The effective alphabet cannot be smaller than the number of distinct words observed.
    k_eff = max(k_eff, length(counts))
    return _nsb_from_counts(counts, k_eff, n_points) / approximate_word_length
end
