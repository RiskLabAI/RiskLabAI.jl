"""
Information-theoretic distance metrics — native Julia port mirroring the Python
`RiskLabAI.data.distance.distance_metric` API (López de Prado, AFML Ch. 3):
variation of information, mutual information, optimal binning, angular distance,
KL divergence and cross-entropy.

The 2-D histogram binning replicates `numpy.histogram2d` (equal-width bins, the
last bin closed on the right) and the mutual-information / entropy formulas
replicate scikit-learn's `mutual_info_score` and SciPy's `entropy` (natural
log), so the metrics match the Python implementation exactly.

Reference: De Prado, M. (2018/2020), Advances in Financial Machine Learning, Ch. 3.
"""

using Statistics: cor

# numpy.histogram2d with an integer bin count (equal-width; last bin closed).
function _histogram2d(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, bins::Integer)
    counts = zeros(Float64, bins, bins)
    bin_index(v, lo, hi) =
        lo == hi ? 1 :
        (v == hi ? bins : clamp(floor(Int, (v - lo) / (hi - lo) * bins) + 1, 1, bins))
    xlo, xhi = minimum(x), maximum(x)
    ylo, yhi = minimum(y), maximum(y)
    for k in eachindex(x)
        counts[bin_index(x[k], xlo, xhi), bin_index(y[k], ylo, yhi)] += 1.0
    end
    return counts
end

# SciPy entropy (natural log) of a count vector.
function _entropy(counts::AbstractVector{<:Real})
    total = sum(counts)
    total == 0 && return 0.0
    h = 0.0
    for c in counts
        if c > 0
            p = c / total
            h -= p * log(p)
        end
    end
    return h
end

# scikit-learn mutual_info_score from a contingency (histogram) matrix, in nats.
function _mutual_info(hist::AbstractMatrix{<:Real})
    total = sum(hist)
    total == 0 && return 0.0
    row = vec(sum(hist; dims = 2))
    col = vec(sum(hist; dims = 1))
    mi = 0.0
    for i in axes(hist, 1), j in axes(hist, 2)
        nij = hist[i, j]
        if nij > 0
            mi += (nij / total) * log(nij * total / (row[i] * col[j]))
        end
    end
    return mi
end

"""
    calculate_variation_of_information(x, y, bins; norm=false) -> Float64

Variation of information `VI = H(X) + H(Y) - 2·I(X, Y)` from a `bins`×`bins`
histogram; optionally normalised by the joint entropy. Mirrors Python's
`calculate_variation_of_information`.
"""
function calculate_variation_of_information(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    bins::Integer;
    norm::Bool = false,
)
    hist = _histogram2d(x, y, bins)
    mutual_information = _mutual_info(hist)
    marginal_x = _entropy(vec(sum(hist; dims = 2)))
    marginal_y = _entropy(vec(sum(hist; dims = 1)))
    variation = marginal_x + marginal_y - 2 * mutual_information
    if norm
        joint = marginal_x + marginal_y - mutual_information
        return joint == 0 ? 0.0 : variation / joint
    end
    return variation
end

"""
    calculate_number_of_bins(num_observations; correlation=nothing) -> Int

Optimal number of histogram bins (univariate when `correlation` is `nothing`,
otherwise bivariate). Mirrors Python's `calculate_number_of_bins`.
"""
function calculate_number_of_bins(num_observations::Integer; correlation = nothing)
    if correlation === nothing
        z =
            (
                8 +
                324 * num_observations +
                12 * sqrt(36 * num_observations + 729 * num_observations^2)
            )^(1 / 3)
        return round(Int, z / 6 + 2 / (3z) + 1 / 3)
    end
    if isapprox(correlation, 1.0) || isapprox(correlation, -1.0)
        correlation = sign(correlation) * (1 - 1e-10)
    end
    (1 - correlation^2) == 0 && return calculate_number_of_bins(num_observations)
    return round(
        Int,
        2^-0.5 * sqrt(1 + sqrt(1 + 24 * num_observations / (1 - correlation^2))),
    )
end

"""
    calculate_variation_of_information_extended(x, y; norm=false) -> Float64

Variation of information using the optimal bivariate bin count. Mirrors Python's
`calculate_variation_of_information_extended`.
"""
function calculate_variation_of_information_extended(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real};
    norm::Bool = false,
)
    bins = calculate_number_of_bins(length(x); correlation = cor(x, y))
    return calculate_variation_of_information(x, y, bins; norm = norm)
end

"""
    calculate_mutual_information(x, y; norm=false) -> Float64

Mutual information using the optimal bivariate bin count; optionally normalised
by `min(H(X), H(Y))`. Mirrors Python's `calculate_mutual_information`.
"""
function calculate_mutual_information(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real};
    norm::Bool = false,
)
    bins = calculate_number_of_bins(length(x); correlation = cor(x, y))
    hist = _histogram2d(x, y, bins)
    mutual_information = _mutual_info(hist)
    if norm
        min_entropy =
            min(_entropy(vec(sum(hist; dims = 2))), _entropy(vec(sum(hist; dims = 1))))
        return min_entropy == 0 ? 0.0 : mutual_information / min_entropy
    end
    return mutual_information
end

"""
    calculate_distance(dependence; metric="angular") -> Matrix

Angular (`"angular"`) or absolute-angular (`"absolute_angular"`) distance matrix
from a dependence (correlation) matrix. Mirrors Python's `calculate_distance`.
"""
function calculate_distance(
    dependence::AbstractMatrix{<:Real};
    metric::AbstractString = "angular",
)
    dep = clamp.(dependence, -1.0, 1.0)
    if metric == "angular"
        return sqrt.(round.(1 .- dep; digits = 6) ./ 2)
    elseif metric == "absolute_angular"
        return sqrt.(round.(1 .- abs.(dep); digits = 6) ./ 2)
    else
        throw(ArgumentError("Unknown metric: $metric"))
    end
end

"""
    calculate_kullback_leibler_divergence(p, q) -> Float64

Kullback–Leibler divergence `D(P‖Q) = Σ pᵢ log(pᵢ/qᵢ)` (natural log; `p`, `q`
are normalised to sum to 1). Mirrors Python's
`calculate_kullback_leibler_divergence`.
"""
function calculate_kullback_leibler_divergence(
    p::AbstractVector{<:Real},
    q::AbstractVector{<:Real},
)
    pn = p ./ sum(p)
    qn = q ./ sum(q)
    any((pn .> 0) .& (qn .== 0)) && return Inf
    divergence = 0.0
    for i in eachindex(pn)
        if pn[i] > 0 && qn[i] > 0
            divergence -= pn[i] * log(qn[i] / pn[i])
        end
    end
    return divergence
end

"""
    calculate_cross_entropy(p, q) -> Float64

Cross-entropy `H(P, Q) = -Σ pᵢ log(qᵢ)` (natural log; `p`, `q` normalised).
Mirrors Python's `calculate_cross_entropy`.
"""
function calculate_cross_entropy(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
    pn = p ./ sum(p)
    qn = q ./ sum(q)
    any((pn .> 0) .& (qn .== 0)) && return Inf
    entropy = 0.0
    for i in eachindex(pn)
        if pn[i] > 0 && qn[i] > 0
            entropy -= pn[i] * log(qn[i])
        end
    end
    return entropy
end
