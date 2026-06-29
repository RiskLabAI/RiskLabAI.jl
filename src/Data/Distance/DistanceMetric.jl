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

# --------------------------------------------------------------------------- #
# Nonparametric codependence kernels: KSG kNN mutual information (Kraskov–
# Stögbauer–Grassberger 2004) and distance correlation (Székely–Rizzo–Bakirov
# 2007). Binning-free alternatives to the histogram MI above: KSG uses adaptive
# kNN distances (Chebyshev norm), distance correlation is a tuning-free nonlinear
# dependence index in [0, 1]. Clean-room from the published math; numeric parity
# asserted in `test/runtests.jl` (distance correlation exactly; KSG via the
# jitter-invariant integer neighbour counts). Admitted in Appraisal 11
# (`library_extension/appraisals/11_verdict.md`).
# --------------------------------------------------------------------------- #

using SpecialFunctions: digamma
using Statistics: mean, std
using Random: MersenneTwister, randn

# Break exact ties with negligible data-scaled noise (numpy population std).
function _jitter(values, rng)
    v = float.(vec(values))
    scale = std(v; corrected = false)
    scale == 0.0 && (scale = 1.0)
    return v .+ randn(rng, length(v)) .* scale .* 1e-10
end

"""
    ksg_mutual_information(x, y; k=4, random_state=0) -> Float64

Kraskov–Stögbauer–Grassberger mutual information (algorithm 1), in nats, using the
Chebyshev (max) norm in the joint space:
`Î(X;Y) = ψ(k) + ψ(N) - ⟨ψ(nₓ+1) + ψ(n_y+1)⟩`. Binning-free, so far less biased
than histogram MI on short / nonlinear / heavy-tailed samples; it can return a
slightly negative value for (near-)independent data (a characterized property, not
an error).

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer KSG over binned MI/VI on short, noisy, or nonlinear/heavy-tailed samples
(e.g. nonlinear-monotone n=1000 RMSE 0.039 vs binned 0.164); it is essentially
unbiased on linear dependence and converges to binned on large-sample near-linear
data (no free lunch there). Use raw KSG (the surrogate de-bias adds nothing).

Brute-force O(N²) kNN (no kd-tree dependency). Mirrors Python's
`ksg_mutual_information`. Reference: Kraskov, Stögbauer & Grassberger (2004),
Physical Review E 69(6).
"""
function ksg_mutual_information(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real};
    k::Integer = 4,
    random_state::Union{Integer,Nothing} = 0,
)
    rng = MersenneTwister(random_state === nothing ? 0 : random_state)
    xj = _jitter(x, rng)
    yj = _jitter(y, rng)
    n = length(xj)
    n < 2 && return 0.0
    k = min(k, n - 1)

    eps = Vector{Float64}(undef, n)
    dists = Vector{Float64}(undef, n)
    for i = 1:n
        @inbounds for j = 1:n
            dists[j] = max(abs(xj[i] - xj[j]), abs(yj[i] - yj[j]))
        end
        partialsort!(dists, k + 1)            # k+1-th smallest = k-th non-self neighbour
        eps[i] = dists[k+1]
    end

    total = 0.0
    for i = 1:n
        radius = eps[i] * (1.0 - 1e-10)
        nx = 0
        ny = 0
        @inbounds for j = 1:n
            abs(xj[i] - xj[j]) <= radius && (nx += 1)
            abs(yj[i] - yj[j]) <= radius && (ny += 1)
        end
        nx = max(nx - 1, 0)                    # exclude self
        ny = max(ny - 1, 0)
        total += digamma(nx + 1) + digamma(ny + 1)
    end
    return digamma(k) + digamma(n) - total / n
end

function _double_center(distance_matrix::AbstractMatrix{<:Real})
    row = sum(distance_matrix; dims = 2) ./ size(distance_matrix, 2)
    col = sum(distance_matrix; dims = 1) ./ size(distance_matrix, 1)
    return distance_matrix .- row .- col .+ (sum(distance_matrix) / length(distance_matrix))
end

"""
    distance_correlation(x, y) -> Float64

Distance correlation (Székely–Rizzo–Bakirov 2007), a dependence index in [0, 1],
from the double-centred pairwise-distance matrices `A`, `B`:
`dCor = √(mean(A·B) / √(mean(A·A)·mean(B·B)))`. Zero only at population
independence; detects nonlinear dependence with no estimation parameter.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer distance correlation as a parameter-free nonlinear screen / for maximally
stable clustering (best real-data ONC stability). It is a dependence index, not a
metric on partitions like the variation of information (keep VI/KSG for the metric
role).

Mirrors Python's `distance_correlation`. Reference: Székely, Rizzo & Bakirov
(2007), Annals of Statistics 35(6).
"""
function distance_correlation(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    xv = float.(vec(x))
    yv = float.(vec(y))
    n = length(xv)
    a = [abs(xv[i] - xv[j]) for i = 1:n, j = 1:n]
    b = [abs(yv[i] - yv[j]) for i = 1:n, j = 1:n]
    centered_a = _double_center(a)
    centered_b = _double_center(b)
    dcov2 = mean(centered_a .* centered_b)
    dvar_x = mean(centered_a .* centered_a)
    dvar_y = mean(centered_b .* centered_b)
    denom = sqrt(dvar_x * dvar_y)
    denom <= 0 && return 0.0
    return sqrt(max(dcov2, 0.0) / denom)
end
