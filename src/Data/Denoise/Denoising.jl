"""
Covariance-matrix denoising via Random Matrix Theory — native Julia port
mirroring the Python `RiskLabAI.data.denoise.denoising` API (López de Prado,
AFML Ch. 2): Marcenko–Pastur PDF, eigenvalue denoising, and cov↔corr helpers.

The closed-form pieces (`marcenko_pastur_pdf`, `pca`, `cov_to_corr`,
`corr_to_cov`, `denoised_corr`, `optimal_portfolio`) match Python to numerical
precision. The Marcenko–Pastur *fit* (`find_max_eval` / `denoise_cov`) uses a
Gaussian KDE and a 1-D minimiser; these are implementation-defined (Python uses
scikit-learn's KDE + SciPy's optimiser), so they are validated behaviourally.

Reference: De Prado, M. (2018), Advances in Financial Machine Learning, Ch. 2.
"""

using LinearAlgebra
using Statistics: mean, std, cov
using Random: AbstractRNG, default_rng, randperm, MersenneTwister

"""
    marcenko_pastur_pdf(variance, q; num_points=1000) -> (eigenvalues, pdf)

Theoretical Marcenko–Pastur density of eigenvalues for a random correlation
matrix with observation/feature ratio `q = T/N`. Mirrors Python's
`marcenko_pastur_pdf` (returns the eigenvalue grid and the density on it).
"""
function marcenko_pastur_pdf(variance::Real, q::Real; num_points::Integer = 1000)
    lambda_min = variance * (1 - sqrt(1 / q))^2
    lambda_max = variance * (1 + sqrt(1 / q))^2
    e_min = max(lambda_min, 1e-10)
    eigenvalues = collect(range(e_min, lambda_max; length = num_points))
    pdf = similar(eigenvalues)
    for i in eachindex(eigenvalues)
        e = eigenvalues[i]
        radicand = (lambda_max - e) * (e - lambda_min)
        pdf[i] = radicand < 0 ? 0.0 : q / (2 * pi * variance * e) * sqrt(radicand)
    end
    return eigenvalues, pdf
end

"""
    pca(matrix) -> (eigenvalues, eigenvectors)

Eigendecomposition of a Hermitian matrix with eigenvalues sorted **descending**
(and eigenvectors reordered to match). Mirrors Python's `pca`.
"""
function pca(matrix::AbstractMatrix{<:Real})
    factorization = eigen(Symmetric(Matrix(matrix)))
    order = sortperm(factorization.values; rev = true)
    return factorization.values[order], factorization.vectors[:, order]
end

"""
    cov_to_corr(cov) -> Matrix

Convert a covariance matrix to a correlation matrix (clamped to [-1, 1], unit
diagonal). Mirrors Python's `cov_to_corr`.
"""
function cov_to_corr(cov::AbstractMatrix{<:Real})
    std = sqrt.(diag(cov))
    std[std .== 0] .= 1.0
    corr = cov ./ (std * std')
    corr = clamp.(corr, -1.0, 1.0)
    corr[diagind(corr)] .= 1.0
    return corr
end

"""
    corr_to_cov(corr, std) -> Matrix

Convert a correlation matrix back to covariance given per-asset std devs.
Mirrors Python's `corr_to_cov`.
"""
corr_to_cov(corr::AbstractMatrix{<:Real}, std::AbstractVector{<:Real}) =
    corr .* (std * std')

"""
    denoised_corr(eigenvalues, eigenvectors, num_facts) -> Matrix

Reconstruct a denoised correlation matrix, keeping the `num_facts` largest
(signal) eigenvalues and replacing the rest with their average (noise). Assumes
`eigenvalues` are sorted descending. Mirrors Python's `denoised_corr`.
"""
function denoised_corr(
    eigenvalues::AbstractVector{<:Real},
    eigenvectors::AbstractMatrix{<:Real},
    num_facts::Integer,
)
    n = length(eigenvalues)
    signal_values = eigenvalues[1:num_facts]
    signal_vectors = eigenvectors[:, 1:num_facts]
    corr = signal_vectors * Diagonal(signal_values) * signal_vectors'

    if num_facts < n
        avg_noise = mean(eigenvalues[(num_facts+1):end])
        noise_vectors = eigenvectors[:, (num_facts+1):end]
        corr += noise_vectors * Diagonal(fill(avg_noise, n - num_facts)) * noise_vectors'
    end

    inv_sqrt = 1.0 ./ sqrt.(diag(corr))
    corr = Diagonal(inv_sqrt) * corr * Diagonal(inv_sqrt)
    corr[diagind(corr)] .= 1.0
    return corr
end

# Gaussian KDE density (matches scikit-learn KernelDensity, gaussian kernel):
# density(x) = (1 / (n·h·√(2π))) Σ_i exp(-((x - xᵢ)/h)² / 2).
function _gaussian_kde_density(
    observations::AbstractVector{<:Real},
    query::AbstractVector{<:Real},
    bandwidth::Real,
)
    n = length(observations)
    scale = 1.0 / (n * bandwidth * sqrt(2 * pi))
    density = similar(query, Float64)
    for (k, x) in enumerate(query)
        acc = 0.0
        for xi in observations
            acc += exp(-0.5 * ((x - xi) / bandwidth)^2)
        end
        density[k] = acc * scale
    end
    return density
end

function _mp_pdf_fit_error(
    variance::Real,
    q::Real,
    eigenvalues::AbstractVector{<:Real},
    bandwidth::Real,
)
    grid, theoretical = marcenko_pastur_pdf(variance, q; num_points = length(eigenvalues))
    empirical = _gaussian_kde_density(eigenvalues, grid, bandwidth)
    return sum((empirical .- theoretical) .^ 2)
end

# 1-D bounded minimiser (golden-section search) — replaces SciPy's `minimize`.
function _golden_section_min(f, a::Real, b::Real; tol::Real = 1e-6, max_iter::Integer = 200)
    invphi = (sqrt(5) - 1) / 2
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ = 1:max_iter
        if b - a < tol
            break
        end
        if fc < fd
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
        end
    end
    return (a + b) / 2
end

"""
    find_max_eval(eigenvalues, q, bandwidth) -> (lambda_max, variance)

Fit the Marcenko–Pastur distribution to the observed eigenvalues to find the
noise cutoff `lambda_max` and the implied variance. Mirrors Python's
`find_max_eval`; the KDE + minimiser are implementation-defined (behavioural
parity only).
"""
function find_max_eval(eigenvalues::AbstractVector{<:Real}, q::Real, bandwidth::Real)
    variance = _golden_section_min(
        v -> _mp_pdf_fit_error(v, q, eigenvalues, bandwidth),
        1e-5,
        1 - 1e-5,
    )
    lambda_max = variance * (1 + sqrt(1 / q))^2
    return lambda_max, variance
end

"""
    denoise_cov(cov0, q; bandwidth=0.01) -> Matrix

De-noise a covariance matrix: convert to correlation, fit Marcenko–Pastur to
locate the signal eigenvalues, rebuild the denoised correlation, and convert
back. Mirrors Python's `denoise_cov` (KDE-fit step is behavioural parity).
"""
function denoise_cov(cov0::AbstractMatrix{<:Real}, q::Real; bandwidth::Real = 0.01)
    corr0 = cov_to_corr(cov0)
    eigenvalues, eigenvectors = pca(corr0)
    lambda_max, _ = find_max_eval(eigenvalues, q, bandwidth)
    num_facts = count(>(lambda_max), eigenvalues)
    corr1 = denoised_corr(eigenvalues, eigenvectors, num_facts)
    return corr_to_cov(corr1, sqrt.(diag(cov0)))
end

"""
    optimal_portfolio(cov; mu=nothing) -> Vector

Closed-form optimal portfolio weights (global-minimum-variance when `mu` is
`nothing`). Mirrors Python's `optimal_portfolio`.
"""
function optimal_portfolio(cov::AbstractMatrix{<:Real}; mu = nothing)
    inv_cov = inv(cov)
    ones_vec = ones(size(cov, 1))
    target = mu === nothing ? ones_vec : vec(mu)
    weights = inv_cov * target
    return weights ./ dot(ones_vec, weights)
end

"""
    optimal_portfolio_denoised(cov, q; mu=nothing, bandwidth=0.01) -> Vector

Optimal portfolio weights computed from a denoised covariance matrix. Mirrors
Python's `optimal_portfolio_denoised`.
"""
function optimal_portfolio_denoised(
    cov::AbstractMatrix{<:Real},
    q::Real;
    mu = nothing,
    bandwidth::Real = 0.01,
)
    return optimal_portfolio(denoise_cov(cov, q; bandwidth = bandwidth); mu = mu)
end

# --------------------------------------------------------------------------- #
# NERCOME: nonparametric eigenvalue-regularized covariance estimation (Lam 2016).
#
# de Prado denoises the covariance by Marcenko–Pastur eigenvalue clipping
# (`denoise_cov`), which assumes a clean noise bulk separated from the signal
# eigenvalues by a gap. When the spectrum has no clean gap (a slowly-decaying
# bulk) or is non-stationary, that assumption breaks and clipping degenerates
# toward the raw sample covariance. NERCOME instead regularizes the eigenvalues
# by sample-splitting: eigenvectors come from one split, the oracle eigenvalues
# from projecting the held-out split's covariance onto them, averaged over many
# random splits. Clean-room from Lam (2016); a data-driven estimator (it takes
# the return matrix, not a covariance). Admitted in Appraisal 24
# (`library_extension/appraisals/24_verdict.md`).
#
# Stochastic-step divergence (documented): the split permutations use Julia's
# `randperm`/`rng` rather than NumPy's PCG64 stream, so the estimate is
# reproducible under a given `rng` but not bit-identical to the Python reference.
# Parity is therefore mechanism-level (lower covariance error than MP clipping on
# a no-gap spectrum, symmetric PD, better conditioning), as the existing
# path-level modules (`denoise_cov` KDE fit, `simulate_psy_critical_values`) do.
# --------------------------------------------------------------------------- #

# Project a symmetric matrix to the nearest PD one by flooring its eigenvalues.
function _ensure_positive_definite(matrix::AbstractMatrix{<:Real}; eps::Real = 1e-10)
    symmetric = (matrix + matrix') / 2.0
    values, vectors = eigen(Symmetric(symmetric))
    max_value = isempty(values) ? 0.0 : maximum(values)
    floor = (length(values) > 0 && max_value > 0) ? max(eps, eps * max_value) : eps
    values = clamp.(values, floor, Inf)
    out = (vectors * Diagonal(values)) * vectors'
    return (out + out') / 2.0
end

"""
    nercome_denoised_covariance(returns; n_splits=50, split_fraction=2/3, random_state=nothing) -> Matrix

NERCOME denoised covariance (Lam 2016), estimated by sample-splitting from a
`T × p` return matrix (rows = observations, columns = assets). For each of
`n_splits` random row permutations the observations are split in two; the
eigenvectors `P` come from the first half's sample covariance, and the
regularized eigenvalues are the oracle projection `dᵢ = pᵢ' S₂ pᵢ` of the second
half's covariance `S₂` onto those eigenvectors. The estimator `P diag(d) P'` is
averaged over the splits (each term is PSD, so the average is PSD). Returns are
standardized to unit sample variance, NERCOME-cleaned in correlation space, then
scaled back by the sample column standard deviations.

Preferred-when / avoid-when (regime tag, verbatim from `appraisals/24_verdict.md`):
prefer NERCOME over MP clipping for covariance estimation when the eigenvalue
spectrum has no clean gap or is non-stationary (better accuracy and conditioning,
and lower OOS volatility via NCO / min-variance); it converges to clipping on
clean-gap stationary spectra. It costs more turnover and concentration than
clipping, gives no risk-adjusted-return edge (none does - 1/N stands), and HRP
does not benefit (use it through NCO / min-variance).

Unlike [`denoise_cov`](@ref), which cleans a covariance matrix, this takes the
return matrix. `random_state` seeds Julia's RNG (see the divergence note above;
the Python reference uses NumPy's PCG64). Mirrors Python's
`nercome_denoised_covariance`.

Reference: Lam, C. (2016). Nonparametric eigenvalue-regularized precision or
covariance matrix estimator. The Annals of Statistics, 44(3), 928–953.
"""
function nercome_denoised_covariance(
    returns::AbstractMatrix{<:Real};
    n_splits::Integer = 50,
    split_fraction::Real = 2.0 / 3.0,
    random_state::Union{Integer,Nothing} = nothing,
)
    data = Float64.(returns)
    n, p = size(data)
    n >= 4 || throw(ArgumentError("NERCOME needs at least 4 observations"))
    sample_std = vec(std(data; dims = 1))                 # ddof=1 (corrected), per column
    sample_std = [s <= 0 ? 1.0 : s for s in sample_std]
    z = data ./ sample_std'

    m = max(p + 1, round(Int, split_fraction * n))
    m = min(m, n - 2)
    rng::AbstractRNG = random_state === nothing ? default_rng() : MersenneTwister(random_state)
    accumulated = zeros(p, p)
    used = 0
    for _ = 1:n_splits
        order = randperm(rng, n)
        first, second = order[1:m], order[(m+1):end]
        length(second) < 2 && continue
        cov_first = cov(view(z, first, :))                # columns = variables (dims=1)
        cov_second = cov(view(z, second, :))
        _, vectors = eigen(Symmetric(cov_first))
        oracle = [dot(view(vectors, :, i), cov_second * view(vectors, :, i)) for i = 1:p]
        oracle = clamp.(oracle, 0.0, Inf)
        accumulated += (vectors * Diagonal(oracle)) * vectors'
        used += 1
    end
    estimate = accumulated / max(used, 1)
    correlation = cov_to_corr(_ensure_positive_definite(estimate))
    return corr_to_cov(correlation, sample_std)
end
