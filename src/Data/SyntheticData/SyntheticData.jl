"""
Synthetic data — native Julia port mirroring the Python
`RiskLabAI.data.synthetic_data` sub-package: random/structured covariance
generators, the Drift-Burst-Hypothesis drift/volatility profile, and a
Heston–Merton regime-switching price simulator.

Parity notes:
  * `form_block_matrix`, `drift_volatility_burst`, `compute_log_returns` and
    `align_params_length` are **deterministic** and match Python exactly
    (verified in `test/runtests.jl`).
  * `random_cov`, `form_true_matrix`, `simulates_cov_mu`,
    `heston_merton_log_returns`, `generate_prices_from_regimes` and
    `parallel_generate_prices` are **stochastic** (Monte-Carlo); they take an
    optional `random_state`/`rng`. Supplied draws allow numerical replay without
    assuming equal random streams across languages.

Deliberate divergence: numba/joblib/quantecon are replaced by native Julia
(plain loops, a native Markov-chain simulate, `Distributions`). Ledoit–Wolf
shrinkage in `simulates_cov_mu` uses the centered maximum-likelihood covariance
and scaled-identity target. The default `shrink=false` uses sample covariance.
Price generators return a plain price vector (Python returns a business-day
indexed Series).

Reference: De Prado, M. (2020), Machine Learning for Asset Managers, Ch. 2.
"""

using Random: AbstractRNG, MersenneTwister, default_rng, shuffle
using Statistics: mean, cov
using LinearAlgebra: Symmetric, Diagonal, diag
using Distributions: MvNormal, Poisson
using CovarianceEstimation: LinearShrinkage, DiagonalCommonVariance

_synth_rng(random_state) =
    random_state === nothing ? default_rng() :
    random_state isa AbstractRNG ? random_state : MersenneTwister(random_state)

"""
    form_block_matrix(n_blocks, block_size, block_correlation) -> Matrix

Block-diagonal correlation matrix: `n_blocks` blocks of size `block_size`, each
with off-diagonal `block_correlation` and unit diagonal. Mirrors Python's
`form_block_matrix`.
"""
function form_block_matrix(n_blocks::Integer, block_size::Integer, block_correlation::Real)
    block = fill(Float64(block_correlation), block_size, block_size)
    for i = 1:block_size
        block[i, i] = 1.0
    end
    n = n_blocks * block_size
    out = zeros(Float64, n, n)
    for b = 0:(n_blocks-1)
        r = b * block_size
        out[(r+1):(r+block_size), (r+1):(r+block_size)] = block
    end
    return out
end

"""
    random_cov(num_columns, num_factors; rng=default_rng()) -> Matrix

Random positive-definite covariance `W Wᵀ + diag(u)` with `W` a
`num_columns × num_factors` standard-normal matrix and `u` uniform. Stochastic.
Mirrors Python's `random_cov`.
Optional `normal_draws` and `uniform_draws` provide the matrix and diagonal
draws directly, with shapes `(num_columns, num_factors)` and `num_columns`.
"""
function random_cov(num_columns::Integer, num_factors::Integer; rng = default_rng(), normal_draws = nothing, uniform_draws = nothing)
    w = normal_draws === nothing ? randn(rng, num_columns, num_factors) : Matrix{Float64}(normal_draws)
    u = uniform_draws === nothing ? rand(rng, num_columns) : Float64.(uniform_draws)
    size(w) == (num_columns, num_factors) && all(isfinite, w) || throw(ArgumentError("normal draws must have the requested finite matrix shape"))
    length(u) == num_columns && all(x -> isfinite(x) && 0 <= x <= 1, u) || throw(ArgumentError("uniform draws must have num_columns entries in [0, 1]"))
    cov = w * w'
    cov += Diagonal(u)
    return cov
end

"""
    form_true_matrix(n_blocks, block_size, block_correlation; rng=default_rng()) -> (mu, cov)

Shuffled block-diagonal correlation converted to a covariance matrix with random
volatilities, plus a random mean vector. Stochastic. Mirrors Python's
`form_true_matrix`.
Optional `permutation`, `standard_deviation_draws` and standard `normal_draws`
replay the public draws. The permutation uses Julia's one-based indices.
"""
function form_true_matrix(
    n_blocks::Integer,
    block_size::Integer,
    block_correlation::Real;
    rng = default_rng(),
    permutation = nothing,
    standard_deviation_draws = nothing,
    normal_draws = nothing,
)
    correlation = form_block_matrix(n_blocks, block_size, block_correlation)
    n = size(correlation, 1)
    order = permutation === nothing ? shuffle(rng, collect(1:n)) : Int.(permutation)
    sort(order) == collect(1:n) || throw(ArgumentError("permutation must contain every matrix index exactly once"))
    correlation = correlation[order, order]
    standard_deviations = standard_deviation_draws === nothing ? rand(rng, n) .* 0.15 .+ 0.05 : Float64.(standard_deviation_draws)  # U(0.05, 0.2)
    length(standard_deviations) == n && all(x -> isfinite(x) && .05 <= x <= .2, standard_deviations) || throw(ArgumentError("standard deviation draws must have n entries in [0.05, 0.2]"))
    normal = normal_draws === nothing ? randn(rng, n) : Float64.(normal_draws)
    length(normal) == n && all(isfinite, normal) || throw(ArgumentError("normal draws must have n finite entries"))
    covariance = corr_to_cov(correlation, standard_deviations)
    mu = normal .* standard_deviations .+ standard_deviations
    return mu, covariance
end

"""
    simulates_cov_mu(mu, covariance, n_observations; shrink=false, rng=default_rng()) -> (mu1, cov1)

Draw `n_observations` multivariate-normal samples from `(mu, covariance)` and
return their sample mean and covariance. Stochastic. Mirrors Python's
`simulates_cov_mu`. The `shrink=false` path uses unbiased sample covariance;
`shrink=true` uses centered Ledoit–Wolf scaled-identity shrinkage with
maximum-likelihood normalization. Degenerate targets retain their empirical
covariance. Native random streams need not match NumPy for equal seeds.
"""
function simulates_cov_mu(
    mu::AbstractVecOrMat{<:Real},
    covariance::AbstractMatrix{<:Real},
    n_observations::Integer;
    shrink::Bool = false,
    rng = default_rng(),
)
    samples = permutedims(
        rand(rng, MvNormal(vec(mu), Symmetric(Matrix(covariance))), n_observations),
    )
    mu1 = vec(mean(samples; dims = 1))
    cov1 = shrink ? _ledoit_wolf_covariance(samples) : cov(samples)
    return mu1, cov1
end

function _ledoit_wolf_covariance(samples::AbstractMatrix{<:Real})
    n, p = size(samples)
    n > 0 && p > 0 && all(isfinite, samples) || throw(ArgumentError("samples must be nonempty and finite"))
    shifted = Float64.(samples) .- samples[1:1, :]
    centered = shifted .- mean(shifted; dims = 1)
    all(isfinite, centered) || throw(ArgumentError("centered samples exceed floating-point range"))
    scale = maximum(abs, centered)
    scale == 0 && return zeros(p, p)
    normalized = centered / scale
    empirical = normalized' * normalized / n
    target = Diagonal(fill(sum(diag(empirical)) / p, p))
    covariance = if p == 1 || all(iszero, empirical - target)
        empirical
    else
        Matrix(cov(LinearShrinkage(DiagonalCommonVariance(), :lw; corrected = false), normalized; dims = 1))
    end
    result = covariance .* scale^2
    all(isfinite, result) || throw(ArgumentError("covariance exceeds floating-point range"))
    return result
end

"""
    drift_volatility_burst(bubble_length, a_before, a_after, b_before, b_after, alpha, beta; explosion_filter_width=0.1) -> (drifts, volatilities)

Drift-Burst-Hypothesis drift and volatility profiles over `[0, 1]`, with an
explosion clamped around the midpoint `t = 0.5`. Deterministic. Mirrors Python's
`drift_volatility_burst`.
"""
function drift_volatility_burst(
    bubble_length::Integer,
    a_before::Real,
    a_after::Real,
    b_before::Real,
    b_after::Real,
    alpha::Real,
    beta::Real;
    explosion_filter_width::Real = 0.1,
)
    w = explosion_filter_width
    steps = bubble_length == 1 ? [0.0] : collect(range(0.0, 1.0; length = bubble_length))
    filtered = copy(steps)
    for k in eachindex(steps)
        s = steps[k]
        if s >= 0.5 - w && s < 0.5
            filtered[k] = 0.5 - w
        elseif s > 0.5 && s <= 0.5 + w
            filtered[k] = 0.5 + w
        end
    end

    a_values = [s <= 0.5 ? Float64(a_before) : Float64(a_after) for s in steps]
    b_values = [s <= 0.5 ? Float64(b_before) : Float64(b_after) for s in steps]

    denominators = abs.(filtered .- 0.5)
    for k in eachindex(steps)
        steps[k] == 0.5 && (denominators[k] = NaN)
    end

    drifts = a_values ./ (denominators .^ alpha)
    volatilities = b_values ./ (denominators .^ beta)

    nan_indices = findall(isnan, denominators)
    if !isempty(nan_indices)
        for k in nan_indices
            drifts[k] = 0.0
        end
        first_nan = nan_indices[1]
        fill_value = first_nan > 1 ? volatilities[first_nan-1] : Float64(b_before)
        for k in nan_indices
            volatilities[k] = fill_value
        end
    end
    return drifts, volatilities
end

"""
    compute_log_returns(n_steps, mu, kappa, theta, xi, dw_stock, dw_vol, jump_comp, poisson_var, dt, sqrt_dt, lambda, m, v, regime_change) -> Vector{Float64}

Heston–Merton log returns given pre-drawn Wiener/jump increments. Deterministic.
Mirrors Python's numba-jitted `compute_log_returns`.
"""
function compute_log_returns(
    n_steps::Integer,
    mu::AbstractVector,
    kappa::AbstractVector,
    theta::AbstractVector,
    xi::AbstractVector,
    dw_stock::AbstractVector,
    dw_vol::AbstractVector,
    jump_comp::AbstractVector,
    poisson_var::AbstractVector,
    dt::Real,
    sqrt_dt::Real,
    lambda::AbstractVector,
    m::AbstractVector,
    v::AbstractVector,
    regime_change::AbstractVector{Bool},
)
    variance = Vector{Float64}(undef, n_steps + 1)
    jump_events = poisson_var .* jump_comp
    variance[1] = theta[1]
    log_returns = Vector{Float64}(undef, n_steps)
    for i = 1:n_steps
        regime_change[i] && (variance[i] = theta[i])
        v_safe = max(variance[i], 0.0)
        variance[i+1] =
            variance[i] +
            kappa[i] * (theta[i] - v_safe) * dt +
            xi[i] * sqrt(v_safe) * dw_vol[i] * sqrt_dt
        log_returns[i] =
            (mu[i] - 0.5 * v_safe - lambda[i] * (m[i] + (v[i]^2) / 2.0)) * dt +
            sqrt(v_safe) * dw_stock[i] * sqrt_dt +
            jump_events[i]
    end
    return log_returns
end

"""
    align_params_length(regime_params) -> (aligned, max_len)

Broadcast scalars and pad/truncate vectors so every parameter in a regime shares
the same length. Deterministic. Mirrors Python's `align_params_length`.
"""
function align_params_length(regime_params::AbstractDict)
    max_len = maximum(v isa AbstractVector ? length(v) : 1 for v in values(regime_params))
    aligned = Dict{keytype(regime_params),Vector{Float64}}()
    for (key, value) in regime_params
        if value isa AbstractVector
            if length(value) < max_len
                aligned[key] = vcat(
                    Float64.(value),
                    fill(Float64(value[end]), max_len - length(value)),
                )
            else
                aligned[key] = Float64.(value[1:max_len])
            end
        else
            aligned[key] = fill(Float64(value), max_len)
        end
    end
    return aligned, max_len
end

"""
    heston_merton_log_returns(total_time, n_steps, mu, kappa, theta, xi, rho, lambda, m, v, regime_change; random_state=nothing) -> Vector{Float64}

Draw correlated Wiener increments and Poisson jumps, then evaluate
`compute_log_returns`. Stochastic. Mirrors Python's `heston_merton_log_returns`.
`gaussian_draws` supplies an `n_steps × 3` matrix of already-correlated stock,
volatility and jump-size outcomes; `poisson_draws` supplies nonnegative integer
counts. Native sampling supports positive-semidefinite covariance, including
`rho = ±1` and zero jump dispersion. Equal language seeds need not agree.
"""
function heston_merton_log_returns(
    total_time::Real,
    n_steps::Integer,
    mu::AbstractVector,
    kappa::AbstractVector,
    theta::AbstractVector,
    xi::AbstractVector,
    rho::AbstractVector,
    lambda::AbstractVector,
    m::AbstractVector,
    v::AbstractVector,
    regime_change::AbstractVector{Bool};
    random_state = nothing,
    gaussian_draws = nothing,
    poisson_draws = nothing,
)
    n_steps > 0 && isfinite(total_time) && total_time >= 0 || throw(ArgumentError("n_steps must be positive and total_time finite and nonnegative"))
    all(length(p) == n_steps for p in (mu, kappa, theta, xi, rho, lambda, m, v, regime_change)) ||
        throw(ArgumentError("all parameter vectors must have length n_steps"))
    all(x -> isfinite(x) && abs(x) <= 1, rho) || throw(ArgumentError("correlations must be finite and in [-1, 1]"))
    all(x -> isfinite(x) && x >= 0, lambda) || throw(ArgumentError("jump intensities must be finite and nonnegative"))
    all(isfinite, m) && all(isfinite, v) || throw(ArgumentError("jump means and dispersions must be finite"))
    gaussian_draws === nothing || (size(gaussian_draws) == (n_steps, 3) && all(isfinite, gaussian_draws)) || throw(ArgumentError("gaussian_draws must be a finite n_steps by 3 matrix"))
    poisson_draws === nothing || (length(poisson_draws) == n_steps && all(x -> isfinite(x) && x >= 0 && isinteger(x), poisson_draws)) || throw(ArgumentError("poisson_draws must contain n_steps nonnegative integer counts"))

    rng = _synth_rng(random_state)
    dt = total_time / n_steps
    sqrt_dt = sqrt(dt)

    dw_stock = Vector{Float64}(undef, n_steps)
    dw_vol = Vector{Float64}(undef, n_steps)
    jump_comp = Vector{Float64}(undef, n_steps)
    poisson_var = Vector{Float64}(undef, n_steps)
    for i = 1:n_steps
        z = gaussian_draws === nothing ? _heston_merton_gaussian(rho[i], m[i], v[i], randn(rng, 3)) : view(gaussian_draws, i, :)
        dw_stock[i] = z[1]
        dw_vol[i] = z[2]
        jump_comp[i] = z[3]
        poisson_var[i] = poisson_draws === nothing ? rand(rng, Poisson(lambda[i] * dt)) : poisson_draws[i]
    end

    return compute_log_returns(
        n_steps,
        mu,
        kappa,
        theta,
        xi,
        dw_stock,
        dw_vol,
        jump_comp,
        poisson_var,
        dt,
        sqrt_dt,
        lambda,
        m,
        v,
        regime_change,
    )
end

function _heston_merton_gaussian(rho, jump_mean, jump_dispersion, normal)
    return (normal[1], rho * normal[1] + sqrt(max(1 - rho^2, 0)) * normal[2], jump_mean + abs(jump_dispersion) * normal[3])
end

# Native categorical sampler + Markov-chain simulate (replaces quantecon).
function _sample_categorical(rng, probabilities; uniform = nothing)
    u = uniform === nothing ? rand(rng) : uniform
    isfinite(u) && 0 <= u < 1 || throw(ArgumentError("transition uniforms must be in [0, 1)"))
    cumulative = 0.0
    last_positive = 0
    for i in eachindex(probabilities)
        cumulative += probabilities[i]
        probabilities[i] > 0 && (last_positive = i)
        u < cumulative && return i
    end
    return last_positive
end

function _markov_simulate(transition_matrix, n_steps, n_states; rng, initial_state = nothing, transition_uniforms = nothing)
    n_steps > 0 && n_states > 0 || throw(ArgumentError("state and step counts must be positive"))
    size(transition_matrix) == (n_states, n_states) && all(x -> isfinite(x) && x >= 0, transition_matrix) && all(x -> isapprox(x, 1; rtol = 1e-12, atol = 1e-12), sum(transition_matrix; dims = 2)) || throw(ArgumentError("transition rows must be finite nonnegative probability vectors"))
    transition_uniforms === nothing || length(transition_uniforms) == n_steps - 1 || throw(ArgumentError("transition_uniforms must have n_steps - 1 entries"))
    states = Vector{Int}(undef, n_steps)
    states[1] = initial_state === nothing ? rand(rng, 1:n_states) : initial_state
    1 <= states[1] <= n_states || throw(ArgumentError("initial_state must index a regime"))
    for t = 2:n_steps
        states[t] = _sample_categorical(rng, view(transition_matrix, states[t-1], :); uniform = transition_uniforms === nothing ? nothing : transition_uniforms[t-1])
    end
    return states
end

"""
    generate_prices_from_regimes(regimes, transition_matrix, total_time, n_steps; random_state=nothing) -> (prices, regime_path)

Simulate a Markov-switching Heston–Merton price path. `regimes` maps a regime
name to a parameter dictionary (keys `"mu","kappa","theta","xi","rho","lam","m","v"`).
Returns the price vector and the per-step regime names. Stochastic. Mirrors
Python's `generate_prices_from_regimes` (without the business-day index).
`regime_order` supplies the ordering of transition-matrix rows and columns;
its default is dictionary iteration order. For cross-language replay, provide
Python's insertion order, one-based `initial_state`, `transition_uniforms`,
and the `gaussian_draws`/`poisson_draws` accepted by `heston_merton_log_returns`.
Missing log returns are forward-filled before cumulative price construction.
"""
function generate_prices_from_regimes(
    regimes::AbstractDict,
    transition_matrix::AbstractMatrix{<:Real},
    total_time::Real,
    n_steps::Integer;
    random_state = nothing,
    regime_order = nothing,
    initial_state = nothing,
    transition_uniforms = nothing,
    gaussian_draws = nothing,
    poisson_draws = nothing,
)
    rng = _synth_rng(random_state)
    regime_names = regime_order === nothing ? collect(keys(regimes)) : collect(regime_order)
    length(regime_names) == length(regimes) && Set(regime_names) == Set(keys(regimes)) || throw(ArgumentError("regime_order must contain each regime name exactly once"))
    simulated =
        _markov_simulate(transition_matrix, n_steps, length(regime_names); rng = rng, initial_state = initial_state, transition_uniforms = transition_uniforms)
    simulated_regimes = [regime_names[s] for s in simulated]

    parameter_keys = ["mu", "kappa", "theta", "xi", "rho", "lam", "m", "v"]
    parameter_lists = Dict(k => Float64[] for k in parameter_keys)
    regime_path = eltype(regime_names)[]

    step = 1
    while step <= n_steps
        name = simulated_regimes[step]
        params, regime_length = align_params_length(regimes[name])
        regime_length > 0 || throw(ArgumentError("each regime must supply at least one parameter step"))
        take = min(regime_length, n_steps - step + 1)
        for k in parameter_keys
            append!(parameter_lists[k], params[k][1:take])
        end
        append!(regime_path, fill(name, take))
        step += take
    end

    regime_change = vcat(false, [regime_path[i] != regime_path[i-1] for i = 2:n_steps])
    log_returns = heston_merton_log_returns(
        total_time,
        n_steps,
        parameter_lists["mu"],
        parameter_lists["kappa"],
        parameter_lists["theta"],
        parameter_lists["xi"],
        parameter_lists["rho"],
        parameter_lists["lam"],
        parameter_lists["m"],
        parameter_lists["v"],
        regime_change;
        random_state = rand(rng, 0:999_999),
        gaussian_draws = gaussian_draws,
        poisson_draws = poisson_draws,
    )

    prices = Vector{Float64}(undef, n_steps)
    previous, cumulative = NaN, 0.0
    for i in eachindex(log_returns)
        isnan(log_returns[i]) || (previous = log_returns[i])
        if isnan(previous)
            prices[i] = NaN
        else
            cumulative += previous
            prices[i] = 100 * exp(cumulative)
        end
    end
    return prices, regime_path
end

"""
    parallel_generate_prices(number_of_paths, regimes, transition_matrix, total_time, n_steps; random_state=nothing) -> (prices, regimes)

Generate `number_of_paths` independent Heston–Merton price paths, returned as
`n_steps × number_of_paths` matrices of prices and regime names. Stochastic.
Mirrors Python's `parallel_generate_prices` (run serially; the `n_jobs`
parallelism is a deliberate divergence).
"""
function parallel_generate_prices(
    number_of_paths::Integer,
    regimes::AbstractDict,
    transition_matrix::AbstractMatrix{<:Real},
    total_time::Real,
    n_steps::Integer;
    random_state = nothing,
)
    rng = _synth_rng(random_state)
    seeds = rand(rng, 0:(10*number_of_paths-1), number_of_paths)
    prices = Matrix{Float64}(undef, n_steps, number_of_paths)
    regimes_out = Matrix{eltype(keys(regimes))}(undef, n_steps, number_of_paths)
    for k = 1:number_of_paths
        path_prices, path_regimes = generate_prices_from_regimes(
            regimes,
            transition_matrix,
            total_time,
            n_steps;
            random_state = seeds[k],
        )
        prices[:, k] = path_prices
        regimes_out[:, k] = path_regimes
    end
    return prices, regimes_out
end
