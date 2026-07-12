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
    optional `random_state`/`rng` and are validated structurally.

Deliberate divergence: numba/joblib/quantecon are replaced by native Julia
(plain loops, a native Markov-chain simulate, `Distributions`). The optional
Ledoit–Wolf shrinkage in `simulates_cov_mu` (`shrink=true`) is deferred to the
covariance-estimation backend; the default `shrink=false` (`np.cov`) is exact.
Price generators return a plain price vector (Python returns a business-day
indexed Series).

Reference: De Prado, M. (2020), Machine Learning for Asset Managers, Ch. 2.
"""

using Random: AbstractRNG, MersenneTwister, default_rng, shuffle
using Statistics: mean, cov
using LinearAlgebra: Symmetric, Diagonal
using Distributions: MvNormal, Poisson

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
"""
function random_cov(num_columns::Integer, num_factors::Integer; rng = default_rng())
    w = randn(rng, num_columns, num_factors)
    cov = w * w'
    cov += Diagonal(rand(rng, num_columns))
    return cov
end

"""
    form_true_matrix(n_blocks, block_size, block_correlation; rng=default_rng()) -> (mu, cov)

Shuffled block-diagonal correlation converted to a covariance matrix with random
volatilities, plus a random mean vector. Stochastic. Mirrors Python's
`form_true_matrix`.
"""
function form_true_matrix(
    n_blocks::Integer,
    block_size::Integer,
    block_correlation::Real;
    rng = default_rng(),
)
    correlation = form_block_matrix(n_blocks, block_size, block_correlation)
    order = shuffle(rng, collect(1:size(correlation, 1)))
    correlation = correlation[order, order]
    standard_deviations = rand(rng, size(correlation, 1)) .* 0.15 .+ 0.05  # U(0.05, 0.2)
    covariance = corr_to_cov(correlation, standard_deviations)
    mu = randn(rng, size(covariance, 1)) .* standard_deviations .+ standard_deviations
    return mu, covariance
end

"""
    simulates_cov_mu(mu, covariance, n_observations; shrink=false, rng=default_rng()) -> (mu1, cov1)

Draw `n_observations` multivariate-normal samples from `(mu, covariance)` and
return their sample mean and covariance. Stochastic. Mirrors Python's
`simulates_cov_mu` for the default `shrink=false` path; `shrink=true`
(Ledoit–Wolf) is deferred to the covariance-estimation backend.
"""
function simulates_cov_mu(
    mu::AbstractVecOrMat{<:Real},
    covariance::AbstractMatrix{<:Real},
    n_observations::Integer;
    shrink::Bool = false,
    rng = default_rng(),
)
    shrink && throw(
        ArgumentError(
            "Ledoit–Wolf shrinkage (shrink=true) is deferred to the " *
            "covariance-estimation backend; use shrink=false",
        ),
    )
    samples = permutedims(
        rand(rng, MvNormal(vec(mu), Symmetric(Matrix(covariance))), n_observations),
    )
    mu1 = vec(mean(samples; dims = 1))
    cov1 = cov(samples)
    return mu1, cov1
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
    steps = collect(range(0.0, 1.0; length = bubble_length))
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
)
    all(length(p) == n_steps for p in (mu, kappa, theta, xi, rho, lambda, m, v)) ||
        throw(ArgumentError("all parameter vectors must have length n_steps"))

    rng = _synth_rng(random_state)
    dt = total_time / n_steps
    sqrt_dt = sqrt(dt)

    dw_stock = Vector{Float64}(undef, n_steps)
    dw_vol = Vector{Float64}(undef, n_steps)
    jump_comp = Vector{Float64}(undef, n_steps)
    poisson_var = Vector{Float64}(undef, n_steps)
    for i = 1:n_steps
        block = [1.0 rho[i] 0.0; rho[i] 1.0 0.0; 0.0 0.0 v[i]^2]
        z = rand(rng, MvNormal([0.0, 0.0, m[i]], Symmetric(block)))
        dw_stock[i] = z[1]
        dw_vol[i] = z[2]
        jump_comp[i] = z[3]
        poisson_var[i] = rand(rng, Poisson(lambda[i] * dt))
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

# Native categorical sampler + Markov-chain simulate (replaces quantecon).
function _sample_categorical(rng, probabilities)
    u = rand(rng)
    cumulative = 0.0
    for i in eachindex(probabilities)
        cumulative += probabilities[i]
        u <= cumulative && return i
    end
    return length(probabilities)
end

function _markov_simulate(transition_matrix, n_steps, n_states; rng)
    states = Vector{Int}(undef, n_steps)
    states[1] = rand(rng, 1:n_states)
    for t = 2:n_steps
        states[t] = _sample_categorical(rng, view(transition_matrix, states[t-1], :))
    end
    return states
end

"""
    generate_prices_from_regimes(regimes, transition_matrix, total_time, n_steps; random_state=nothing) -> (prices, regime_path)

Simulate a Markov-switching Heston–Merton price path. `regimes` maps a regime
name to a parameter dictionary (keys `"mu","kappa","theta","xi","rho","lam","m","v"`).
Returns the price vector and the per-step regime names. Stochastic. Mirrors
Python's `generate_prices_from_regimes` (without the business-day index).
"""
function generate_prices_from_regimes(
    regimes::AbstractDict,
    transition_matrix::AbstractMatrix{<:Real},
    total_time::Real,
    n_steps::Integer;
    random_state = nothing,
)
    rng = _synth_rng(random_state)
    regime_names = collect(keys(regimes))
    simulated =
        _markov_simulate(transition_matrix, n_steps, length(regime_names); rng = rng)
    simulated_regimes = [regime_names[s] for s in simulated]

    parameter_keys = ["mu", "kappa", "theta", "xi", "rho", "lam", "m", "v"]
    parameter_lists = Dict(k => Float64[] for k in parameter_keys)
    regime_path = eltype(regime_names)[]

    step = 1
    while step <= n_steps
        name = simulated_regimes[step]
        params, regime_length = align_params_length(regimes[name])
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
    )

    prices = 100.0 .* exp.(cumsum(log_returns))
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
