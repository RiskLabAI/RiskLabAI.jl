"""
PDE equations — native Julia port mirroring the Python `RiskLabAI.pde.equation`
module: the forward-SDE sampler and the BSDE generator functions (driver `f`,
Hamiltonian `H(z)`, terminal payoff) for four financial PDEs solved by the Deep
BSDE method (Han, Jentzen & E, 2018). The neural Deep-BSDE solver itself is wired
separately (it needs a deep-learning backend).

Parity notes: the generator functions (`pde_driver`, `pde_hamiltonian`,
`pde_terminal`, `pde_sigma`) are **deterministic** and match Python exactly
(verified in `test/runtests.jl`). `pde_sample` is **stochastic** (Euler–Maruyama
forward simulation) and validated structurally.

Representation note (deliberate divergence): torch tensors become `Matrix`es with
shape `batch × dim`; `pde_sample` returns dense `num_sample × dim × steps` arrays.

Reference: Han, J., Jentzen, A., E, W. (2018), *Solving high-dimensional PDEs
using deep learning*, PNAS. Based on https://github.com/frankhan91/DeepBSDE.
"""

using Random: default_rng

abstract type Equation end

_relu(v) = max.(v, 0.0)

# Common config accessors.
delta_t(eq::Equation) = eq.delta_t
sqrt_delta_t(eq::Equation) = eq.sqrt_delta_t

"""
    HJBLQ(dim, total_time, num_time_interval)

Hamilton–Jacobi–Bellman equation with linear-quadratic control.
"""
struct HJBLQ <: Equation
    dim::Int
    total_time::Float64
    num_time_interval::Int
    delta_t::Float64
    sqrt_delta_t::Float64
    x_init::Vector{Float64}
    sigma::Float64
    lambd::Float64
end

function HJBLQ(dim::Integer, total_time::Real, num_time_interval::Integer)
    dt = total_time / num_time_interval
    return HJBLQ(dim, total_time, num_time_interval, dt, sqrt(dt), zeros(dim), sqrt(2.0), 1.0)
end

pde_sigma(eq::HJBLQ, x) = eq.sigma
pde_driver(eq::HJBLQ, t, x, y, z) = zeros(size(x, 1), 1)
pde_hamiltonian(eq::HJBLQ, t, x, y, z) = sum(z .^ 2; dims = 2) ./ eq.sigma^2
pde_terminal(eq::HJBLQ, t, x) = log.(0.5 .* (1 .+ sum(x .^ 2; dims = 2)))

"""
    BlackScholesBarenblatt(dim, total_time, num_time_interval)

Black–Scholes–Barenblatt equation.
"""
struct BlackScholesBarenblatt <: Equation
    dim::Int
    total_time::Float64
    num_time_interval::Int
    delta_t::Float64
    sqrt_delta_t::Float64
    x_init::Vector{Float64}
    sigma::Float64
    rate::Float64
    mu_bar::Float64
end

function BlackScholesBarenblatt(dim::Integer, total_time::Real, num_time_interval::Integer)
    dt = total_time / num_time_interval
    x_init = [1.0 / (1.0 + (i - 1) % 2) for i = 1:dim]
    return BlackScholesBarenblatt(
        dim, total_time, num_time_interval, dt, sqrt(dt), x_init, 0.4, 0.05, 0.0,
    )
end

pde_sigma(eq::BlackScholesBarenblatt, x) = eq.sigma .* x
pde_driver(eq::BlackScholesBarenblatt, t, x, y, z) = fill(eq.rate, size(x, 1), 1)
pde_hamiltonian(eq::BlackScholesBarenblatt, t, x, y, z) =
    -sum(z; dims = 2) .* eq.rate ./ eq.sigma
pde_terminal(eq::BlackScholesBarenblatt, t, x) = sum(x .^ 2; dims = 2)

"""
    PricingDefaultRisk(dim, total_time, num_time_interval)

Nonlinear pricing PDE with default risk (piecewise-linear hazard rate).
"""
struct PricingDefaultRisk <: Equation
    dim::Int
    total_time::Float64
    num_time_interval::Int
    delta_t::Float64
    sqrt_delta_t::Float64
    x_init::Vector{Float64}
    sigma::Float64
    rate::Float64
    delta::Float64
    gammah::Float64
    gammal::Float64
    mu_bar::Float64
    vh::Float64
    vl::Float64
    slope::Float64
end

function PricingDefaultRisk(dim::Integer, total_time::Real, num_time_interval::Integer)
    dt = total_time / num_time_interval
    gammah, gammal, vh, vl = 0.2, 0.02, 50.0, 70.0
    return PricingDefaultRisk(
        dim, total_time, num_time_interval, dt, sqrt(dt), ones(dim) .* 100.0,
        0.2, 0.02, 2.0 / 3, gammah, gammal, 0.02, vh, vl, (gammah - gammal) / (vh - vl),
    )
end

pde_sigma(eq::PricingDefaultRisk, x) = eq.sigma .* x
function pde_driver(eq::PricingDefaultRisk, t, x, y, z)
    piecewise = _relu(_relu(y .- eq.vh) .* eq.slope .+ eq.gammah .- eq.gammal) .+ eq.gammal
    return (1 - eq.delta) .* piecewise .+ eq.rate
end
pde_hamiltonian(eq::PricingDefaultRisk, t, x, y, z) = zeros(size(x, 1), 1)
pde_terminal(eq::PricingDefaultRisk, t, x) = _relu(minimum(x; dims = 2))

"""
    PricingDiffRate(dim, total_time, num_time_interval)

Nonlinear Black–Scholes with different borrowing/lending rates.
"""
struct PricingDiffRate <: Equation
    dim::Int
    total_time::Float64
    num_time_interval::Int
    delta_t::Float64
    sqrt_delta_t::Float64
    x_init::Vector{Float64}
    sigma::Float64
    mu_bar::Float64
    rl::Float64
    rb::Float64
    alpha::Float64
end

function PricingDiffRate(dim::Integer, total_time::Real, num_time_interval::Integer)
    dt = total_time / num_time_interval
    return PricingDiffRate(
        dim, total_time, num_time_interval, dt, sqrt(dt), ones(dim) .* 100.0,
        0.2, 0.06, 0.04, 0.06, 1.0 / dim,
    )
end

pde_sigma(eq::PricingDiffRate, x) = eq.sigma .* x
function pde_driver(eq::PricingDiffRate, t, x, y, z)
    temp = sum(z; dims = 2) ./ eq.sigma .- y
    return ifelse.(temp .> 0, eq.rb, eq.rl)
end
function pde_hamiltonian(eq::PricingDiffRate, t, x, y, z)
    temp = sum(z; dims = 2) ./ eq.sigma .- y
    sum_z = sum(z; dims = 2)
    return ifelse.(
        temp .> 0,
        (eq.mu_bar - eq.rb) .* sum_z ./ eq.sigma,
        (eq.mu_bar - eq.rl) .* sum_z ./ eq.sigma,
    )
end
function pde_terminal(eq::PricingDiffRate, t, x)
    temp = maximum(x; dims = 2)
    return _relu(temp .- 120) .- 2 .* _relu(temp .- 150)
end

"""
    pde_sample(eq, num_sample; rng=default_rng()) -> (dw, x)

Euler–Maruyama simulation of the forward SDE: returns Wiener increments
`dw` (`num_sample × dim × num_time_interval`) and the asset paths `x`
(`num_sample × dim × num_time_interval+1`). Stochastic. Mirrors Python's
`Equation.sample`.
"""
function pde_sample(eq::Equation, num_sample::Integer; rng = default_rng())
    dw = randn(rng, num_sample, eq.dim, eq.num_time_interval) .* eq.sqrt_delta_t
    x = zeros(num_sample, eq.dim, eq.num_time_interval + 1)
    x[:, :, 1] .= reshape(eq.x_init, 1, eq.dim)
    for i = 1:eq.num_time_interval
        x[:, :, i+1] = _sample_step(eq, x[:, :, i], dw[:, :, i])
    end
    return dw, x
end

_sample_step(eq::HJBLQ, xi, dwi) = xi .+ eq.sigma .* dwi
_sample_step(eq::BlackScholesBarenblatt, xi, dwi) =
    (1 + eq.mu_bar * eq.delta_t) .* xi .+ (eq.sigma .* xi) .* dwi
_sample_step(eq::PricingDefaultRisk, xi, dwi) =
    (1 + eq.mu_bar * eq.delta_t) .* xi .+ (eq.sigma .* xi) .* dwi
function _sample_step(eq::PricingDiffRate, xi, dwi)
    factor = exp((eq.mu_bar - (eq.sigma^2) / 2) * eq.delta_t)
    return (factor .* exp.(eq.sigma .* dwi)) .* xi
end
