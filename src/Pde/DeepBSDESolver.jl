"""
Deep BSDE convenience interface. The public wrapper delegates to the native
`FBSDESolver` and `solve_pde` contract, retaining the public terminal MSE,
bias-free ReLU networks and separate Adam schedules for network weights,
initial value and initial gradient. Shared weights and supplied paths validate
the deterministic objective, gradients and finite-step updates. Julia and
Python random seeds do not imply the same initial weights or Brownian paths.
The earlier internal Lux helpers remain available for compatibility.

Reference: Han, J., Jentzen, A., E, W. (2018), PNAS.
"""

using Random: AbstractRNG, default_rng
using Lux
using Zygote
using Optimisers

_act(x) = max(x, zero(x))

# Per-time-step MLP: ReLU after each hidden bias-free linear layer, raw output.
function _build_subnet(sizes)
    layers = Any[]
    for i = 1:(length(sizes)-2)
        push!(layers, Lux.Dense(sizes[i] => sizes[i+1], _act; use_bias = false))
    end
    push!(layers, Lux.Dense(sizes[end-1] => sizes[end]; use_bias = false))
    return Lux.Chain(layers...)
end

# Terminal-condition MSE of the propagated backward equation.
function _bsde_loss(θ, eq, x, dw, models, states)
    batch = size(x, 1)
    y = ones(batch, 1) .* θ.y0                      # Y_0 (trainable, 1-vector)
    for i = 1:eq.num_time_interval
        s0 = x[:, :, i]
        if i == 1
            out_z = ones(batch, 1) * θ.z0           # Z_0 (trainable, 1×dim)
        else
            net_out, _ = Lux.apply(models[i-1], permutedims(s0), θ.nets[i-1], states[i-1])
            out_z = permutedims(net_out)
        end
        dwi = dw[:, :, i]
        rate = pde_driver(eq, 0.0, s0, y, out_z)
        hamiltonian = pde_hamiltonian(eq, 0.0, s0, y, out_z)
        y = y .* (1 .+ rate .* eq.delta_t) .+ hamiltonian .* eq.delta_t .+
            sum(out_z .* dwi; dims = 2)
    end
    payoff = pde_terminal(eq, eq.total_time, x[:, :, end])
    return sum((payoff .- y) .^ 2) / batch
end

"""
    solve_deep_bsde(eq; hidden_sizes=[16], iterations=20, batch_size=64,
                    init_y=0.0, learning_rate=0.01, rng=default_rng())
        -> (losses, y0_estimates)

Train the Deep BSDE solver for equation `eq` (any `Pde.Equation`). Returns the
per-iteration validation loss and running `Y_0` estimate. The network learning
rate is supplied; initial value and gradient rates remain 0.1 and 0.001, with
public Adam betas `(0.9,0.99)`. Use `FBSDESolver` and `solve_pde` directly for
explicit parameter trees and supplied training/validation batches.
"""
function solve_deep_bsde(
    eq::Equation;
    hidden_sizes::AbstractVector{<:Integer} = [16],
    iterations::Integer = 20,
    batch_size::Integer = 64,
    init_y::Real = 0.0,
    learning_rate::Real = 0.01,
    rng::AbstractRNG = default_rng(),
)
    dim = eq.dim
    sizes = vcat(dim, collect(hidden_sizes), dim)
    model = FBSDESolver(eq, sizes, learning_rate, "DeepBSDE"; seed = rand(rng, UInt32))
    return solve_pde(model; iterations, batch_size, init_y, rng)
end
