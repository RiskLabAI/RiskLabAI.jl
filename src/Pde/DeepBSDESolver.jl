"""
Deep BSDE solver — native Julia port mirroring the Python
`RiskLabAI.pde.solver.FBSDESolver` (`"DeepBSDE"` method) on the **Lux.jl**
backend (Han, Jentzen & E, 2018). A separate feed-forward subnetwork per time
step predicts the BSDE control `Z_t`; the initial value `Y_0` and gradient `Z_0`
are trainable scalars/vectors. Training minimises the terminal-condition MSE by
Adam (`Optimisers.jl`) with reverse-mode gradients (`Zygote.jl`).

Parity note: this is a **behavioural** port — a trained network is not
numerically identical to the PyTorch one. It is validated structurally (the
training loss is finite and decreases). The per-step subnetwork is the plain MLP
of the Python `DeepBSDE` module (ReLU between bias-free linear layers; the
commented-out batch-norm is omitted). The PyTorch set-transformer architectures
(`ISAB`/`MAB`/`SAB`/`PMA`/`DeepTimeSetTransformer`) are research scaffolding and
are not ported.

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
per-iteration validation loss and the running `Y_0` estimate. Behavioural.
Mirrors Python's `FBSDESolver(..., "DeepBSDE").solve`.
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
    n_nets = eq.num_time_interval - 1

    models = [_build_subnet(sizes) for _ = 1:n_nets]
    net_params = []
    net_states = []
    for model in models
        ps, st = Lux.setup(rng, model)
        push!(net_params, ps)
        push!(net_states, st)
    end
    states = Tuple(net_states)

    θ = (y0 = [Float64(init_y)], z0 = zeros(1, dim), nets = Tuple(net_params))

    dw_val, x_val = pde_sample(eq, 128; rng = rng)
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), θ)

    losses = Float64[]
    y0_estimates = Float64[]
    for _ = 1:iterations
        dw_train, x_train = pde_sample(eq, batch_size; rng = rng)
        gradient = Zygote.gradient(p -> _bsde_loss(p, eq, x_train, dw_train, models, states), θ)[1]
        opt_state, θ = Optimisers.update(opt_state, θ, gradient)
        push!(losses, _bsde_loss(θ, eq, x_val, dw_val, models, states))
        push!(y0_estimates, θ.y0[1])
    end
    return losses, y0_estimates
end
