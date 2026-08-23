"""
    RiskLabAI.Pde

PDE submodule, mirroring the Python `RiskLabAI.pde` sub-package: financial PDEs
solved by the Deep BSDE method (Han, Jentzen & E, 2018).

The base module wires the **equations** — the forward-SDE sampler and the BSDE
generator functions. The neural Deep-BSDE solver is supplied only by the
optional extension because it requires a deep-learning backend.
"""
module Pde

include("Equations.jl")

"""
    solve_deep_bsde(args...; kwargs...)

Solve a PDE with the optional neural Deep-BSDE backend. The base package keeps
the PDE equations available without loading a deep-learning stack. Installing
and loading the `deep_bsde` extension adds the equation-specific method.
"""
function solve_deep_bsde(args...; kwargs...)
    throw(
        ArgumentError(
            "solve_deep_bsde requires the optional deep_bsde extension " *
            "(Lux, Optimisers, and Zygote).",
        ),
    )
end

export
    Equation,
    HJBLQ,
    BlackScholesBarenblatt,
    PricingDefaultRisk,
    PricingDiffRate,
    pde_sample,
    pde_driver,
    pde_hamiltonian,
    pde_terminal,
    pde_sigma,
    solve_deep_bsde

end # module Pde
