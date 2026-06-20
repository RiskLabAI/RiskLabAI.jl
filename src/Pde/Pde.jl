"""
    RiskLabAI.Pde

PDE submodule, mirroring the Python `RiskLabAI.pde` sub-package: financial PDEs
solved by the Deep BSDE method (Han, Jentzen & E, 2018).

This slice wires the **equations** — the forward-SDE sampler and the BSDE
generator functions. The neural Deep-BSDE solver (which needs a deep-learning
backend) is wired in a follow-up.
"""
module Pde

include("Equations.jl")

# Neural Deep-BSDE solver (Lux.jl backend).
include("DeepBSDESolver.jl")

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
