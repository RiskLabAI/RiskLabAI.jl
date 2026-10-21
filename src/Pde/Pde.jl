"""
    RiskLabAI.Pde

PDE submodule, mirroring the Python `RiskLabAI.pde` sub-package: financial PDEs
solved by the Deep BSDE method (Han, Jentzen & E, 2018).

The module provides forward-SDE samplers, generator and terminal functions,
native neural components and fixed-coordinate constrained neural SDEs.
"""
module Pde

include("Equations.jl")
include("EquationContracts.jl")

include("DeepBSDESolver.jl")
include("NeuralSDEGeometry.jl")
include("NeuralSDEModels.jl")
include("NeuralSDETraining.jl")
include("NetworkPrimitives.jl")
include("SetNetworks.jl")
include("TimeDependentNetworks.jl")
include("NeuralPDESolvers.jl")
include("NeuralPDETraining.jl")

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
    pde_terminal_for_sample,
    pde_sigma,
    solve_deep_bsde,
    normalized_call_grid_constraints,
    normalized_call_grid_witness,
    affine_price_state_interval,
    IntervalNeuralSDE,
    PolytopeNeuralSDE,
    negative_log_likelihood,
    fit_interval_neural_sde,
    fit_polytope_neural_sde,
    simulate_interval_neural_sde,
    simulate_polytope_neural_sde,
    hjm_drift_residual,
    Net1,
    TimeNet,
    FBSNNNetwork,
    DeepBSDE,
    TimeNetForSet,
    MAB,
    SAB,
    ISAB,
    PMA,
    DeepTimeSetTransformer,
    TimeDependentNetwork,
    TimeDependentNetworkMonteCarlo,
    initialize_weights,
    FBSDESolver,
    FBSNNSolver,
    compute_pde_loss,
    solve_pde

end # module Pde
