"""
    RiskLabAI.Data

Data-processing submodule, mirroring the Python `RiskLabAI.data` sub-package.

This PR wires `Data.Structures` for **standard bars** (the parametric
`Metric`-dispatched taxonomy + the shared `AbstractBars` base). Time,
imbalance, and run bars — and the rest of the `data` sub-package — are wired in
subsequent PRs.
"""
module Data

using ..Utils                # @field_inherit, constants
using Dates, DataFrames

# Order matters: abstract type taxonomy, then the base struct, then concretes.
include("Structures/types.jl")
include("Structures/abstract_bars.jl")
include("Structures/standard_bars.jl")
include("Structures/time_bars.jl")
include("Structures/abstract_information_driven_bars.jl")
include("Structures/abstract_imbalance_bars.jl")
include("Structures/imbalance_bars.jl")
include("Structures/abstract_run_bars.jl")
include("Structures/run_bars.jl")

export
    # metric taxonomy
    Metric, Dollar, Volume, Tick,
    # bar types
    AbstractBars, StandardBars, TimeBars,
    ExpectedImbalanceBars, FixedImbalanceBars,
    ExpectedRunBars, FixedRunBars,
    # construction API
    construct_bars_from_data, bar_construction_condition

end # module Data
