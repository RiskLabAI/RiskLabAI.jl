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

# Fractional differentiation (AFML Ch. 5).
include("Differentiation/Differentiation.jl")

# Sample weighting: uniqueness, return attribution, time decay (AFML Ch. 4).
include("Weights/SampleWeight.jl")

# Covariance denoising via Random Matrix Theory (AFML Ch. 2).
include("Denoise/Denoising.jl")

export
    # metric taxonomy
    Metric, Dollar, Volume, Tick,
    # bar types
    AbstractBars, StandardBars, TimeBars,
    ExpectedImbalanceBars, FixedImbalanceBars,
    ExpectedRunBars, FixedRunBars,
    # construction API
    construct_bars_from_data, bar_construction_condition,
    # differentiation
    calculate_weights_std, calculate_weights_ffd,
    fractional_difference_std, fractional_difference_fixed,
    find_optimal_ffd, fractionally_differentiated_log_price,
    # sample weights
    expand_label_for_meta_labeling, calculate_average_uniqueness,
    sample_weight_absolute_return_meta_labeling, calculate_time_decay,
    # denoising
    marcenko_pastur_pdf, pca, cov_to_corr, corr_to_cov, denoised_corr,
    find_max_eval, denoise_cov, optimal_portfolio, optimal_portfolio_denoised

end # module Data
