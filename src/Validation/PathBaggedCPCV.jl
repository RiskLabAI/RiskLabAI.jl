"""
Path-level Bagged Combinatorial Purged Cross-Validation PBO (Arian–Norouzi–Seco
2024). de Prado's CPCV / CSCV estimates the Probability of Backtest Overfitting
(PBO) from a finite set of combinatorial backtest paths, so the estimate carries
estimation variance. Bagging the PATHS — taking a moving-block bootstrap of the
performance series, recomputing the CSCV PBO on each resample, and averaging —
reduces that variance, giving a more accurate PBO estimate when the path set is
small or noisy, and converging to plain CPCV in the data-rich limit.

This is the path-level mechanism; it is distinct from, and does not touch, the
existing bagged-*estimator* cross-validators. The plain-CPCV baseline is the repo's
CSCV PBO (`Backtest.probability_of_backtest_overfitting`), called on each resample.

Deliberate divergence (behavioural): the moving-block bootstrap uses Julia's `rng`
rather than NumPy's PCG64 stream, so the bagged PBO is reproducible under a given
`rng` but not bit-identical to the Python reference (the plain CSCV PBO it builds
on is parity-tested exactly). Admitted in Appraisal 09
(`library_extension/appraisals/09_verdict.md`; in-house method, COI, held to the
identical bar).

Reference: Arian, H., Norouzi, M. L. & Seco, L. (2024). Bagged and Adaptive
Combinatorial Purged Cross-Validation. Bailey, Borwein, López de Prado & Zhu
(2017), Journal of Computational Finance 20(4).
"""

using ..Backtest: probability_of_backtest_overfitting
using Statistics: mean
using Random: AbstractRNG, MersenneTwister

"""
    moving_block_bootstrap_indices(n_observations, block_size, rng) -> Vector{Int}

Moving-block bootstrap row indices (1-based) for a length-`n_observations` series:
random block start points each contribute `block_size` consecutive wrap-around
rows, truncated to `n_observations`. Blocks preserve the serial structure. Mirrors
Python's `moving_block_bootstrap_indices` (which returns 0-based indices).
"""
function moving_block_bootstrap_indices(
    n_observations::Integer,
    block_size::Integer,
    rng::AbstractRNG,
)
    block_size = max(Int(block_size), 1)
    n_blocks = cld(n_observations, block_size)
    starts = rand(rng, 0:(n_observations-1), n_blocks)
    out = Int[]
    for s in starts, o = 0:(block_size-1)
        push!(out, (s + o) % n_observations)
    end
    return out[1:n_observations] .+ 1
end

"""
    bagged_probability_of_backtest_overfitting(performances; n_partitions=16, n_bag=30,
        block_size=nothing, risk_free_return=0.0, metric=nothing, rng=MersenneTwister(0))

Bagged (path-level) Probability of Backtest Overfitting: takes `n_bag` moving-block
bootstrap resamples of the `T×N` `performances` matrix, computes the CSCV PBO on
each, and returns `(bagged_pbo, per_resample_pbos)` — a lower-variance estimate
than a single CPCV PBO. `block_size` defaults to `max(T ÷ 20, 5)`.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
prefer path-level Bagged CPCV over plain CPCV for a more accurate, lower-variance
overfitting (PBO) estimate whenever the CPCV path set is small or noisy; it
converges to plain CPCV in the data-rich limit and is neutral on which model is
selected.

Mirrors Python's `bagged_probability_of_backtest_overfitting`.
"""
function bagged_probability_of_backtest_overfitting(
    performances::AbstractMatrix{<:Real};
    n_partitions::Integer = 16,
    n_bag::Integer = 30,
    block_size::Union{Integer,Nothing} = nothing,
    risk_free_return::Real = 0.0,
    metric = nothing,
    rng::AbstractRNG = MersenneTwister(0),
)
    P = float.(performances)
    t_len = size(P, 1)
    bs = block_size === nothing ? max(t_len ÷ 20, 5) : block_size
    pbos = Float64[]
    for _ = 1:n_bag
        rows = moving_block_bootstrap_indices(t_len, bs, rng)
        pbo, _ = probability_of_backtest_overfitting(
            P[rows, :];
            n_partitions = n_partitions,
            risk_free_return = risk_free_return,
            metric = metric,
        )
        push!(pbos, pbo)
    end
    return (mean(pbos), pbos)
end
