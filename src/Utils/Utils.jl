"""
    RiskLabAI.Utils

Shared, dependency-light helpers used across the package: column/stat name
constants, the exponentially-weighted moving average (`ewma`), and the
`@field_inherit` macro used to build the bar-type struct hierarchy.

This is the first submodule of the Phase-2 reconstruction; it mirrors
`RiskLabAI.utils` on the Python side. Further submodules (`Data`, `Backtest`,
`Features`, `Optimization`, `HPC`) are wired in subsequent PRs.
"""
module Utils

include("constants.jl")
include("ewma.jl")
include("field_inheritance.jl")

export ewma
export @field_inherit

export DATE_TIME, TIMESTAMP, TICK_NUMBER, OPEN_PRICE, HIGH_PRICE, LOW_PRICE,
    CLOSE_PRICE, CUMULATIVE_TICKS, CUMULATIVE_DOLLAR, THRESHOLD,
    CUMULATIVE_VOLUME, CUMULATIVE_BUY_VOLUME, CUMULATIVE_SELL_VOLUME,
    CUMULATIVE_THETA, CUMULATIVE_BUY_THETA, CUMULATIVE_SELL_THETA,
    EXPECTED_IMBALANCE, EXPECTED_TICKS_NUMBER, EXPECTED_BUY_IMBALANCE,
    EXPECTED_SELL_IMBALANCE, EXPECTED_BUY_TICKS_PROPORTION, BUY_TICKS_NUMBER,
    N_TICKS_ON_BAR_FORMATION, PREVIOUS_TICK_RULE, EXPECTED_IMBALANCE_WINDOW,
    PREVIOUS_BARS_N_TICKS_LIST, PREVIOUS_TICK_IMBALANCES_LIST,
    PREVIOUS_TICK_IMBALANCES_BUY_LIST, PREVIOUS_TICK_IMBALANCES_SELL_LIST,
    PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST,
    N_PREVIOUS_BARS_FOR_EXPECTED_N_TICKS_ESTIMATION

end # module Utils
