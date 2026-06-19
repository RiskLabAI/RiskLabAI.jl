# Base struct + shared logic for every bar type (standard, time, imbalance,
# run). Mirrors RiskLabAI.py `data/structures/abstract_bars.py`.
# `using` statements are centralized in the parent `Data` module.

"""
    AbstractBars

Base mutable struct holding the properties shared by all bar subtypes
(`StandardBars`, `TimeBars`, and the information-driven bars). Concrete bar
types inherit these fields via `@field_inherit`.
"""
mutable struct AbstractBars <: AbstractBarsType
    # Base properties
    bar_type::String
    tick_counter::Int

    # Cache properties (current bar in progress)
    previous_tick_price::Float64

    open_price::Float64
    close_price::Float64

    low_price::Float64
    high_price::Float64

    previous_tick_rule::Float64
    cumulative_ticks::Float64
    cumulative_dollar::Float64
    cumulative_volume::Float64
    cumulative_buy_volume::Float64
    n_ticks_on_bar_formation::Int
end

"""
    AbstractBars(bar_type::String) -> NamedTuple

Return the initial values for the shared base fields, in declaration order, so a
concrete constructor can splat them ahead of its own extra fields.
"""
function AbstractBars(bar_type::String)::NamedTuple
    return (
        bar_type = bar_type,
        tick_counter = 0,
        previous_tick_price = NaN,
        open_price = NaN,
        close_price = NaN,
        # Seed high/low to -Inf/+Inf so the first bar's max()/min() work
        # (NaN would propagate through max/min). Matches reset_cached_fields
        # and the Python implementation.
        low_price = Inf,
        high_price = -Inf,
        previous_tick_rule = 0.0,
        cumulative_ticks = 0.0,
        cumulative_dollar = 0.0,
        cumulative_volume = 0.0,
        cumulative_buy_volume = 0.0,
        n_ticks_on_bar_formation = 0,
    )
end

"""
Update the shared base fields with the current tick's price, tick rule and
volume.
"""
function update_base_fields(
    bars::AbstractBarsType,
    price::Float64,
    tick_rule::Float64,
    volume::Float64,
)
    dollar_value = price * volume

    bars.open_price = isnan(bars.open_price) ? price : bars.open_price
    bars.high_price, bars.low_price = high_and_low_price_update(bars, price)
    bars.cumulative_ticks += 1

    bars.cumulative_dollar += dollar_value
    bars.cumulative_volume += volume

    if tick_rule == 1
        bars.cumulative_buy_volume += volume
    end
end

"""
Reset the cached fields after a bar is sampled.
"""
function reset_cached_fields(bars::AbstractBarsType)
    bars.open_price = NaN
    bars.high_price, bars.low_price = -Inf, +Inf

    bars.cumulative_ticks = 0
    bars.cumulative_dollar = 0
    bars.cumulative_volume = 0
    bars.cumulative_buy_volume = 0
end

"""
Compute the tick rule (signed tick), AFML p.29. Carries forward the previous
non-zero sign when the price is unchanged.
"""
function tick_rule(bars::AbstractBarsType, price::Float64)::Float64
    tick_difference = isnan(bars.previous_tick_price) ? 0 : price - bars.previous_tick_price

    if tick_difference != 0
        bars.previous_tick_rule = signed_tick = sign(tick_difference)
    else
        signed_tick = bars.previous_tick_rule
    end

    bars.previous_tick_price = price
    return signed_tick
end

"""
Update the running high/low with the current tick price.
"""
function high_and_low_price_update(
    bars::AbstractBarsType,
    price::Float64,
)::Tuple{Float64,Float64}
    high_price = max(price, bars.high_price)
    low_price = min(price, bars.low_price)
    return (high_price, low_price)
end

"""
Build the next bar row:
`[date_time, tick_index, open, high, low, close, cumulative_volume,
cumulative_buy_volume, cumulative_sell_volume, cumulative_ticks,
cumulative_dollar_value, threshold]` (same column order as the Python package).
"""
function construct_next_bar(
    bars::AbstractBarsType,
    date_time::Union{DateTime,String},
    tick_index::Int,
    price::Float64,
    high_price::Float64,
    low_price::Float64,
    threshold::Float64,
)::Vector{Union{DateTime,Int,Float64}}
    open_price = bars.open_price
    high_price = max(high_price, open_price)
    low_price = min(low_price, open_price)
    close_price = price

    cumulative_volume = bars.cumulative_volume
    cumulative_buy_volume = bars.cumulative_buy_volume
    cumulative_sell_volume = cumulative_volume - cumulative_buy_volume
    cumulative_ticks = bars.cumulative_ticks
    cumulative_dollar_value = bars.cumulative_dollar

    return Union{DateTime,Int,Float64}[
        date_time,
        tick_index,
        open_price,
        high_price,
        low_price,
        close_price,
        cumulative_volume,
        cumulative_buy_volume,
        cumulative_sell_volume,
        cumulative_ticks,
        cumulative_dollar_value,
        threshold,
    ]
end
