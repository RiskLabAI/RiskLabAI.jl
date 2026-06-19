# Standard (threshold) bars: sample a bar when cumulative dollar / volume /
# ticks reaches a fixed threshold. Mirrors RiskLabAI.py
# `data/structures/standard_bars.py`. `using` is centralized in `Data`.

@field_inherit StandardBars{T<:Metric} StandardBarsType{T} AbstractBars begin
    threshold::Float64  # threshold at which to sample a bar
end

"""
    StandardBars{T}(; bar_type::String, threshold::Float64) where {T<:Metric}

Construct standard bars. `T` selects the sampling metric (`Dollar`, `Volume`,
or `Tick`); `threshold` is the sampling threshold.
"""
function StandardBars{T}(; bar_type::String, threshold::Float64) where {T<:Metric}
    base = AbstractBars(bar_type)
    return StandardBars{T}(values(base)..., threshold)
end

"""
Construct standard bars from tick data. `data` is iterated row-by-row; each row
provides `(date_time, price, volume)`.
"""
function construct_bars_from_data(standard_bars::StandardBars{T}; data) where {T<:Metric}
    bars_list = Vector{Union{DateTime,Int,Float64}}[]

    tick_counter = standard_bars.tick_counter
    for row in eachrow(data)
        tick_counter += 1
        (date_time, price, volume) = row

        signed_tick = tick_rule(standard_bars, Float64(price))
        update_base_fields(standard_bars, Float64(price), signed_tick, Float64(volume))

        threshold = standard_bars.threshold
        if bar_construction_condition(standard_bars, threshold)
            next_bar = construct_next_bar(
                standard_bars,
                date_time,
                tick_counter,
                Float64(price),
                standard_bars.high_price,
                standard_bars.low_price,
                threshold,
            )
            push!(bars_list, next_bar)
            reset_cached_fields(standard_bars)
        end
    end

    return bars_list
end

# Sampling condition, dispatched on the metric type.
function bar_construction_condition(
    standard_bars::StandardBars{Dollar},
    threshold::Float64,
)::Bool
    return standard_bars.cumulative_dollar ≥ threshold
end

function bar_construction_condition(
    standard_bars::StandardBars{Volume},
    threshold::Float64,
)::Bool
    return standard_bars.cumulative_volume ≥ threshold
end

function bar_construction_condition(
    standard_bars::StandardBars{Tick},
    threshold::Float64,
)::Bool
    return standard_bars.cumulative_ticks ≥ threshold
end
