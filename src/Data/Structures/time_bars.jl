# Time bars: sample a bar every fixed time interval. Mirrors RiskLabAI.py
# `data/structures/time_bars.py` (the previous Julia logic grouped ticks
# incorrectly and used the tick timestamp instead of the bucket boundary; this
# rewrite matches the Python semantics, asserted in the tests).
# `using Dates`/`DataFrames` are centralized in the parent `Data` module.

@field_inherit TimeBars TimeBarsType AbstractBars begin
    resolution_to_n_seconds::Dict{String,Int}
    resolution_type::String
    resolution_units::Int
    threshold_in_seconds::Int
    current_bar_timestamp::Float64       # NaN until the first tick
    current_bar_end_timestamp::Float64   # NaN until the first tick
end

const _RESOLUTION_TO_N_SECONDS =
    Dict("S" => 1, "MIN" => 60, "H" => 3600, "D" => 86400, "W" => 604800)

"""
    TimeBars(; resolution_type::String, resolution_units::Int)

Construct time bars. `resolution_type` is one of `"S"`, `"MIN"`, `"H"`, `"D"`,
`"W"`; `resolution_units` is the number of those units per bar (e.g. 5 + `"MIN"`
= 5-minute bars).
"""
function TimeBars(; resolution_type::String, resolution_units::Int)
    resolution_upper = uppercase(resolution_type)
    if !haskey(_RESOLUTION_TO_N_SECONDS, resolution_upper)
        throw(
            ArgumentError(
                "Invalid resolution_type $(resolution_type); use one of " *
                "$(collect(keys(_RESOLUTION_TO_N_SECONDS))).",
            ),
        )
    end

    threshold_in_seconds = resolution_units * _RESOLUTION_TO_N_SECONDS[resolution_upper]
    base = AbstractBars("time")

    return TimeBars(
        values(base)...,
        copy(_RESOLUTION_TO_N_SECONDS),
        resolution_upper,
        resolution_units,
        threshold_in_seconds,
        NaN,
        NaN,
    )
end

"""
Construct time bars from tick data. Each row provides `(date_time, price,
volume)` with `date_time::DateTime`.
"""
function construct_bars_from_data(time_bars::TimeBars; data)
    bars_list = Vector{Union{DateTime,Int,Float64}}[]

    for row in eachrow(data)
        time_bars.tick_counter += 1
        (date_time, price, volume) = row

        tick_timestamp_seconds = datetime2unix(date_time)
        bar_start_seconds =
            floor(Int, tick_timestamp_seconds / time_bars.threshold_in_seconds) *
            time_bars.threshold_in_seconds

        # Initialize the first bar window.
        if isnan(time_bars.current_bar_timestamp)
            time_bars.current_bar_timestamp = bar_start_seconds
            time_bars.current_bar_end_timestamp =
                bar_start_seconds + time_bars.threshold_in_seconds
        end

        # If the tick crosses into a new window, emit the *previous* bar first.
        if bar_construction_condition(time_bars, tick_timestamp_seconds)
            bar_end_time = unix2datetime(time_bars.current_bar_end_timestamp)
            next_bar = construct_next_bar(
                time_bars,
                bar_end_time,
                time_bars.tick_counter - 1,
                time_bars.close_price,
                time_bars.high_price,
                time_bars.low_price,
                time_bars.current_bar_end_timestamp,
            )
            push!(bars_list, next_bar)

            reset_cached_fields(time_bars)
            time_bars.current_bar_timestamp = bar_start_seconds
            time_bars.current_bar_end_timestamp =
                bar_start_seconds + time_bars.threshold_in_seconds
        end

        # Add the current tick to the (possibly new) bar in progress.
        signed_tick = tick_rule(time_bars, Float64(price))
        update_base_fields(time_bars, Float64(price), signed_tick, Float64(volume))
        time_bars.close_price = Float64(price)
    end

    return bars_list
end

# A bar closes once a tick's timestamp reaches the current bar's end boundary.
function bar_construction_condition(
    time_bars::TimeBarsType,
    tick_timestamp_seconds::Float64,
)::Bool
    return tick_timestamp_seconds ≥ time_bars.current_bar_end_timestamp
end
