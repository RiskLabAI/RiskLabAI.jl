# Imbalance bars (Fixed + Expected): sample when |cumulative imbalance θ|
# crosses the dynamic threshold E[T]·|E[b]|. Mirrors RiskLabAI.py
# data/structures/abstract_imbalance_bars.py. `using` centralized in `Data`
# (dropped ProfileView/TimerOutputs/CSV/ResumableFunctions/Parameters).

@field_inherit AbstractImbalanceBars{T<:Metric} AbstractImbalanceBarsType{T} AbstractInformationDrivenBars{
    T,
} where {T<:Metric} begin
    cumulative_theta::Float64
    expected_imbalance::Float64
    previous_bars_number_of_ticks::Vector{Int}
    previous_tick_imbalances::Vector{Float64}
    analyse_thresholds::Union{Vector,Nothing}
end

function AbstractImbalanceBars{T}(;
    bar_type::String,
    window_size_for_expected_n_ticks_estimation::Union{Int,Nothing},
    expected_imbalance_window::Int,
    initial_estimate_of_expected_n_ticks_in_bar::Float64,
    does_analyse_thresholds::Bool,
) where {T<:Metric}
    base = AbstractInformationDrivenBars{T}(
        bar_type = bar_type,
        window_size_for_expected_n_ticks_estimation = window_size_for_expected_n_ticks_estimation,
        initial_estimate_of_expected_n_ticks_in_bar = initial_estimate_of_expected_n_ticks_in_bar,
        expected_imbalance_window = expected_imbalance_window,
    )
    return (
        base...,
        cumulative_theta = 0.0,
        expected_imbalance = NaN,
        previous_bars_number_of_ticks = Int[],
        previous_tick_imbalances = Float64[],
        analyse_thresholds = does_analyse_thresholds ? [] : nothing,
    )
end

function construct_bars_from_data(
    imbalance_bars::AbstractImbalanceBarsType{T};
    data,
) where {T<:Metric}
    bars_list = Vector{Union{DateTime,Int,Float64}}[]
    tick_counter = imbalance_bars.tick_counter

    n_row = size(data, 1)
    imbalance_bars.previous_tick_imbalances = zeros(Float64, n_row)

    for (k, tick_data) in enumerate(eachrow(data))
        tick_counter += 1
        (date_time, price, volume) = tick_data
        price = Float64(price)
        volume = Float64(volume)

        signed_tick = tick_rule(imbalance_bars, price)
        update_base_fields(imbalance_bars, price, signed_tick, volume)

        imbalance = imbalance_at_tick(T, price, signed_tick, volume)
        imbalance_bars.previous_tick_imbalances[k] = imbalance
        imbalance_bars.cumulative_theta += imbalance

        # First-time warm-up of E[b].
        if isnan(imbalance_bars.expected_imbalance)
            imbalance_bars.expected_imbalance = ewma_expected_imbalance(
                imbalance_bars,
                view(imbalance_bars.previous_tick_imbalances, 1:k),
                imbalance_bars.expected_imbalance_window;
                warm_up = true,
            )
        end

        expected_ticks = imbalance_bars.expected_ticks_number
        expected_imbalance = imbalance_bars.expected_imbalance
        threshold =
            (isnan(expected_ticks) || isnan(expected_imbalance)) ? Inf :
            expected_ticks * abs(expected_imbalance)

        if bar_construction_condition(imbalance_bars, threshold)
            next_bar = construct_next_bar(
                imbalance_bars,
                date_time,
                tick_counter,
                price,
                imbalance_bars.high_price,
                imbalance_bars.low_price,
                threshold,
            )
            push!(bars_list, next_bar)
            push!(
                imbalance_bars.previous_bars_number_of_ticks,
                Int(imbalance_bars.cumulative_ticks),
            )

            # Update E[T] (sub-type specific) and E[b] for the next bar.
            imbalance_bars.expected_ticks_number = expected_number_of_ticks(imbalance_bars)
            imbalance_bars.expected_imbalance = ewma_expected_imbalance(
                imbalance_bars,
                view(imbalance_bars.previous_tick_imbalances, 1:k),
                imbalance_bars.expected_imbalance_window;
                warm_up = false,
            )

            reset_cached_fields(imbalance_bars)
        end
    end

    return bars_list
end

# |θ| ≥ threshold (matches the Python `>=`); a NaN/Inf threshold never samples.
function bar_construction_condition(
    imbalance_bars::AbstractImbalanceBarsType,
    threshold::Float64,
)::Bool
    (isnan(threshold) || isinf(threshold)) && return false
    return abs(imbalance_bars.cumulative_theta) ≥ threshold
end

function reset_cached_fields(imbalance_bars::AbstractImbalanceBarsType)
    invoke(reset_cached_fields, Tuple{AbstractBarsType}, imbalance_bars)
    imbalance_bars.cumulative_theta = 0.0
end
