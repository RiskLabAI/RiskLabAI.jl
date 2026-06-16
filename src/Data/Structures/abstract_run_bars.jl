# Run bars (Fixed + Expected): sample when the cumulative buy- or sell-run
# crosses the dynamic threshold E[T]·max(P_buy·E[b_buy], (1-P_buy)·E[b_sell]).
# Mirrors RiskLabAI.py data/structures/abstract_run_bars.py. `using` centralized
# in `Data`.

@field_inherit AbstractRunBars{T<:Metric} AbstractRunBarsType{T} AbstractInformationDrivenBars{T} where {T<:Metric} begin
    cumulative_buy_theta::Float64
    cumulative_sell_theta::Float64
    expected_buy_imbalance::Float64
    expected_sell_imbalance::Float64
    expected_buy_ticks_proportion::Float64
    buy_ticks_number::Int
    previous_bars_number_of_ticks::Vector{Int}
    previous_tick_imbalances_buy::Vector{Float64}
    previous_tick_imbalances_sell::Vector{Float64}
    previous_bars_buy_ticks_proportions::Vector{Float64}
    analyse_thresholds::Union{Vector,Nothing}
end

function AbstractRunBars{T}(;
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
        cumulative_buy_theta = 0.0,
        cumulative_sell_theta = 0.0,
        expected_buy_imbalance = NaN,
        expected_sell_imbalance = NaN,
        expected_buy_ticks_proportion = NaN,
        buy_ticks_number = 0,
        previous_bars_number_of_ticks = Int[],
        previous_tick_imbalances_buy = Float64[],
        previous_tick_imbalances_sell = Float64[],
        previous_bars_buy_ticks_proportions = Float64[],
        analyse_thresholds = does_analyse_thresholds ? [] : nothing,
    )
end

function construct_bars_from_data(run_bars::AbstractRunBarsType{T}; data) where {T<:Metric}
    bars_list = Vector{Union{DateTime,Int,Float64}}[]
    tick_counter = run_bars.tick_counter

    for tick_data in eachrow(data)
        tick_counter += 1
        (date_time, price, volume) = tick_data
        price = Float64(price)
        volume = Float64(volume)

        signed_tick = tick_rule(run_bars, price)
        update_base_fields(run_bars, price, signed_tick, volume)

        imbalance = imbalance_at_tick(T, price, signed_tick, volume)
        if imbalance > 0
            push!(run_bars.previous_tick_imbalances_buy, imbalance)
            run_bars.cumulative_buy_theta += imbalance
            run_bars.buy_ticks_number += 1
        elseif imbalance < 0
            push!(run_bars.previous_tick_imbalances_sell, -imbalance)
            run_bars.cumulative_sell_theta += -imbalance
        end

        # Warm-up E[b_buy], E[b_sell], P[buy] until all three are available.
        if isnan(run_bars.expected_buy_imbalance) ||
           isnan(run_bars.expected_sell_imbalance) ||
           isnan(run_bars.expected_buy_ticks_proportion)
            run_bars.expected_buy_imbalance = ewma_expected_imbalance(
                run_bars, run_bars.previous_tick_imbalances_buy,
                run_bars.expected_imbalance_window; warm_up = true,
            )
            run_bars.expected_sell_imbalance = ewma_expected_imbalance(
                run_bars, run_bars.previous_tick_imbalances_sell,
                run_bars.expected_imbalance_window; warm_up = true,
            )
            if run_bars.cumulative_ticks > 0
                run_bars.expected_buy_ticks_proportion =
                    run_bars.buy_ticks_number / run_bars.cumulative_ticks
            end
        end

        threshold = calculate_run_threshold(run_bars)

        if bar_construction_condition(run_bars, threshold)
            next_bar = construct_next_bar(
                run_bars, date_time, tick_counter, price,
                run_bars.high_price, run_bars.low_price, threshold,
            )
            push!(bars_list, next_bar)

            cumulative_ticks = Int(run_bars.cumulative_ticks)
            push!(run_bars.previous_bars_number_of_ticks, cumulative_ticks)
            buy_proportion = cumulative_ticks > 0 ?
                run_bars.buy_ticks_number / cumulative_ticks : 0.0
            push!(run_bars.previous_bars_buy_ticks_proportions, buy_proportion)

            run_bars.expected_ticks_number = expected_number_of_ticks(run_bars)

            window = isnothing(run_bars.window_size_for_expected_n_ticks_estimation) ?
                run_bars.expected_imbalance_window :
                run_bars.window_size_for_expected_n_ticks_estimation
            proportions = run_bars.previous_bars_buy_ticks_proportions
            run_bars.expected_buy_ticks_proportion =
                ewma(collect(Float64, last(proportions, window)), window)[end]

            run_bars.expected_buy_imbalance = ewma_expected_imbalance(
                run_bars, run_bars.previous_tick_imbalances_buy,
                run_bars.expected_imbalance_window,
            )
            run_bars.expected_sell_imbalance = ewma_expected_imbalance(
                run_bars, run_bars.previous_tick_imbalances_sell,
                run_bars.expected_imbalance_window,
            )

            reset_cached_fields(run_bars)
        end
    end

    return bars_list
end

function calculate_run_threshold(run_bars::AbstractRunBarsType)::Float64
    expected_ticks = run_bars.expected_ticks_number
    expected_buy_proportion = run_bars.expected_buy_ticks_proportion
    expected_buy_imbalance = run_bars.expected_buy_imbalance
    expected_sell_imbalance = run_bars.expected_sell_imbalance

    if isnan(expected_ticks) || isnan(expected_buy_proportion) ||
       isnan(expected_buy_imbalance) || isnan(expected_sell_imbalance)
        return Inf
    end

    buy_threshold = expected_buy_proportion * expected_buy_imbalance
    sell_threshold = (1 - expected_buy_proportion) * expected_sell_imbalance
    return expected_ticks * max(buy_threshold, sell_threshold)
end

function bar_construction_condition(run_bars::AbstractRunBarsType, threshold::Float64)::Bool
    (isinf(threshold) || isnan(threshold)) && return false
    max_theta = max(run_bars.cumulative_buy_theta, run_bars.cumulative_sell_theta)
    return max_theta ≥ threshold
end

function reset_cached_fields(run_bars::AbstractRunBarsType)
    invoke(reset_cached_fields, Tuple{AbstractBarsType}, run_bars)
    run_bars.cumulative_buy_theta = 0.0
    run_bars.cumulative_sell_theta = 0.0
    run_bars.buy_ticks_number = 0
end
