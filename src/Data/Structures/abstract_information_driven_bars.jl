# Base for information-driven bars (imbalance + run). Mirrors RiskLabAI.py
# data/structures/abstract_information_driven_bars.py. `using` centralized in
# the parent `Data` module (was: ResumableFunctions/CSV/Parameters — unused).

@field_inherit AbstractInformationDrivenBars{T<:Metric} AbstractInformationDrivenBarsType{T} AbstractBars begin
    expected_ticks_number::Float64
    expected_imbalance_window::Int
    window_size_for_expected_n_ticks_estimation::Union{Int,Nothing}
end

"""
Initial field values (NamedTuple, in declaration order) for an
information-driven bar; concrete constructors splat these ahead of their own
fields.
"""
function AbstractInformationDrivenBars{T}(;
    bar_type::String,
    window_size_for_expected_n_ticks_estimation::Union{Int,Nothing},
    initial_estimate_of_expected_n_ticks_in_bar::Float64,
    expected_imbalance_window::Int,
) where {T<:Metric}
    base = AbstractBars(bar_type)
    return (
        base...,
        expected_ticks_number = initial_estimate_of_expected_n_ticks_in_bar,
        expected_imbalance_window = expected_imbalance_window,
        window_size_for_expected_n_ticks_estimation = window_size_for_expected_n_ticks_estimation,
    )
end

"""
EWMA of the expected imbalance E[b] over the most recent `window` observed
imbalances (AFML p.29). With `warm_up`, returns `NaN` until at least E[T]
ticks have been seen.
"""
function ewma_expected_imbalance(
    bars::AbstractInformationDrivenBarsType,
    array::AbstractVector{<:Real},
    window::Int;
    warm_up::Bool = false,
)::Float64
    if warm_up &&
       (isnan(bars.expected_ticks_number) || length(array) < bars.expected_ticks_number)
        return NaN
    end
    ewma_window = Int(min(length(array), window))
    ewma_window == 0 && return NaN
    return ewma(collect(Float64, last(array, ewma_window)), ewma_window)[end]
end

# Per-tick imbalance θ_t, dispatched on the metric (AFML p.29).
imbalance_at_tick(
    ::Type{Tick},
    price::Float64,
    signed_tick::Float64,
    volume::Float64,
)::Float64 = signed_tick
imbalance_at_tick(
    ::Type{Volume},
    price::Float64,
    signed_tick::Float64,
    volume::Float64,
)::Float64 = signed_tick * volume
imbalance_at_tick(
    ::Type{Dollar},
    price::Float64,
    signed_tick::Float64,
    volume::Float64,
)::Float64 = signed_tick * volume * price
