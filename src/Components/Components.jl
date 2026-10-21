module Components

using ..Data
using ..Validation
using ..Features
using ..Utils
using DataFrames

"""Optional dispatch interface for caller-defined labelers; define `label(model, data)` methods."""
abstract type AbstractLabeler end

function label end

mutable struct ComponentEntry
    factory::Any
    loader::Any
    metadata::Dict{String,Any}
    accepted_keywords::Union{Nothing,Set{Symbol}}
end

"""Case-insensitive component discovery with canonical names, aliases and native factories."""
struct ComponentRegistry
    name::String
    interface::Any
    entries::Dict{String,ComponentEntry}
    index::Dict{String,String}
end

ComponentRegistry(name::AbstractString; interface = nothing) =
    ComponentRegistry(String(name), interface, Dict{String,ComponentEntry}(), Dict{String,String}())

function _component_entry(registry::ComponentRegistry, key::AbstractString)
    normalized = lowercase(key)
    haskey(registry.index, normalized) || throw(KeyError("$(repr(key)) is not registered in $(registry.name); available: $(join(available_components(registry), ", "))"))
    return registry.entries[registry.index[normalized]]
end

"""Register a native factory, optionally replacing the canonical entry reached through an alias."""
function register_component!(registry::ComponentRegistry, key::AbstractString, factory;
    aliases = String[], metadata = Dict(), override::Bool = false,
    accepted_keywords = nothing, loader = nothing)
    isempty(key) && throw(ArgumentError("component keys must be nonempty"))
    all(alias -> alias isa AbstractString && !isempty(alias), aliases) || throw(ArgumentError("aliases must be nonempty strings"))
    normalized = lowercase(key)
    haskey(registry.index, normalized) && !override && throw(KeyError("component $(repr(key)) is already registered"))
    haskey(registry.index, normalized) && unregister_component!(registry, key)
    accepted = accepted_keywords === nothing ? nothing : Set(Symbol.(accepted_keywords))
    registry.entries[String(key)] = ComponentEntry(factory, loader, Dict{String,Any}(String(k) => v for (k, v) in pairs(metadata)), accepted)
    registry.index[normalized] = String(key)
    for alias in aliases
        registry.index[lowercase(alias)] = String(key)
    end
    return factory
end

"""Register a loader callback returning a native factory on its first use."""
function register_lazy!(registry::ComponentRegistry, key::AbstractString, loader; kwargs...)
    register_component!(registry, key, nothing; loader = loader, kwargs...)
    return registry
end

"""Remove one registry entry and every alias pointing to its canonical name."""
function unregister_component!(registry::ComponentRegistry, key::AbstractString)
    _component_entry(registry, key)
    canonical = registry.index[lowercase(key)]
    delete!(registry.entries, canonical)
    filter!(pair -> last(pair) != canonical, registry.index)
    return registry
end

function component_factory(registry::ComponentRegistry, key::AbstractString)
    entry = _component_entry(registry, key)
    if entry.factory === nothing
        entry.loader === nothing && throw(ArgumentError("component has no factory or loader"))
        entry.factory = entry.loader()
        entry.factory === nothing && throw(ArgumentError("component loader returned no factory"))
    end
    return entry.factory
end

"""
Create a component with native positional/keyword arguments. Optional keyword
filtering uses `accepted_keywords` recorded at registration; `nothing` means
forward every keyword. This replaces Python signature introspection explicitly.
"""
function create_component(registry::ComponentRegistry, key::AbstractString, args...;
    filter_unknown_kwargs::Bool = false, kwargs...)
    factory = component_factory(registry, key)
    accepted = _component_entry(registry, key).accepted_keywords
    selected = filter_unknown_kwargs && accepted !== nothing ? (; (name => value for (name, value) in pairs(kwargs) if name in accepted)...) : kwargs
    return factory(args...; selected...)
end

component_metadata(registry::ComponentRegistry, key::AbstractString) = copy(_component_entry(registry, key).metadata)
available_components(registry::ComponentRegistry) = sort!(collect(keys(registry.entries)))
component_aliases(registry::ComponentRegistry) = Dict(key => value for (key, value) in registry.index if key != lowercase(value))
is_lazy(registry::ComponentRegistry, key::AbstractString) = _component_entry(registry, key).factory === nothing
Base.in(key, registry::ComponentRegistry) = key isa AbstractString && haskey(registry.index, lowercase(key))
Base.length(registry::ComponentRegistry) = length(registry.entries)
Base.keys(registry::ComponentRegistry) = available_components(registry)
Base.getindex(registry::ComponentRegistry, key::AbstractString) = component_factory(registry, key)

include("BarController.jl")

const BARS = ComponentRegistry("bars")
const CROSS_VALIDATORS = ComponentRegistry("cross_validators")
const FEATURE_IMPORTANCE = ComponentRegistry("feature_importance")
const LABELERS = ComponentRegistry("labelers"; interface = AbstractLabeler)
const BET_SIZERS = ComponentRegistry("bet_sizers")
const PORTFOLIO_OPTIMIZERS = ComponentRegistry("portfolio_optimizers")
const REGISTRIES = Dict("bars" => BARS, "cross_validators" => CROSS_VALIDATORS,
    "feature_importance" => FEATURE_IMPORTANCE, "labelers" => LABELERS,
    "bet_sizers" => BET_SIZERS, "portfolio_optimizers" => PORTFOLIO_OPTIMIZERS)

function _builtin_bar_factory(family)
    return function (; metric = "dollar", kwargs...)
        metric_name = lowercase(String(metric))
        method = family == "standard_bars" ? metric_name*"_standard_bars" : replace(family, "_bars" => "", count = 1)
        if family != "standard_bars"
            parts = split(method, "_")
            method = parts[1]*"_"*metric_name*"_"*parts[2]*"_bars"
        end
        return initialize_bars(method; kwargs...)
    end
end

for family in ("standard_bars", "expected_imbalance_bars", "fixed_imbalance_bars", "expected_run_bars", "fixed_run_bars")
    register_component!(BARS, family, _builtin_bar_factory(family); metadata = Dict("family" => "bars", "afml_chapter" => 2))
end
register_component!(BARS, "time_bars", (; kwargs...) -> initialize_bars("time_bars"; kwargs...); metadata = Dict("family" => "bars", "afml_chapter" => 2))
for (name, factory) in (("kfold", Validation.KFoldCV), ("walkforward", Validation.WalkForwardCV), ("purgedkfold", Validation.PurgedKFoldCV), ("combinatorialpurged", Validation.CombinatorialPurgedCV), ("baggedcombinatorialpurged", Validation.BaggedCombinatorialPurgedCV), ("adaptivecombinatorialpurged", Validation.AdaptiveCombinatorialPurgedCV))
    register_component!(CROSS_VALIDATORS, name, factory; metadata = Dict("family" => "cross_validator"))
end

struct ConfiguredComponent{F,A,K}
    function_value::F
    arguments::A
    options::K
end

(component::ConfiguredComponent)(args...; kwargs...) = component.function_value(component.arguments..., args...; merge(component.options, (; kwargs...))...)

function _configured_factory(method)
    return (args...; kwargs...) -> ConfiguredComponent(method, args, (; kwargs...))
end

for (name, method) in (("MDI", Features.feature_importance_mdi), ("MDA", Features.feature_importance_mda), ("SFI", Features.feature_importance_sfi), ("ClusteredMDI", Features.clustered_feature_importance_mdi), ("ClusteredMDA", Features.clustered_feature_importance_mda))
    register_component!(FEATURE_IMPORTANCE, name, _configured_factory(method); metadata = Dict("family" => "feature_importance"))
end

"""Return a native component family by its canonical family name."""
get_registry(family::AbstractString) = REGISTRIES[String(family)]

"""Return sorted canonical component names for each public family."""
list_components() = Dict(family => available_components(registry) for (family, registry) in REGISTRIES)

export ComponentRegistry, register_component!, register_lazy!, unregister_component!,
    component_factory, create_component, component_metadata, available_components,
    component_aliases, is_lazy, AbstractLabeler, label,
    BARS, CROSS_VALIDATORS, FEATURE_IMPORTANCE, LABELERS, BET_SIZERS,
    PORTFOLIO_OPTIMIZERS, get_registry, list_components,
    BAR_COLUMNS, BarController, initialize_bars, bar_methods, process_bars

end
