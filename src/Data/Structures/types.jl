abstract type Metric end
abstract type Dollar <: Metric end
abstract type Volume <: Metric end
abstract type Tick <: Metric end

abstract type AbstractBarsType end

abstract type AbstractInformationDrivenBarsType{T<:Metric} <: AbstractBarsType end

abstract type AbstractImbalanceBarsType{T<:Metric} <: AbstractInformationDrivenBarsType{T} end
abstract type ExpectedImbalanceBarsType{T<:Metric} <: AbstractImbalanceBarsType{T} end
abstract type FixedImbalanceBarsType{T<:Metric} <: AbstractImbalanceBarsType{T} end

abstract type AbstractRunBarsType{T<:Metric} <: AbstractInformationDrivenBarsType{T} end
abstract type ExpectedRunBarsType{T<:Metric} <: AbstractRunBarsType{T} end
abstract type FixedRunBarsType{T<:Metric} <: AbstractRunBarsType{T} end

abstract type StandardBarsType{T<:Metric} <: AbstractBarsType end

abstract type TimeBarsType <: AbstractBarsType end