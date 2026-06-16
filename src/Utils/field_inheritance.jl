"""
    @field_inherit Name{T} SuperType BaseStruct begin
        extra_field::Type
        ...
    end

Define a `mutable struct Name <: SuperType` whose fields are those of
`BaseStruct` (copied in declaration order) followed by the extra fields in the
block. Used to build the bar-type hierarchy without repeating the shared base
fields.

The base type is resolved in the **calling** module (`__module__`), so the base
struct only needs to be defined where the macro is used (e.g. `AbstractBars`
defined in `RiskLabAI.Data` before the concrete bar types). The previous
version evaluated in the macro's own module, which broke once the base lived in
a different module.
"""
macro field_inherit(name, super_type, base_type, fields)
    base = Core.eval(__module__, base_type)
    base_fieldnames = fieldnames(base)
    base_types = fieldtypes(base)
    base_fields = [:($f::$T) for (f, T) in zip(base_fieldnames, base_types)]

    result_expression = :(mutable struct $name <: $super_type end)
    append!(result_expression.args[end].args, base_fields)
    append!(result_expression.args[end].args, fields.args)
    # Escape so the struct name, super-type and field type references (e.g.
    # `Metric`, `StandardBarsType`) resolve in the *calling* module (`Data`),
    # not in `Utils` where this macro is defined.
    return esc(result_expression)
end
