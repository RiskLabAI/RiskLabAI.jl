macro field_inherit(name, super_type, base_type, fields)
    base_type = eval(base_type)
    base_fieldnames = fieldnames(base_type)
    base_types = fieldtypes(base_type)
    base_fields = [:($f::$T) for (f, T) in zip(base_fieldnames, base_types)]
    result_expression = :(mutable struct $name <: $super_type end)
    push!(result_expression.args[end].args, base_fields...)
    push!(result_expression.args[end].args, fields.args...)
    return result_expression
end