function _feature_real_array(values, name, dimensions; empty = false)
    values isa AbstractArray && ndims(values) == dimensions && eltype(values) <: Real && !(eltype(values) <: Bool) || throw(ArgumentError("$name must be a real array of dimension $dimensions"))
    converted = Float64.(values)
    (empty || !isempty(converted)) && all(isfinite, converted) || throw(ArgumentError("$name must contain finite values and cannot be empty"))
    return converted
end

function _feature_real_scalar(value, name; minimum = -Inf, strict = false)
    value isa Real && !(value isa Bool) || throw(ArgumentError("$name must be a real scalar"))
    converted = Float64(value)
    isfinite(converted) && (strict ? converted > minimum : converted >= minimum) || throw(ArgumentError("$name is outside its allowed domain"))
    return converted
end

function _feature_positive_integer(value, name)
    value isa Integer && !(value isa Bool) && value > 0 || throw(ArgumentError("$name must be a positive integer"))
    return Int(value)
end
