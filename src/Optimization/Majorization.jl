function _majorization_tolerance(atol)
    value = _portfolio_scalar(atol, "atol")
    0 < value < 1 || throw(ArgumentError("atol must lie in (0, 1)"))
    return value
end

function _majorization_array(values)
    values isa AbstractArray && eltype(values) <: Real || throw(ArgumentError("inputs must contain finite real numbers"))
    result = Float64.(values)
    all(isfinite, result) || throw(ArgumentError("inputs must be finite"))
    return result
end

function _majorization_vectors(target, source)
    left, right = _majorization_array(target), _majorization_array(source)
    ndims(left) == 1 && size(left) == size(right) && !isempty(left) || throw(ArgumentError("target and source must be nonempty vectors of equal length"))
    return left, right
end

"""Check finite nonnegative entries and unit row/column sums within `atol`, without normalization."""
function is_doubly_stochastic(matrix; atol = 1e-10)
    tolerance = _majorization_tolerance(atol)
    values = try
        _majorization_array(matrix)
    catch exception
        exception isa ArgumentError || rethrow()
        return false
    end
    ndims(values) == 2 && size(values, 1) > 0 && size(values, 1) == size(values, 2) || return false
    return all(values .>= -tolerance) && all(abs.(sum(values; dims = 1) .- 1) .<= tolerance) && all(abs.(sum(values; dims = 2) .- 1) .<= tolerance)
end

"""Test equal-total finite vector majorization using descending partial sums and absolute tolerance."""
function is_majorized(target, source; atol = 1e-10)
    tolerance = _majorization_tolerance(atol)
    left, right = _majorization_vectors(target, source)
    left_sums = cumsum(BigFloat.(sort(left; rev = true)))
    right_sums = cumsum(BigFloat.(sort(right; rev = true)))
    return abs(left_sums[end] - right_sums[end]) <= tolerance && all(left_sums[1:end-1] .<= right_sums[1:end-1] .+ tolerance)
end

"""
    majorization_matrix(target, source; atol=1e-10)

Construct a checked doubly stochastic witness `D * source == target` with a
linear feasibility problem. The witness is generally nonunique. Row sums,
column sums, nonnegativity and the unscaled mapping residual must satisfy
the requested absolute tolerance. Inputs are not normalized or modified.
"""
function majorization_matrix(target, source; atol = 1e-10)
    tolerance = _majorization_tolerance(atol)
    left, right = _majorization_vectors(target, source)
    is_majorized(left, right; atol = tolerance) || throw(ArgumentError("target is not majorized by source"))
    n = length(left)
    all(abs.(left - right) .<= tolerance) && return Matrix{Float64}(I, n, n)
    scale = max(1.0, maximum(abs, left), maximum(abs, right))
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    for name in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
        set_optimizer_attribute(model, name, min(1e-12, tolerance / (10 * scale)))
    end
    @variable(model, 0 <= witness[1:n, 1:n] <= 1)
    @constraint(model, [i = 1:n], sum(witness[i, :]) == 1)
    @constraint(model, [j = 1:n], sum(witness[:, j]) == 1)
    @constraint(model, [i = 1:n], sum(witness[i, j] * (right[j] / scale) for j in 1:n) == left[i] / scale)
    @objective(model, Min, 0)
    optimize!(model)
    termination_status(model) == JuMP.MOI.OPTIMAL && has_values(model) || error("No verified majorization witness: $(termination_status(model))")
    result = value.(witness)
    is_doubly_stochastic(result; atol = tolerance) && all(abs.(result * right - left) .<= tolerance) || error("Majorization witness exceeds the requested residual tolerance")
    return result
end

function _positive_support_matching(residual)
    n = size(residual, 1)
    column_rows = zeros(Int, n)
    function augment(row, seen)
        for column in 1:n
            if residual[row, column] > 0 && !seen[column]
                seen[column] = true
                if column_rows[column] == 0 || augment(column_rows[column], seen)
                    column_rows[column] = row
                    return true
                end
            end
        end
        return false
    end
    for row in 1:n
        augment(row, falses(n)) || error("Residual support has no complete positive matching")
    end
    permutation = zeros(Int, n)
    for column in 1:n
        permutation[column_rows[column]] = column
    end
    return permutation
end

"""
    birkhoff_von_neumann_decomposition(matrix; atol=1e-10)

Return positive weights and a matrix of one-based permutations. Row `k`
selects column `permutations[k,i]` in matrix row `i`. Repeated complete
matchings on positive residual support construct a nonunique decomposition;
the weighted permutation sum and total mass are checked against the original
input. Tiny negative entries within tolerance are treated as roundoff.
"""
function birkhoff_von_neumann_decomposition(matrix; atol = 1e-10)
    tolerance = _majorization_tolerance(atol)
    is_doubly_stochastic(matrix; atol = tolerance) || throw(ArgumentError("matrix must be doubly stochastic"))
    original = _majorization_array(matrix)
    n = size(original, 1)
    residual = max.(original, 0)
    reconstruction = zeros(n, n)
    weights, permutations = Float64[], Vector{Int}[]
    mass = 0.0
    for _ in 1:length(original)
        abs(mass - 1) <= tolerance && maximum(abs, residual) <= tolerance && break
        permutation = _positive_support_matching(residual)
        weight = minimum(residual[i, permutation[i]] for i in 1:n)
        weight > 0 || error("Decomposition did not make positive progress")
        push!(weights, weight)
        push!(permutations, permutation)
        for i in 1:n
            residual[i, permutation[i]] -= weight
            reconstruction[i, permutation[i]] += weight
        end
        mass += weight
    end
    abs(mass - 1) <= tolerance && maximum(abs, reconstruction - original) <= tolerance || error("Decomposition exceeds the requested residual tolerance")
    return weights, reduce(vcat, permutedims.(permutations))
end
