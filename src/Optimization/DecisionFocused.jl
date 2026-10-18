using Zygote

"""
    spo_plus_loss(predicted_cost, true_cost, vertices)

Return SPO+ loss, a subgradient and one-based oracle vertex indices for
linear cost minimization over the convex hull of supplied vertex rows.
Complete enumeration is the exact oracle for this fixed finite hull. Ties
choose the first supplied vertex; vertices are never inferred or generated.
"""
function spo_plus_loss(predicted_cost, true_cost, vertices)
    predicted = _portfolio_array(predicted_cost, "predicted_cost", 1)
    truth = _portfolio_array(true_cost, "true_cost", 1)
    choices = _portfolio_array(vertices, "vertices", 2)
    length(predicted) == length(truth) == size(choices, 2) || throw(ArgumentError("cost and vertex dimensions must match"))
    true_objectives = choices * truth
    transformed_objectives = choices * (2 * predicted - truth)
    all(isfinite, true_objectives) && all(isfinite, transformed_objectives) || throw(ArgumentError("oracle objectives exceed floating-point range"))
    true_index, transformed_index = argmin(true_objectives), argmin(transformed_objectives)
    true_decision, transformed = choices[true_index, :], choices[transformed_index, :]
    loss = dot(truth - 2 * predicted, transformed) + dot(2 * predicted - truth, true_decision)
    subgradient = 2 * (true_decision - transformed)
    isfinite(loss) && all(isfinite, subgradient) || throw(ArgumentError("loss or subgradient exceeds floating-point range"))
    return (; loss, subgradient, true_vertex = true_index, transformed_vertex = transformed_index)
end

"""
    spo_plus(predicted_cost, true_cost, vertices)

Return differentiable SPO+ loss for a floating vector or a matrix with one
cost vector per row. All arrays share an element type. Zygote treats truth,
vertices and first-index oracle selections as constants; the gradient with
respect to predictions is the selected SPO+ subgradient. Batched losses are
returned without reduction. This is the native Julia counterpart of the
Python tensor adapter.
"""
function spo_plus(predicted_cost, true_cost, vertices)
    truth, selected_true, selected_transformed = Zygote.ignore() do
        all(x -> x isa AbstractArray && eltype(x) <: AbstractFloat && all(isfinite, x), (predicted_cost, true_cost, vertices)) || throw(ArgumentError("all inputs must be finite floating arrays"))
        eltype(predicted_cost) == eltype(true_cost) == eltype(vertices) || throw(ArgumentError("inputs must share an element type"))
        ndims(predicted_cost) in (1, 2) && size(predicted_cost) == size(true_cost) && !isempty(predicted_cost) &&
            ndims(vertices) == 2 && size(vertices, 1) > 0 && size(vertices, 2) == size(predicted_cost, ndims(predicted_cost)) || throw(ArgumentError("cost and vertex shapes are incompatible"))
        truth = copy(true_cost)
        predicted_rows = ndims(predicted_cost) == 1 ? permutedims(predicted_cost) : predicted_cost
        truth_rows = ndims(truth) == 1 ? permutedims(truth) : truth
        true_objectives = truth_rows * vertices'
        transformed_objectives = (2 * predicted_rows - truth_rows) * vertices'
        all(isfinite, true_objectives) && all(isfinite, transformed_objectives) || throw(ArgumentError("oracle objectives exceed floating-point range"))
        first_indices = [argmin(row) for row in eachrow(true_objectives)]
        transformed_indices = [argmin(row) for row in eachrow(transformed_objectives)]
        selected_true = vertices[first_indices, :]
        selected_transformed = vertices[transformed_indices, :]
        all(isfinite, 2 * (selected_true - selected_transformed)) || throw(ArgumentError("subgradient exceeds floating-point range"))
        if ndims(predicted_cost) == 1
            selected_true, selected_transformed = vec(selected_true), vec(selected_transformed)
        end
        return truth, selected_true, selected_transformed
    end
    terms = (truth - 2 * predicted_cost) .* selected_transformed + (2 * predicted_cost - truth) .* selected_true
    loss = ndims(predicted_cost) == 1 ? sum(terms) : vec(sum(terms; dims = 2))
    Zygote.ignore() do
        all(isfinite, loss isa Number ? (loss,) : loss) || throw(ArgumentError("loss exceeds floating-point range"))
    end
    return loss
end
