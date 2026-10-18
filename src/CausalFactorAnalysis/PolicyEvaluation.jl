"""
    evaluate_policy(actions, rewards, behavior_probabilities, policy_probabilities,
                    outcome_predictions)

Return one-step IPW and doubly robust policy values, observation scores and
importance-weight diagnostics. Matrices have shape `(rows, actions)`; actions
are one-based columns. Predictions must be out of fold or independently trained.
The caller justifies consistency and conditional exchangeability. All policy
support must have behavior support. Probabilities are neither clipped nor
renormalized, and no sequential reinforcement-learning claim is made.
"""
function evaluate_policy(actions, rewards, behavior_probabilities, policy_probabilities, outcome_predictions)
    rewards = _real_array(rewards, "rewards", 1)
    behavior = _probabilities(behavior_probabilities, "behavior_probabilities", 2)
    policy = _probabilities(policy_probabilities, "policy_probabilities", 2)
    predictions = _real_array(outcome_predictions, "outcome_predictions", 2)
    size(behavior) == size(policy) == size(predictions) && size(behavior, 1) == length(rewards) ||
        throw(ArgumentError("outcome and probability shapes are incompatible"))
    actions = _column_indices(actions, length(rewards), size(behavior, 2), "actions")
    observed = [behavior[i, actions[i]] for i in eachindex(rewards)]
    if any((policy .> 0) .& (behavior .== 0)) || any(iszero, observed)
        throw(ArgumentError("policy evaluation requires behavior-policy overlap"))
    end
    weights = [policy[i, actions[i]] / observed[i] for i in eachindex(rewards)]
    ipw = weights .* rewards
    selected_predictions = [predictions[i, actions[i]] for i in eachindex(rewards)]
    dr = vec(sum(policy .* predictions; dims = 2)) + weights .* (rewards - selected_predictions)
    all(isfinite, ipw) && all(isfinite, dr) && all(isfinite, weights) ||
        throw(ArgumentError("policy scores exceed floating-point range"))
    maximum_weight = maximum(weights)
    normalized = iszero(maximum_weight) ? weights : weights ./ maximum_weight
    effective = all(iszero, normalized) ? 0.0 : sum(normalized)^2 / dot(normalized, normalized)
    return (ipw_value = mean(ipw), dr_value = mean(dr), ipw_scores = ipw, dr_scores = dr,
        importance_weights = weights, effective_sample_size = effective,
        max_weight = maximum_weight, mean_weight = mean(weights))
end

function _split_indices(values, row_count)
    values isa AbstractVector || throw(ArgumentError("split indices must be vectors"))
    indices = _column_indices(values, length(values), row_count, "split indices")
    !isempty(indices) && length(unique(indices)) == length(indices) ||
        throw(ArgumentError("split indices must be nonempty and distinct"))
    return indices
end

"""
    select_policy(actions, rewards, behavior_probabilities, candidate_probabilities,
                  outcome_predictions; training_indices, evaluation_indices)

Select the largest training DR value and evaluate the chosen policy on later
held-out rows. Candidates have shape `(policies, rows, actions)` and must be
fixed before evaluation. The first candidate wins ties. Split rows and the
returned policy index are one-based. Training must precede evaluation, with no
duplicate rows. The caller remains responsible for nuisance-model training
history and independent evaluation; evaluation rewards never choose the policy.
"""
function select_policy(actions, rewards, behavior_probabilities, candidate_probabilities, outcome_predictions; training_indices, evaluation_indices)
    rewards = _real_array(rewards, "rewards", 1)
    candidates = _real_array(candidate_probabilities, "candidate_probabilities", 3)
    size(candidates, 2) == length(rewards) && size(actions) == size(rewards) &&
        size(behavior_probabilities) == size(outcome_predictions) == (size(candidates, 2), size(candidates, 3)) ||
        throw(ArgumentError("candidate and observation arrays must match"))
    train = _split_indices(training_indices, length(rewards))
    evaluation = _split_indices(evaluation_indices, length(rewards))
    maximum(train) < minimum(evaluation) || throw(ArgumentError("training must precede held-out evaluation"))
    values = [evaluate_policy(actions[train], rewards[train], behavior_probabilities[train, :], candidates[policy, train, :], outcome_predictions[train, :]).dr_value for policy in axes(candidates, 1)]
    selected = argmax(values)
    result = evaluate_policy(actions[evaluation], rewards[evaluation], behavior_probabilities[evaluation, :], candidates[selected, evaluation, :], outcome_predictions[evaluation, :])
    return (selected_policy = selected, training_values = values, evaluation = result)
end
