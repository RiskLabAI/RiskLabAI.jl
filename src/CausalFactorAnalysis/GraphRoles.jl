"""
    FactorControlRoles

Reachability roles for factors relative to one observed target in an accepted
causal DAG. The observed fields partition every observed non-target node. The
unobserved fields separately record latent reachability evidence.
"""
struct FactorControlRoles
    target_factor::String
    ancestor_factors::Tuple{Vararg{String}}
    descendant_factors::Tuple{Vararg{String}}
    other_factors::Tuple{Vararg{String}}
    unobserved_ancestors::Tuple{Vararg{String}}
    unobserved_descendants::Tuple{Vararg{String}}
end

const _TREATMENT_OUTCOME_ROLE_VALUES = (
    :cause_of_treatment,
    :consequence_of_treatment,
    :cause_of_outcome,
    :consequence_of_outcome,
    :confounder,
    :collider,
    :mediator,
    :independent,
)

"""
    TreatmentOutcomeRole(value)

Validated symbol-backed representation of one of the eight admitted one-hop
treatment/outcome roles.
"""
struct TreatmentOutcomeRole
    value::Symbol

    function TreatmentOutcomeRole(value::Symbol)
        value in _TREATMENT_OUTCOME_ROLE_VALUES ||
            throw(ArgumentError("value must identify one of the eight admitted roles"))
        return new(value)
    end
end

TreatmentOutcomeRole(value::AbstractString) = TreatmentOutcomeRole(Symbol(value))

Base.:(==)(left::TreatmentOutcomeRole, right::TreatmentOutcomeRole) =
    left.value == right.value
Base.isequal(left::TreatmentOutcomeRole, right::TreatmentOutcomeRole) =
    isequal(left.value, right.value)
Base.hash(role::TreatmentOutcomeRole, seed::UInt) = hash(role.value, seed)
Base.string(role::TreatmentOutcomeRole) = String(role.value)

function Base.show(io::IO, role::TreatmentOutcomeRole)
    return print(io, "TreatmentOutcomeRole(:", role.value, ")")
end

"""Direct-edge evidence for one admitted treatment/outcome role."""
struct TreatmentOutcomeRoleEvidence
    node::String
    treatment::String
    outcome::String
    role::TreatmentOutcomeRole
    node_to_treatment::Bool
    treatment_to_node::Bool
    node_to_outcome::Bool
    outcome_to_node::Bool
end

function _role_observed_node(dag::CausalDAG, value, subject::AbstractString)
    node = _node(value, subject)
    node in dag.nodes || throw(ArgumentError("$subject must be a known graph node"))
    node in dag.observed_nodes ||
        throw(ArgumentError("$subject must be an observed graph node"))
    return node
end

"""
    factor_control_roles(dag, target_factor)

Partition observed non-target factors into ancestors, descendants, and all
remaining factors. Latent ancestors and descendants are reported separately.
This is reachability evidence, not a claim that every ancestor is a minimal or
sufficient adjustment set.
"""
function factor_control_roles(dag::CausalDAG, target_factor)
    target = _role_observed_node(dag, target_factor, "target_factor")
    ancestors = _ancestors(dag, target)
    descendants = _descendants(dag, target)
    observed_candidates = setdiff(Set(dag.observed_nodes), Set((target,)))
    unobserved = setdiff(Set(dag.nodes), Set(dag.observed_nodes))

    ancestor_factors = intersect(observed_candidates, ancestors)
    descendant_factors = intersect(observed_candidates, descendants)
    other_factors =
        setdiff(observed_candidates, union(ancestor_factors, descendant_factors))

    return FactorControlRoles(
        target,
        Tuple(sort(collect(ancestor_factors))),
        Tuple(sort(collect(descendant_factors))),
        Tuple(sort(collect(other_factors))),
        Tuple(sort(collect(intersect(unobserved, ancestors)))),
        Tuple(sort(collect(intersect(unobserved, descendants)))),
    )
end

factor_control_roles(dag, target_factor) = throw(ArgumentError("dag must be a CausalDAG"))

function _role_from_signature(signature::NTuple{4,Bool})
    signature == (true, false, false, false) &&
        return TreatmentOutcomeRole(:cause_of_treatment)
    signature == (false, true, false, false) &&
        return TreatmentOutcomeRole(:consequence_of_treatment)
    signature == (false, false, true, false) &&
        return TreatmentOutcomeRole(:cause_of_outcome)
    signature == (false, false, false, true) &&
        return TreatmentOutcomeRole(:consequence_of_outcome)
    signature == (true, false, true, false) && return TreatmentOutcomeRole(:confounder)
    signature == (false, true, false, true) && return TreatmentOutcomeRole(:collider)
    signature == (false, true, true, false) && return TreatmentOutcomeRole(:mediator)
    signature == (false, false, false, false) && return TreatmentOutcomeRole(:independent)
    throw(
        ArgumentError(
            "the direct treatment/outcome edge signature is not one of the eight admitted roles",
        ),
    )
end

"""
    classify_treatment_outcome_role(dag, treatment, outcome, node)

Classify an observed node from its four direct-edge indicators relative to an
observed treatment and outcome. Only the eight published signatures are
accepted; reverse mediation and every other signature fail closed.
"""
function classify_treatment_outcome_role(dag::CausalDAG, treatment, outcome, node)
    treatment_node = _role_observed_node(dag, treatment, "treatment")
    outcome_node = _role_observed_node(dag, outcome, "outcome")
    role_node = _role_observed_node(dag, node, "node")
    length(Set((treatment_node, outcome_node, role_node))) == 3 ||
        throw(ArgumentError("treatment, outcome, and node must be distinct"))

    edges = Set(dag.directed_edges)
    signature = (
        (role_node, treatment_node) in edges,
        (treatment_node, role_node) in edges,
        (role_node, outcome_node) in edges,
        (outcome_node, role_node) in edges,
    )
    return TreatmentOutcomeRoleEvidence(
        role_node,
        treatment_node,
        outcome_node,
        _role_from_signature(signature),
        signature...,
    )
end

classify_treatment_outcome_role(dag, treatment, outcome, node) =
    throw(ArgumentError("dag must be a CausalDAG"))
