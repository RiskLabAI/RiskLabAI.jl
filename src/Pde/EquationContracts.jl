"""
    pde_terminal_for_sample(eq, states)

Evaluate the released multi-sample terminal helper on a batch-by-sample-by-
dimension Float64 array, retaining the final singleton axis. The public
`PricingDiffRate` helper returns only the strike-120 call, whereas
`pde_terminal` returns the strike-120/150 call spread. These are distinct
released contracts and must not be interchanged.
"""
function pde_terminal_for_sample(eq::Equation, states)
    Zygote.ignore() do
        states isa AbstractArray{Float64, 3} && !isempty(states) && size(states, 3) == eq.dim && all(isfinite, states) || throw(ArgumentError("Sample states must be finite nonempty batch-by-sample-by-dimension Float64 arrays"))
    end
    return _pde_sample_terminal(eq, states)
end

_pde_sample_terminal(eq::HJBLQ, states) = log.(0.5 .* (1 .+ sum(states.^2; dims = 3)))
_pde_sample_terminal(eq::BlackScholesBarenblatt, states) = sum(states.^2; dims = 3)
_pde_sample_terminal(eq::PricingDefaultRisk, states) = _relu(minimum(states; dims = 3))
_pde_sample_terminal(eq::PricingDiffRate, states) = _relu(maximum(states; dims = 3) .- 120)
