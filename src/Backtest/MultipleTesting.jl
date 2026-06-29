"""
Multiple-testing haircuts for the Sharpe ratio (Holm FWER, Benjamini–Hochberg–
Yekutieli FDR; Harvey–Liu 2015). When a researcher screens many strategies or
factors and reports the significant ones, the selection inflates the apparent
significance. These corrections discount it by controlling, across the whole
family of trials, either the family-wise error rate (the chance of *any* false
discovery, Holm) or the false-discovery rate (the expected *fraction* of false
discoveries, BHY). They complement the Deflated Sharpe Ratio
(`probabilistic_sharpe_ratio` / `expected_max_sharpe_ratio`): the DSR asks whether
the single best strategy survives its trial count, while these screen a whole
family with a stated error-control target.

Clean-room Julia port of the validated Python
`RiskLabAI.backtest.multiple_testing` reference (numeric parity asserted in
`test/runtests.jl`). Admitted in Appraisal 07
(`library_extension/appraisals/07_verdict.md`).

References: Holm, S. (1979), Scandinavian Journal of Statistics 6(2);
Benjamini, Y. & Yekutieli, D. (2001), Annals of Statistics 29(4); Harvey, C. R. &
Liu, Y. (2015), Journal of Portfolio Management 42(1).
"""

using Distributions: Normal, ccdf, cquantile

"""
    sharpe_ratio_p_values(sharpe_ratios, number_of_returns) -> Vector{Float64}

One-sided p-values for a positive edge (H1: SR > 0) from per-period Sharpe ratios:
under the i.i.d. null the Sharpe t-statistic is `t = ŜR·√T`, so the p-value is
`Φ(-t)`. Mirrors Python's `sharpe_ratio_p_values`.
"""
function sharpe_ratio_p_values(
    sharpe_ratios::AbstractVector{<:Real},
    number_of_returns::Integer,
)
    t_stats = float.(sharpe_ratios) .* sqrt(number_of_returns)
    return ccdf.(Normal(), t_stats)
end

"""
    holm_adjusted_p_values(p_values) -> Vector{Float64}

Holm (1979) step-down family-wise-error-rate adjusted p-values: sorting ascending,
the adjusted value at rank `i` is `maxⱼ≤ᵢ (M - j + 1) p₍ⱼ₎`, clipped to 1, then
mapped back to the input order. Controls the FWER under any dependence and
dominates Bonferroni. Mirrors Python's `holm_adjusted_p_values`.
"""
function holm_adjusted_p_values(p_values::AbstractVector{<:Real})
    p = float.(p_values)
    m = length(p)
    m == 0 && return copy(p)
    order = sortperm(p)
    sorted_p = p[order]
    raw = [(m - (i - 1)) * sorted_p[i] for i = 1:m]
    adjusted_sorted = clamp.(accumulate(max, raw), 0.0, 1.0)
    adjusted = similar(p)
    adjusted[order] = adjusted_sorted
    return adjusted
end

"""
    benjamini_hochberg_yekutieli_adjusted_p_values(p_values) -> Vector{Float64}

Benjamini–Hochberg–Yekutieli (2001) step-up false-discovery-rate adjusted
p-values, using the Yekutieli dependence constant `c(M) = Σ₁ᴹ 1/i`. Sorting
ascending, the adjusted value at rank `i` is the running minimum (from the largest
rank down) of `c(M)·M·p₍ᵢ₎/i`, clipped to 1. Controls the FDR under any
dependence. Mirrors Python's `benjamini_hochberg_yekutieli_adjusted_p_values`.
"""
function benjamini_hochberg_yekutieli_adjusted_p_values(p_values::AbstractVector{<:Real})
    p = float.(p_values)
    m = length(p)
    m == 0 && return copy(p)
    order = sortperm(p)
    sorted_p = p[order]
    c_m = sum(1.0 / i for i = 1:m)
    factor = [c_m * m / i for i = 1:m]
    vals = factor .* sorted_p
    # Step-up: enforce monotonicity from the largest p-value downward.
    adjusted_sorted = clamp.(reverse(accumulate(min, reverse(vals))), 0.0, 1.0)
    adjusted = similar(p)
    adjusted[order] = adjusted_sorted
    return adjusted
end

"""
    haircut_sharpe_ratios(sharpe_ratios, number_of_returns; method="holm",
                          significance_level=0.05)

Apply a multiple-testing haircut to a family of screened Sharpe ratios. Converts
each Sharpe to its one-sided p-value, adjusts for multiple testing (`"holm"` for
FWER or `"bhy"` for FDR), flags survivors at `significance_level`, and reports the
haircut Sharpe (the Sharpe implied by the adjusted p-value, `Φ⁻¹(1 - p_adj)/√T`).
Returns a `NamedTuple` `(p_values, adjusted_p_values, significant,
haircut_sharpe_ratios)`.

Preferred-when / avoid-when (regime tag, verbatim from `CONTRIBUTIONS_LEDGER.md`):
to judge a *family* of screened strategies/factors (not just the single best),
prefer Holm when you must control the chance of any false positive across the
family (FWER), and BHY when you want to bound the expected fraction of false
discoveries (FDR) and can accept lower power. The DSR remains the tool for whether
one specific best strategy survives its trial count. Bonferroni is dominated by
Holm; the t>3 hurdle over-rejects; the double-bootstrap does not control its stated
error as built (all shelved on record).

Mirrors Python's `haircut_sharpe_ratios`.
"""
function haircut_sharpe_ratios(
    sharpe_ratios::AbstractVector{<:Real},
    number_of_returns::Integer;
    method::AbstractString = "holm",
    significance_level::Real = 0.05,
)
    p_values = sharpe_ratio_p_values(sharpe_ratios, number_of_returns)
    if method == "holm"
        adjusted = holm_adjusted_p_values(p_values)
    elseif method == "bhy"
        adjusted = benjamini_hochberg_yekutieli_adjusted_p_values(p_values)
    else
        throw(ArgumentError("unknown method $(repr(method)); use \"holm\" or \"bhy\"."))
    end
    significant = adjusted .< significance_level
    haircut_t = cquantile.(Normal(), clamp.(adjusted, 1e-300, 1.0))
    haircut_sr = haircut_t ./ sqrt(number_of_returns)
    return (
        p_values = p_values,
        adjusted_p_values = adjusted,
        significant = significant,
        haircut_sharpe_ratios = haircut_sr,
    )
end
