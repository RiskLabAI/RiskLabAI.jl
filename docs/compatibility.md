# Julia runtime and dependency support

Status: **matrix verified; release blocked**.

This policy applies to the blocked RiskLabAI `1.0.0` release candidate. The
assigned version does not authorize package registration, artifact creation,
publication, or release.

## Supported Julia policy

RiskLabAI supports the maintained LTS and stable Julia lines tested for this
release:

- Julia 1.10.12 LTS;
- Julia 1.12.7 stable.

Julia 1.11 is best-effort rather than a supported lane. If the upstream stable
line changes before release metadata is frozen, the new stable line must
replace 1.12 in the matrix and pass again.

Both official runtime archives were verified against their published SHA-256
checksums before testing.

## Required and optional dependencies

The required external dependency floors are:

- Clustering 0.15.8;
- Combinatorics 1.1.0;
- DataFrames 1.8.2;
- DecisionTree 0.12.4;
- Distributions 0.25.128;
- HypothesisTests 0.11.8;
- SpecialFunctions 2.8.0.

The current-compatible lanes resolved Distributions 0.25.131 and
SpecialFunctions 2.9.0; the other required packages remained at their stated
floors. Dates, LinearAlgebra, Random, and Statistics are required standard
libraries.

Lux, Optimisers, and Zygote are isolated behind the optional `deep_bsde`
capability. The tested minimums are Lux 1.31.4, Optimisers 0.4.7, and Zygote
0.7.11. Current lanes used Lux 1.31.4, Optimisers 0.4.9, and Zygote 0.7.12.
The base source loads without these packages. Calling `solve_deep_bsde` without
the extension raises a direct dependency error.

TimeSeries is not a required dependency. No preserved source behavior uses it,
and the complete base library passes without it.

## Matrix evidence

Four required lanes were exercised: Julia 1.10.12 and 1.12.7, each with the
minimum and current-compatible dependency sets.

In every lane:

- the isolated base source loaded without Lux, Optimisers, or Zygote;
- all 138 frozen causal-factor tests passed;
- all 508 non-deep assertions in the 45 preserved test sets passed;
- all 4 deep-BSDE assertions passed with the optional stack enabled.

That is 650 passing assertions per lane and 2,600 across the four lanes. The
causal suite includes independent analytical and exhaustive graph oracles; it
does not use Python-Julia agreement as its sole correctness test.

The deep solver now converts Lux parameters and state to Float64, matching the
PDE state and eliminating a mixed-precision fallback present in every original
lane. This changes neither the public API nor the estimand.

The causal public design remains exactly 57 concepts and matches the Python
namespace. No excluded legacy causal implementation was used as a correctness
authority.

## Remaining release gates

`Project.toml`, the public surface, package-scoped tests, documentation, and
source allowlists are complete. No package tree has been installed or
registered. Temporary artifact inspection and separate human authorization for
version-control, registration, publication, and release remain outstanding.
