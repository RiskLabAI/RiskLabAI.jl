# Julia runtime and dependency support

Status: **local 1.1.0 candidate; release blocked**.

This policy is inherited from the approved RiskLabAI `1.0.0` baseline and
applies to the additive `1.1.0` candidate. Local compatibility and package
inspection do not authorize version control, registration, publication, or
release.

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
- QuadGK 2.11.3;
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

The governed base test target contains only `Test`; it does not promote weak
dependencies into required dependencies. Base lanes verify four explicit
extension-absence and fallback assertions. Separate `deep_bsde` lanes install
and load Lux, Optimisers, and Zygote before running the four numerical solver
assertions. Every CI lane creates a temporary test environment and adds all
eight required external packages as direct dependencies at either their exact
floors or current compatible versions. Extension lanes additionally add the
three weak dependencies. This keeps direct imports in the preserved tests
independent of incidental manifest state.

TimeSeries is not a required dependency. No preserved source behavior uses it,
and the complete base library passes without it.

## Matrix evidence

Four runtime/dependency combinations were exercised: Julia 1.10.12 and 1.12.7,
each with the minimum and current-compatible required dependency sets. Every
combination was run once as a base lane and once as a `deep_bsde` extension
lane, for eight required executions.

Every lane passed 12,926 assertions, including 12,414 causal-factor
assertions. That is 103,408 passing assertions in the eight executions,
including 99,312 causal-factor assertions. Base lanes loaded without Lux,
Optimisers, or Zygote and passed all four extension-absence and fallback
assertions. Extension lanes loaded the three weak dependencies and passed all
four numerical deep-BSDE assertions. There were no failures, errors, broken
tests, or skips. The causal suite includes independent analytical and
exhaustive graph oracles; it does not use Python-Julia agreement as its sole
correctness test.

The CI workflow keeps Julia 1.10.12 LTS and Julia 1.12.7 stable as explicit
base and extension jobs. Both minimum and current jobs construct their
temporary test environments explicitly. JuliaFormatter 2.9.0 checks only the
independently maintained causal source, causal tests, and test entry point;
preserved source remains governed by the complete analytical test suite.

The deep solver now converts Lux parameters and state to Float64, matching the
PDE state and eliminating a mixed-precision fallback present in every original
lane. This changes neither the public API nor the estimand.

The original 57-concept causal contract remains unchanged. Thirty additive
concepts bring the public causal namespace to 87 names, with the same public
set in Python and Julia. No excluded legacy causal implementation was used as
a correctness authority.

## Human-controlled release gates

`Project.toml`, the public surface, package-scoped tests, documentation, source
allowlists, and governed CI workflow are complete. The local 1.1.0 candidate is
bound to the approved merged 1.0.0 baseline commit and tree. Package-inspection
evidence is retained outside the distributable tree. Version-control,
registration, publication, and release actions require separate human action.
