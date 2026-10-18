# Julia runtime and dependency support

Status: **published 1.1.0 baseline with unreleased local additions**.

This policy records the published RiskLabAI `1.1.0` baseline and distinguishes
it from the current local integration. Local compatibility and package
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

## Required dependencies

The required external dependency floors are:

- Clarabel 0.11.1;
- Clustering 0.15.8;
- Combinatorics 1.1.0;
- CovarianceEstimation 0.2.13;
- DataFrames 1.8.2;
- DecisionTree 0.12.4;
- Distributions 0.25.128;
- HypothesisTests 0.11.8;
- JuMP 1.32.1;
- Lux 1.31.4;
- Optim 2.3.2;
- Optimisers 0.4.7;
- QuadGK 2.11.3;
- SpecialFunctions 2.8.0;
- Zygote 0.7.11.

The current-compatible lanes resolved Distributions 0.25.131 and
SpecialFunctions 2.9.0; the other required packages remained at their stated
floors. Dates, LinearAlgebra, Random, and Statistics are required standard
libraries.

Lux, Optimisers, and Zygote are direct dependencies because the native
Deep-BSDE solver is part of the package. Each test lane constructs a temporary
environment and checks all external packages at either their stated floors or
current compatible versions.

TimeSeries is not a required dependency. No preserved source behavior uses it,
and the complete base library passes without it.

## Matrix evidence

The published 1.1.0 baseline was exercised on Julia 1.10.12 and 1.12.7 with
minimum and current-compatible dependency sets. That earlier matrix separated
the base package from the optional Deep-BSDE extension.

Those published-baseline runs passed 12,926 assertions per lane, including
12,414 causal-factor assertions. The causal suite includes independent
analytical and exhaustive graph oracles; it does not use Python-Julia
agreement as its sole correctness test.

The current merged source has also passed the released causal suite, the new
regression files, and the native Deep-BSDE checks in the local Windows
environment. This local result does not replace the two-runtime compatibility
matrix. Both supported Julia lines and both dependency sets must pass again
before a later release.

The CI workflow keeps Julia 1.10.12 LTS and Julia 1.12.7 stable as explicit
minimum and current jobs. JuliaFormatter 2.9.0 checks the independently
maintained causal source, causal tests, and test entry point; preserved source
remains governed by the complete analytical test suite.

The deep solver now converts Lux parameters and state to Float64, matching the
PDE state and eliminating a mixed-precision fallback present in every original
lane. This changes neither the public API nor the estimand.

The original 57-concept causal contract remains unchanged. Thirty additive
concepts bring the public causal namespace to 87 names, with the same public
set in Python and Julia. No excluded legacy causal implementation was used as
a correctness authority.

## Human-controlled release gates

The current integration must pass its full compatibility matrix before a
release decision. Package-inspection evidence is retained outside the
distributable tree. Version-control, registration, publication, and release
actions require separate human action.
