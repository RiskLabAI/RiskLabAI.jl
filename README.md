# RiskLabAI

[![CI](https://github.com/RiskLabAI/RiskLabAI.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/RiskLabAI/RiskLabAI.jl/actions/workflows/CI.yml)

RiskLabAI is a Julia library for quantitative finance, financial machine
learning, and causal factor analysis. It provides research-oriented
implementations of methods associated with Marcos LÃ³pez de Prado's *Advances
in Financial Machine Learning*, *Machine Learning for Asset Managers*, and
*Causal Factor Investing*.

RiskLabAI 1.0.0 preserves the previously published Julia library and adds a
clean causal-factor-analysis module. The companion
[RiskLabAI.py](https://github.com/RiskLabAI/RiskLabAI.py) package independently
implements the same 57 causal concepts. This parity statement applies to the
causal API, not to every Julia module.

## What is included

- **Causal factor analysis** - constrained minimum-variance allocation,
  factor-mirage diagnostics, graphical identification, treatment-effect
  formulas, specification experiments, and evidence records for the
  seven-stage causal-factor protocol
- **Financial data structures** - tick, volume, dollar, imbalance, run, and
  time bars
- **Market features** - entropy, microstructure, structural-break, and feature-
  importance utilities
- **Portfolio and clustering methods** - hierarchical risk parity, nested
  clustered optimization, hedging, correlation clustering, and silhouette
  diagnostics
- **Backtest analytics** - Sharpe-ratio inference, probability of backtest
  overfitting, strategy risk, multiple-testing corrections, bet sizing, and
  Ornstein-Uhlenbeck trading rules
- **Validation** - K-fold, purged, combinatorial-purged, and walk-forward
  validation, plus grid and random search
- **Optional capability** - a Deep-BSDE PDE solver using Lux, Optimisers, and
  Zygote through a Julia package extension

## Compatibility

RiskLabAI 1.0.0 supports Julia 1.10.12 LTS and Julia 1.12.7. The complete
tested policy and dependency details are in
[`docs/compatibility.md`](https://github.com/RiskLabAI/RiskLabAI.jl/blob/main/docs/compatibility.md).

## Installation

Until RiskLabAI is registered in Julia's General registry, install it directly
from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/RiskLabAI/RiskLabAI.jl")
```

After registration, the standard installation command will be:

```julia
using Pkg
Pkg.add("RiskLabAI")
```

To enable the optional Deep-BSDE solver, add its three weak dependencies to
the active environment:

```julia
using Pkg
Pkg.add(["Lux", "Optimisers", "Zygote"])
```

The base package does not require this automatic-differentiation stack.

## Causal-factor quick start

```julia
using RiskLabAI
using RiskLabAI.CausalFactorAnalysis
using LinearAlgebra: Diagonal

covariance = Matrix(Diagonal([1.0, 2.0, 4.0]))
factor_exposures = [1.0 0.0; 0.0 1.0; 1.0 1.0]
target_exposures = [0.0, 1.0]

weights = minimum_variance_factor_weights(
    covariance,
    factor_exposures,
    target_exposures,
)
@assert isapprox(weights, [-2.0 / 7.0, 5.0 / 7.0, 2.0 / 7.0])

effect = average_treatment_effect(3.5, 1.25)
@assert effect == 2.25

dag = CausalDAG(
    ("T", "U", "Y"),
    (("U", "T"), ("U", "Y"), ("T", "Y")),
    ("T", "U", "Y"),
)
@assert check_backdoor_adjustment_set(dag, "T", "Y", ("U",)).admissible
```

The complete deterministic example is
[`examples/causal_factor_analysis_quickstart.jl`](https://github.com/RiskLabAI/RiskLabAI.jl/blob/main/examples/causal_factor_analysis_quickstart.jl).
The causal API and its limits are documented in
[`docs/causal_factor_analysis.md`](https://github.com/RiskLabAI/RiskLabAI.jl/blob/main/docs/causal_factor_analysis.md).

## Development

```bash
git clone https://github.com/RiskLabAI/RiskLabAI.jl
cd RiskLabAI.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Please branch from `main`, keep changes focused, and include tests for behavior
changes.

## Scope

RiskLabAI is research software, not investment advice. Graph routines evaluate
criteria on a caller-supplied directed acyclic graph; they do not discover or
certify that graph. Protocol records validate declared evidence structures;
they do not prove that empirical assumptions are true.

## License

RiskLabAI is distributed under the
[BSD 3-Clause License](https://github.com/RiskLabAI/RiskLabAI.jl/blob/main/LICENSE).
