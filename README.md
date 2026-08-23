# RiskLabAI 1.0.0 pre-package source

Status: **local-only and blocked**.

This directory is the source-only Julia continuity root for the next major
RiskLabAI release. It is the same RiskLabAI product and preserves package UUID
`a72881da-fdaa-49c1-8962-99caf4ccfee8`; it is not a separate package.

The `src` directory preserves all 56 source paths captured from the current
public Julia `main` baseline and adds seven causal-factor source files. No prior
source path or non-causal module has been deleted. Fifty-five baseline source
files remain byte-identical; the root module differs only by the additive
`CausalFactorAnalysis` include and export.

The causal module exposes the same 57 public concepts as the completed Python
namespace. Its 138 focused tests pass on Julia 1.10.12 LTS and Julia 1.12.7
stable at both dependency endpoints, including the independent four-node graph
oracles. All 508 non-deep preserved assertions also pass in each lane, and the
four deep-BSDE assertions pass with the optional extension enabled.

The static `Project.toml` contract now records the existing package UUID,
version `1.0.0`, supported Julia lines, exact required dependencies, and the
optional `deep_bsde` extension. No package tree has been registered, installed,
published, or released. The full runtime matrix, documentation, API inventory,
12-file package-scoped test inventory, and intended package file list are
complete. Artifact inspection and human-controlled version-control,
registration, publication, and release decisions remain blocked, and the
recorded source-conflict exclusions remain binding.

## Confirmed identity and stewardship

- Product and package name: `RiskLabAI`
- Julia package UUID: `a72881da-fdaa-49c1-8962-99caf4ccfee8`
- Intended license: BSD-3-Clause
- Rights holder and public maintainer: Hamid Arian
- Copyright: 2022-2026
- Contact: arian@risklab.ai
- Repository: https://github.com/RiskLabAI/RiskLabAI.jl
- Issues: https://github.com/RiskLabAI/RiskLabAI.jl/issues
- Documentation: https://github.com/RiskLabAI/RiskLabAI.jl#readme

The rights holder has confirmed permission to publish and license every file
admitted to the clean public package. This confirmation does not clear
artifact-inspection or human-authorization gates.

See `PACKAGE_IDENTITY.json` for the fail-closed machine-readable state.

The complete Julia contract is recorded in `Project.toml`, `PUBLIC_API.json`,
`TEST_INVENTORY.json`, and `PACKAGE_FILES.json`. Future package-tree checks are
specified in `PACKAGE_INSPECTION.md`; that document grants no registry or
release authority.

The 57-concept contract is documented in `docs/causal_factor_analysis.md`, and
runtime details are in `docs/compatibility.md`. A small deterministic example
is in `examples/causal_factor_analysis_quickstart.jl`, and the blocked
next-major summary is in `RELEASE_NOTES_NEXT_MAJOR.md`.
