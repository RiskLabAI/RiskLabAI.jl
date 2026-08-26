# RiskLabAI Julia next-major release notes — blocked draft

Target version: **1.0.0**. The owner approved this exact version, but assigning
it does not authorize registration, publication, or release.

The next major Julia release preserves the prior public repository source
modules and adds the verified 57-concept `RiskLabAI.CausalFactorAnalysis`
module under the existing RiskLabAI package UUID.

The supported runtime policy is Julia 1.10 LTS and Julia 1.12 stable at their
tested latest patches. TimeSeries is removed as an unused requirement. Lux,
Optimisers, and Zygote move behind the optional `deep_bsde` extension; the
base library remains usable without that stack. The deep solver uses a
consistent Float64 numerical path.

The test target now keeps weak dependencies out of the base environment. Base
lanes verify the explicit fallback, while separate Julia 1.10.12 and 1.12.7
extension lanes install the weak dependencies and run the numerical solver
tests. Every minimum and current lane explicitly adds the seven required base
dependencies to its temporary test environment; extension lanes additionally
add Lux, Optimisers, and Zygote. The governed workflow also pins JuliaFormatter
2.9.0 for the clean causal source, causal tests, and test entry point.

No source-conflicted causal result is admitted. The static Project metadata,
63-file source inventory, 12-file package-scoped test inventory, public API
inventory, documentation, examples, dependency extension, governed CI
workflow, and 88-file intended release-source list are frozen. The registry
tree also preserves 60 already-public repository-only files; they are not
loaded by the active package and do not alter its runtime API. No package
artifact has been created. Future artifact inspection and separate human
authorization for version-control, registration, publication, and release
remain blocked.
