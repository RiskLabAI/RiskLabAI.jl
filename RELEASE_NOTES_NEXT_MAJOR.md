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

No source-conflicted causal result is admitted. The static Project metadata,
63-file source inventory, 12-file package-scoped test inventory, public API
inventory, documentation, examples, dependency extension, and intended package
file list are frozen. No package artifact has been created. Future artifact
inspection and separate human authorization for version-control, registration,
publication, and release remain blocked.
