# Future Julia package inspection specification

This specification applies only after a separate human authorization creates a
temporary RiskLabAI 1.0.0 package tree. It does not authorize registration,
installation, upload, publication, tagging, or release.

## Inputs and membership

The inspector must bind the final source bytes and compare the package tree
with `PACKAGE_FILES.json`, `PUBLIC_API.json`, `TEST_INVENTORY.json`, and
`Project.toml`. Reject absolute or traversing paths, case-fold or Unicode
normalization collisions, duplicate paths, links, reparse points, special
files, caches, compiled output, manifests, credentials, or any undeclared
member.

The inspected tree must contain exactly the 88 declared RiskLabAI 1.0.0 release
members, including the governed `.github/workflows/CI.yml`. Its package name
and existing UUID must be unchanged. The canonical BSD-3-Clause license,
maintainer and contact, repository, issue, documentation, and version records
must agree with the reviewed metadata contract.

## Dependency and extension behavior

`Project.toml` must expose only the approved required dependencies and standard
libraries. Lux, Optimisers, and Zygote must remain weak dependencies loaded
only through `RiskLabAIDeepBSDEExt`. TimeSeries and any undeclared dependency
must be absent. All dependencies and weak dependencies must have bounded
compatibility entries.

## Runtime checks

In fresh Julia 1.10.12 and 1.12.7 environments, verify the exact package origin,
`Base.pkgversion(RiskLabAI) == v"1.0.0"`, the 148 root exports, all module
exports in `PUBLIC_API.json`, and the complete 57-feature causal namespace.
Run the exact package-scoped test inventory at the approved dependency floors
and current versions. Run base lanes with only the `Test` target and verify the
four extension-absence/fallback assertions. Repeat all lanes with Lux,
Optimisers, and Zygote explicitly installed and loaded, then verify the four
numerical Deep-BSDE assertions.

## Final review

Scan every member name and every decoded public text member against the release
hygiene denylist. Record the tree hash, member hashes, resolved dependency
versions, module origins, test results, and every deviation. Any deviation
keeps the package rejected. Human registration, release, publication, and
version-control authority remain separate decisions after inspection.
