# Julia 1.1.0 package inspection specification

This specification governs local-only inspection of the RiskLabAI 1.1.0
candidate created from the approved 1.0.0 merged source and the governed
additive delta. It does not authorize version control, registration,
installation into a user environment, upload, publication, tagging, or release.

## Inputs and membership

The inspector must bind the final source bytes and compare the package tree
with `PACKAGE_FILES.json`, `PUBLIC_API.json`, `TEST_INVENTORY.json`, and
`Project.toml`. Reject absolute or traversing paths, case-fold or Unicode
normalization collisions, duplicate paths, links, reparse points, special
files, caches, compiled output, manifests, credentials, or any undeclared
member.

The inspected registry tree must contain exactly the source members declared in
`PACKAGE_FILES.json`. Already-public repository-only paths remain preserved,
but they must not be imported or referenced by the active `src`, `ext`, or test
load graph and do not enlarge the public runtime API.

The package name and existing UUID must be unchanged. The canonical
BSD-3-Clause license, maintainer and contact, repository, issue,
documentation, and version records must agree with the reviewed metadata
contract. `PACKAGE_IDENTITY.json` must remain absent from the registry tree.

## Dependency and extension behavior

`Project.toml` must expose only the approved required dependencies and standard
libraries. QuadGK must be a direct bounded dependency. Lux, Optimisers, and
Zygote must remain weak dependencies loaded only through
`RiskLabAIDeepBSDEExt`. TimeSeries and every undeclared dependency must be
absent. All dependencies and weak dependencies must have bounded compatibility
entries.

## Runtime checks

In fresh Julia 1.10.12 and 1.12.7 environments, verify the exact package origin,
version 1.1.0 and the existing UUID, every root and module export in
`PUBLIC_API.json`, and the complete 87-name causal namespace. Require the
released 57-name causal set and the 30 additions to match the parity contract.
Run the exact package-scoped test inventory at the approved dependency floors
and current versions. In every temporary test environment, explicitly add all
eight required external dependencies before running tests. Run base lanes with
only the `Test` target and verify the extension-absence/fallback contract.
Repeat all lanes with Lux, Optimisers, and Zygote explicitly installed and
loaded, then verify the numerical Deep-BSDE contract.

## Final review

Scan every member name and every decoded public text member, including the 60
preserved repository-only members, against the release hygiene denylist.
Verify that no preserved path is reachable from an active `include`, `using`,
or `import`. Record the tree hash, member hashes, resolved dependency versions,
module origins, test results, and every deviation. Any deviation keeps the
package rejected. Human registration, release, publication, and
version-control authority remain separate decisions after inspection.
