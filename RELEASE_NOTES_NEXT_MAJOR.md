# RiskLabAI Julia 1.1.0 additive causal release notes — local candidate

Version: **1.1.0**. The package metadata is frozen at this version in the local
candidate. This does not authorize version control, registration, publication,
tagging, upload, or release.

The candidate starts from the approved RiskLabAI 1.0.0 merged source and
preserves its complete public library, existing package UUID, and released
57-name `RiskLabAI.CausalFactorAnalysis` contract. It adds 30 paper-derived
names, producing an 87-name causal module with a matching Python
implementation. No released causal name, signature, default, or behavior is
removed or silently changed.

The support policy remains Julia 1.10.12 LTS and Julia 1.12.7 stable. QuadGK
`>=2.11.3,<3` is a direct required dependency for the improper
selected-winner integral. Lux, Optimisers, and Zygote remain behind the
optional `deep_bsde` extension; the base library remains usable without that
stack.

The additive analytical families cover general-variance factor-mirage
coefficients, allocation-misspecification evidence, accepted-DAG factor roles,
deterministic structural-model evaluation, and family- and selection-level
false-discovery calculations for searched trials. The two false-discovery
estimands remain explicitly separate.

Every implemented method is linked to an authoritative public source and checked
against direct analytical examples, independent mathematical or graph oracles,
validation boundaries, stability cases, and—in addition—shared numerical
Python-Julia fixtures. Cross-language agreement is not used as the sole
correctness authority. Thirteen method units remain source-blocked and are
documented rather than guessed.

The exact 1.1.0 inventories are frozen against the approved merged 1.0.0
baseline commit `3a9ff26c7293616a112f0b36eef9beeea892e58c` and Git tree
`5498e2c4cd407d421dbb6b193b1f89223af63992`. Any package archive created for
inspection is local-only and is not a release artifact. Version control,
registration, publication, and release remain human-only and separately
authorized.
