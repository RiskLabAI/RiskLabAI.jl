"""
    RiskLabAI.Features

Feature-engineering submodule, mirroring the Python `RiskLabAI.features`
sub-package.

This PR wires the **entropy features** (Shannon, plug-in, Lempel–Ziv and
Kontoyiannis entropy). Microstructural features and structural-break tests are
wired in subsequent PRs; classifier-driven feature importance is deferred
pending a Julia ML-backend decision.
"""
module Features

# Entropy estimators (AFML Ch. 18).
include("EntropyFeatures.jl")

export
    # entropy features
    shannon_entropy,
    probability_mass_function,
    plug_in_entropy_estimator,
    lempel_ziv_entropy,
    longest_match_length,
    kontoyiannis_entropy

end # module Features
