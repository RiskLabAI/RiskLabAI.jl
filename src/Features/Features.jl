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

# Microstructural features: Corwin–Schultz spread, Bekker–Parkinson vol (AFML Ch. 19).
include("MicrostructuralFeatures.jl")

# Structural-break tests: ADF design + (Backward) Supremum ADF (AFML Ch. 17).
include("StructuralBreaks.jl")

# Feature importance — backend-independent pieces (PCA orthogonalisation, weighted-τ).
# Classifier-driven importances (MDI/MDA/SFI) follow with the DecisionTree.jl backend.
include("FeatureImportance.jl")

export
    # entropy features
    shannon_entropy,
    probability_mass_function,
    plug_in_entropy_estimator,
    lempel_ziv_entropy,
    longest_match_length,
    kontoyiannis_entropy,
    miller_madow_entropy,
    grassberger_entropy,
    nsb_entropy,
    # microstructural features
    beta_estimates,
    gamma_estimates,
    alpha_estimates,
    corwin_schultz_estimator,
    sigma_estimates,
    bekker_parkinson_volatility_estimates,
    edge_estimator,
    # structural breaks
    lag_dataframe,
    prepare_data,
    compute_beta,
    get_expanding_window_adf,
    get_bsadf_statistic,
    psy_minimum_window,
    get_sadf_sequence,
    get_bsadf_sequence,
    get_gsadf_statistic,
    get_bubble_episodes,
    simulate_psy_critical_values,
    # feature importance (backend-independent)
    orthogonal_features,
    calculate_weighted_tau,
    # feature importance (DecisionTree.jl backend)
    feature_importance_mdi,
    feature_importance_mda,
    feature_importance_sfi,
    get_test_dataset,
    mdi_plus_importance,
    conditional_predictive_impact

end # module Features
