function protocol_horizons()
    return (
        EventHorizon(observation_id = "o1", start = 0.0, stop = 2.0),
        EventHorizon(observation_id = "o2", start = 1.0, stop = 3.0),
        EventHorizon(observation_id = "o3", start = 4.0, stop = 5.0),
        EventHorizon(observation_id = "o4", start = 6.0, stop = 7.0),
        EventHorizon(observation_id = "o5", start = 8.0, stop = 9.0),
        EventHorizon(observation_id = "o6", start = 10.0, stop = 11.0),
    )
end

function protocol_fold(fold_id, path_id, train_ids, test_ids, refit_id)
    return FoldEvidence(
        fold_id = fold_id,
        path_id = path_id,
        train_ids = train_ids,
        test_ids = test_ids,
        fit_ids = train_ids,
        prediction_ids = test_ids,
        refit_id = refit_id,
        refit_components = (
            "VARIABLE_SELECTION",
            "CAUSAL_ESTIMATION",
            "PORTFOLIO_CONSTRUCTION",
        ),
    )
end

function protocol_validation(; method_labels = ("PURGED_CROSS_VALIDATION",))
    return ValidationEvidence(
        method_labels = method_labels,
        event_horizons = protocol_horizons(),
        folds = (
            protocol_fold("f1", "p1", ("o3", "o4", "o5", "o6"), ("o1", "o2"), "r1"),
            protocol_fold("f2", "p1", ("o1", "o2", "o5", "o6"), ("o3", "o4"), "r2"),
            protocol_fold("f3", "p1", ("o1", "o2", "o3", "o4"), ("o5", "o6"), "r3"),
        ),
        embargo = 1.0,
    )
end

function protocol_cpcv_validation()
    return ValidationEvidence(
        method_labels = ("COMBINATORIAL_PURGED_CROSS_VALIDATION",),
        event_horizons = protocol_horizons(),
        folds = (
            protocol_fold("p1-f1", "p1", ("o3", "o4", "o5", "o6"), ("o1", "o2"), "p1-r1"),
            protocol_fold("p1-f2", "p1", ("o1", "o2", "o5", "o6"), ("o3", "o4"), "p1-r2"),
            protocol_fold("p1-f3", "p1", ("o1", "o2", "o3", "o4"), ("o5", "o6"), "p1-r3"),
            protocol_fold("p2-f1", "p2", ("o4", "o5", "o6"), ("o1", "o3"), "p2-r1"),
            protocol_fold("p2-f2", "p2", ("o3", "o4", "o6"), ("o2", "o5"), "p2-r2"),
            protocol_fold("p2-f3", "p2", ("o1", "o2", "o3"), ("o4", "o6"), "p2-r3"),
        ),
        embargo = 1.0,
    )
end

function protocol_report(;
    variable_selection = nothing,
    causal_discovery = nothing,
    causal_adjustment_set = nothing,
    causal_explanatory_and_predictive_power = nothing,
    causal_portfolio_construction = nothing,
    backtest = nothing,
    multiple_testing_adjustments = nothing,
)
    trials = ("strategy-a", "strategy-b", "strategy-c")
    family = "preregistered-family"
    selection =
        variable_selection === nothing ?
        VariableSelectionStage(
            purpose = "RISK_PREMIA_HARVESTING",
            selected_variables = ("X", "Z", "I", "M", "C", "D"),
            method_labels = (
                "MUTUAL_INFORMATION",
                "SHAPLEY_VALUES",
                "MEAN_DECREASE_IMPURITY",
                "PERMUTATION_FEATURE_IMPORTANCE",
            ),
            overlapping_returns = true,
            strong_time_dependence = true,
            validation = protocol_validation(),
        ) : variable_selection
    discovery =
        causal_discovery === nothing ?
        CausalDiscoveryStage(
            method_labels = (
                "PC",
                "ECONOMIC_REASONING",
                "DOMAIN_EXPERTISE",
                "OBSERVED_OUTCOMES",
            ),
            graph_id = "resolved-dag",
            graph_nodes = ("X", "Y", "Z", "I", "M", "C", "D"),
            directed_edges = (
                ("Z", "X"),
                ("Z", "Y"),
                ("I", "X"),
                ("X", "M"),
                ("M", "Y"),
                ("X", "Y"),
                ("X", "C"),
                ("Y", "C"),
                ("X", "D"),
            ),
            graph_kind = "DAG",
            ambiguous_edges = (),
            assumptions = (
                "Recorded variables are sufficient for the target effect.",
                "Directions use temporal and economic restrictions.",
            ),
        ) : causal_discovery
    adjustment =
        causal_adjustment_set === nothing ?
        CausalAdjustmentSetStage(
            treatment = "X",
            outcome = "Y",
            method_label = "BACKDOOR_ADJUSTMENT",
            identified = true,
            admissible_adjustment_sets = (("Z",),),
            selected_adjustment_set = ("Z",),
            confounders = ("Z",),
            descendants = ("M", "C", "D"),
            mediators = ("M",),
            colliders = ("C",),
            instruments = ("I",),
            control_justifications = (("Z", "Common cause of factor and return."),),
            open_backdoor_paths = (),
        ) : causal_adjustment_set
    performance =
        causal_explanatory_and_predictive_power === nothing ?
        CausalExplanatoryAndPredictivePowerStage(
            task_types = ("RETURN_SIZE",),
            estimator_label = "fold-refit causal regression",
            explanatory_metric_labels = ("R_SQUARED",),
            predictive_metric_labels = ("MEAN_SQUARED_ERROR", "SPEARMAN_CORRELATION"),
            explanatory_evidence_id = "explanatory-evidence",
            predictive_evidence_id = "predictive-evidence",
            naive_benchmark_id = "training-mean-benchmark",
            validation = protocol_validation(),
        ) : causal_explanatory_and_predictive_power
    portfolio =
        causal_portfolio_construction === nothing ?
        CausalPortfolioConstructionStage(
            method_labels = (
                "POSITION_SIZING",
                "EXPOSURE_CONTROL",
                "ECONOMIC_RATIONALE",
                "FRAGILITY_STRESS_TEST",
                "TRANSACTION_COST_OPTIMIZATION",
                "TRANSFER_COEFFICIENT",
            ),
            causal_exposures = ("X",),
            controlled_unintended_exposures = ("C", "D"),
            cost_model_id = "cost-model-v1",
            constraint_set_id = "constraints-v1",
            economic_rationale = "Positions target the identified effect.",
            fragility_scenarios = ("weaken X-to-Y", "perturb Z-to-X"),
            transfer_coefficient = 0.82,
        ) : causal_portfolio_construction
    backtest_stage =
        backtest === nothing ?
        BacktestStage(
            method_labels = ("COMBINATORIAL_PURGED_CROSS_VALIDATION",),
            trial_family_id = family,
            declared_trial_ids = trials,
            validation = protocol_cpcv_validation(),
        ) : backtest
    corrections =
        multiple_testing_adjustments === nothing ?
        MultipleTestingAdjustmentsStage(
            method_labels = ("HOLM", "DEFLATED_SHARPE_RATIO"),
            trial_family_id = family,
            declared_trial_ids = trials,
            family_partitions = (
                ("primary", ("strategy-a", "strategy-b")),
                ("robustness", ("strategy-c",)),
            ),
            alpha = 0.05,
            backtests_independent = false,
            p_value_estimator = "dependence-adjusted-estimator",
            time_dependence_model = "purged temporal folds",
            p_value_inputs_assume_independence = false,
            sharpe_variance = 0.04,
            effective_trials = 2.0,
            sample_length = 252,
            skewness = 0.1,
            kurtosis = 3.2,
            selection_bias_evidence_id = "selection-bias-evidence",
        ) : multiple_testing_adjustments
    return CausalFactorProtocolReport(
        trial_family_id = family,
        declared_trial_ids = trials,
        variable_selection = selection,
        causal_discovery = discovery,
        causal_adjustment_set = adjustment,
        causal_explanatory_and_predictive_power = performance,
        causal_portfolio_construction = portfolio,
        backtest = backtest_stage,
        multiple_testing_adjustments = corrections,
    )
end

@testset "causal-factor protocol" begin
    report = protocol_report()
    @test validate_causal_factor_protocol(report) === report
    @test report.stage_names == (
        "Variable Selection",
        "Causal Discovery",
        "Causal Adjustment Set",
        "Causal Explanatory and Predictive Power",
        "Causal Portfolio Construction",
        "Backtest",
        "Multiple Testing Adjustments",
    )
    @test first.(report.stages) == report.stage_names
    @test report.source_oracle == "PROTOCOL-01"
    @test_throws Exception setfield!(report, :trial_family_id, "changed")

    overlap_mismatch = VariableSelectionStage(
        purpose = report.variable_selection.purpose,
        selected_variables = report.variable_selection.selected_variables,
        method_labels = report.variable_selection.method_labels,
        overlapping_returns = false,
        strong_time_dependence = true,
        validation = report.variable_selection.validation,
    )
    @test_throws ArgumentError validate_causal_factor_protocol(
        protocol_report(variable_selection = overlap_mismatch),
    )

    cyclic = CausalDiscoveryStage(
        method_labels = report.causal_discovery.method_labels,
        graph_id = "cyclic",
        graph_nodes = report.causal_discovery.graph_nodes,
        directed_edges = (report.causal_discovery.directed_edges..., ("Y", "X")),
        graph_kind = "DAG",
        ambiguous_edges = (),
        assumptions = report.causal_discovery.assumptions,
    )
    @test_throws ArgumentError validate_causal_factor_protocol(
        protocol_report(causal_discovery = cyclic),
    )

    invalid_adjustment = CausalAdjustmentSetStage(
        treatment = "X",
        outcome = "Y",
        method_label = "BACKDOOR_ADJUSTMENT",
        identified = true,
        admissible_adjustment_sets = (("C",),),
        selected_adjustment_set = ("C",),
        confounders = ("Z",),
        descendants = ("M", "C", "D"),
        mediators = ("M",),
        colliders = ("C",),
        instruments = ("I",),
        control_justifications = (("C", "Invalid collider control."),),
        open_backdoor_paths = (),
    )
    @test_throws ArgumentError validate_causal_factor_protocol(
        protocol_report(causal_adjustment_set = invalid_adjustment),
    )

    invalid_multiple_testing = MultipleTestingAdjustmentsStage(
        method_labels = ("HOLM", "DEFLATED_SHARPE_RATIO"),
        trial_family_id = report.trial_family_id,
        declared_trial_ids = report.declared_trial_ids,
        family_partitions = (("primary", report.declared_trial_ids),),
        alpha = 1.0,
        backtests_independent = true,
        p_value_estimator = "estimator",
        time_dependence_model = "model",
        p_value_inputs_assume_independence = true,
        sharpe_variance = 0.0,
        effective_trials = 3.0,
        sample_length = 1,
        skewness = 2.0,
        kurtosis = 1.0,
        selection_bias_evidence_id = nothing,
    )
    @test_throws ArgumentError validate_causal_factor_protocol(
        protocol_report(multiple_testing_adjustments = invalid_multiple_testing),
    )
end
