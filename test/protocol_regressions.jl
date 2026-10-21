using Test
using TOML
using RiskLabAI
using RiskLabAI.CausalFactorAnalysis

function protocol_fixture_value(encoded)
    kind = encoded["kind"]
    kind == "none" && return nothing
    kind == "scalar" && return encoded["value"]
    kind == "float" && return encoded["value"] == "nan" ? NaN : encoded["value"] == "inf" ? Inf : -Inf
    if kind in ("tuple", "list")
        items = map(protocol_fixture_value, encoded["items"])
        return kind == "tuple" ? Tuple(items) : items
    end
    kind == "record" || error("Unexpected protocol fixture value")
    types = (EventHorizon, FoldEvidence, ValidationEvidence, VariableSelectionStage,
        CausalDiscoveryStage, CausalAdjustmentSetStage, CausalExplanatoryAndPredictivePowerStage,
        CausalPortfolioConstructionStage, BacktestStage, MultipleTestingAdjustmentsStage, CausalFactorProtocolReport)
    index = findfirst(type -> string(nameof(type)) == encoded["type"], types)
    isnothing(index) && error("Unexpected protocol fixture record type")
    return types[index](; ((key == "end" ? :stop : Symbol(key)) => protocol_fixture_value(value) for (key, value) in encoded["fields"])...)
end

function protocol_fixture_change(value, path, replacement)
    isempty(path) && return replacement
    key, rest = first(path), path[2:end]
    if value isa Tuple || value isa AbstractVector
        index = parse(Int, key) + 1
        items = [position == index ? protocol_fixture_change(item, rest, replacement) : item for (position, item) in enumerate(value)]
        return value isa Tuple ? Tuple(items) : items
    end
    field = key == "end" ? :stop : Symbol(key)
    index = findfirst(==(field), fieldnames(typeof(value)))
    isnothing(index) && error("Unexpected protocol fixture path")
    updated = protocol_fixture_change(getfield(value, field), rest, replacement)
    declared = fieldtype(typeof(value), index)
    bool_as_number = updated isa Bool && declared != Bool && declared != Union{Nothing,Bool}
    updated isa declared && !bool_as_number || throw(ArgumentError("Protocol fixture value is rejected by the released Julia record type"))
    fields = value isa CausalFactorProtocolReport ? fieldnames(typeof(value))[1:9] : fieldnames(typeof(value))
    return typeof(value)(; (name => (name == field ? updated : getfield(value, name)) for name in fields)...)
end

function protocol_case_result(base, changes)
    report = base
    try
        for change in changes
            report = protocol_fixture_change(report, change["path"], protocol_fixture_value(change["value"]))
        end
        output = protocol_validation_result(report)
        return (report = report, result = output.result, error = output.error, constructor_rejection = false)
    catch error
        return (report = report, result = nothing, error = error, constructor_rejection = true)
    end
end

function protocol_validation_result(report)
    try
        return (result = validate_causal_factor_protocol(report), error = nothing)
    catch error
        return (result = nothing, error = error)
    end
end

function protocol_diagnostics(error)
    error isa ArgumentError || return String[]
    return [String(item.match) for item in eachmatch(r"(?:Protocol|Stage [1-7]): .*?(?= (?:Protocol|Stage [1-7]): |$)", error.msg)]
end

@testset "Seven-stage protocol records against public Python 3.2.0" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "protocol_python_320.toml"))
    bases = Dict(name => protocol_fixture_value(value["value"]) for (name, value) in fixture["bases"])
    for case in fixture["cases"]
        output = protocol_case_result(bases[case["base"]], case["changes"])
        @testset "$(case["name"])" begin
            @test isnothing(output.error) == case["accepted"]
            if case["accepted"]
                @test output.result === output.report
                @test isempty(case["errors"])
            elseif output.constructor_rejection
                @test !case["accepted"]
            else
                @test output.error isa ArgumentError
                messages = protocol_diagnostics(output.error)
                @test !isempty(messages)
                @test all(startswith(message, "Protocol:") || startswith(message, "Stage ") for message in messages)
                @test length(messages) == length(unique(messages))
            end
        end
    end
end

@testset "Protocol tuple snapshots, defaults and fixed source properties" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "protocol_python_320.toml"))
    report = protocol_fixture_value(fixture["bases"]["backdoor"]["value"])
    @test report.stage_names == ("Variable Selection", "Causal Discovery", "Causal Adjustment Set", "Causal Explanatory and Predictive Power", "Causal Portfolio Construction", "Backtest", "Multiple Testing Adjustments")
    @test first.(report.stages) == report.stage_names
    @test last.(report.stages) == Tuple(getfield(report, field) for field in fieldnames(typeof(report))[3:9])
    @test report.source_oracle == "PROTOCOL-01"
    @test report.source_locator == "PDF pages 18-22 (journal pages 27-31), Exhibit 10"
    @test all(hasproperty(report, name) for name in (:stages, :stage_names, :source_oracle, :source_locator))
    @test_throws ErrorException setproperty!(report, :trial_family_id, "changed")
    @test_throws ErrorException setproperty!(report, :source_oracle, "changed")
    train, test = ["a", "b"], ["c"]
    components = ["VARIABLE_SELECTION"]
    fold = FoldEvidence("fold", "path", train, test, train, test, "refit", components)
    train[1], test[1], components[1] = "changed", "changed", "changed"
    @test fold.train_ids == fold.fit_ids == ("a", "b")
    @test fold.test_ids == fold.prediction_ids == ("c",)
    @test fold.refit_components == ("VARIABLE_SELECTION",)
    @test_throws ErrorException setproperty!(fold, :fold_id, "changed")
    edges = [["x", "y"]]
    nodes = ["x", "y"]
    discovery = CausalDiscoveryStage(["PC"], "graph", nodes, edges, "DAG", [], ["assumption"])
    edges[1][1], nodes[1] = "changed", "changed"
    @test discovery.directed_edges == (("x", "y"),)
    @test discovery.graph_nodes == ("x", "y")
    trials = ["a", "b"]
    partitions = [["group", trials]]
    adjustment = MultipleTestingAdjustmentsStage(["HOLM"], "family", trials, partitions, 0.05, false)
    trials[1] = "changed"
    partitions[1][1] = "changed"
    @test adjustment.declared_trial_ids == ("a", "b")
    @test adjustment.family_partitions == (("group", ("a", "b")),)
    @test isnothing(adjustment.sample_length)
    @test isnothing(adjustment.p_value_estimator)
    horizon = EventHorizon("a", 0, 1)
    @test getproperty(horizon, :stop) == 1
    @test EventHorizon(; observation_id = "a", start = 0, stop = 1).stop == 1
    horizons, folds, methods = [horizon], [fold], ["RESAMPLING"]
    evidence = ValidationEvidence(methods, horizons, folds)
    empty!(horizons)
    empty!(folds)
    empty!(methods)
    @test evidence.event_horizons == (horizon,)
    @test evidence.folds == (fold,)
    @test evidence.method_labels == ("RESAMPLING",)
    @test evidence.embargo == 0.0
    for (_, stage) in report.stages
        @test !ismutable(stage)
    end
    touching = Dict("a" => (0.0, 1.0), "b" => (1.0, 2.0))
    @test !RiskLabAI.CausalFactorAnalysis._has_any_overlap(touching)
    @test RiskLabAI.CausalFactorAnalysis._has_any_overlap(Dict("a" => (0.0, nextfloat(1.0)), "b" => (1.0, 2.0)))
end
