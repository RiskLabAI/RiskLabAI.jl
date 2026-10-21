using Test
using TOML
using Statistics
using RiskLabAI

hpo_matrix(rows) = reduce(vcat, [permutedims(Vector{Float64}(row)) for row in rows])
hpo_close(left, right) = isapprox(left, right; rtol = 1e-10, atol = 2e-13, nans = true)

mutable struct HpoDiagnosticEstimator
    params::Dict
    threshold::Float64
    fits::Int
end

function hpo_callbacks(records)
    calls = Ref(0)
    factory(params) = HpoDiagnosticEstimator(deepcopy(params), NaN, 0)
    fit = function (model, x, y)
        calls[] += 1
        record = records[calls[]]
        @test model.fits == 0
        @test Int.(x[:, 1]) == record["train"]
        model.fits += 1
        model.threshold = model.params["cut"]+.15*mean(x[:, 2])+.05*mean(y)
        @test hpo_close(model.threshold, record["threshold"])
        return model
    end
    predict = function (model, x)
        @test model.fits == 1
        @test Int.(x[:, 1]) == records[calls[]]["test"]
        return Int.(model.params["slope"].*x[:, 2] .> model.threshold)
    end
    score = (model, x, y) -> mean(predict(model, x) .== y)+.001*mean(x[:, 2])
    return factory, fit, predict, score, calls
end

@testset "HPO: public fold skips and caller-estimator scoring" begin
    V = RiskLabAI.Validation
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "hpo_python_320.toml"))
    for case in fixtures["objective"]
        x, y = hpo_matrix(case["x"]), Vector{Int}(case["y"])
        cv = V.PurgedKFoldCV(case["splits"], case["starts"], case["ends"]; embargo = case["embargo"])
        factory, fit, predict, score, calls = hpo_callbacks(case["records"])
        actual = V._purged_cv_score(factory, case["params"], x, y, cv;
            fit_estimator = fit, predict = predict, scoring = case["custom"] ? score : nothing)
        @test hpo_close(actual, case["score"])
        @test calls[] == length(case["records"])
    end
end

@testset "HPO: recorded TPE trials and sampler feedback" begin
    V = RiskLabAI.Validation
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "hpo_python_320.toml"))
    for case in fixtures["study"]
        proposals = case["proposed"]
        records = reduce(vcat, [proposal["records"] for proposal in proposals])
        factory, fit, predict, score, calls = hpo_callbacks(records)
        asked, told = Ref(0), Ref(0)
        ask = function (number, history, rng)
            asked[] += 1
            @test asked[] == number && length(history) == number-1
            for (previous, proposal) in zip(history, proposals)
                @test previous.params == proposal["parameters"]
                @test hpo_close(previous.score, proposal["score"])
                @test previous.state == :complete
            end
            return proposals[number]
        end
        tell = function (trial, value)
            told[] += 1
            @test trial === proposals[told[]]
            @test hpo_close(value, trial["score"])
        end
        actual = V.leakage_aware_hpo(trial -> trial["configuration"], factory,
            hpo_matrix(case["x"]), Vector{Int}(case["y"]);
            event_starts = case["starts"], event_ends = case["ends"],
            ask_trial = ask, fit_estimator = fit, predict = predict,
            scoring = case["custom"] ? score : nothing,
            trial_parameters = trial -> trial["parameters"], tell_trial! = tell,
            n_trials = length(proposals), n_splits = case["splits"], embargo = case["embargo"])
        expected = case["output"]
        @test actual.best_params == expected["best_params"]
        @test !haskey(actual.best_params, "fixed")
        @test hpo_close(actual.best_score, expected["best_score"])
        @test hpo_close(actual.trial_scores, Vector{Float64}(expected["trial_scores"]))
        @test hpo_close(actual.mean_trial_score, expected["mean_trial_score"])
        @test hpo_close(actual.std_trial_score, expected["std_trial_score"])
        @test actual.n_trials == length(proposals) == asked[] == told[]
        @test calls[] == length(records)
    end
end

@testset "HPO: failed, infinite and singleton trial summaries" begin
    V = RiskLabAI.Validation
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "hpo_python_320.toml"))
    for case in fixtures["boundary"]
        history = [(number = index, params = proposal["parameters"], score = proposal["score"], state = isnan(proposal["score"]) ? :failed : :complete) for (index, proposal) in enumerate(case["proposals"])]
        if !isempty(case["error"])
            @test_throws ArgumentError V._hpo_trial_summary(history, length(history))
        else
            actual = V._hpo_trial_summary(history, length(history))
            @test actual.best_params == case["best_params"]
            @test hpo_close(actual.best_score, case["best_score"])
            @test hpo_close(actual.trial_scores, Vector{Float64}(case["scores"]))
            @test hpo_close(actual.mean_trial_score, case["mean"])
            @test hpo_close(actual.std_trial_score, case["deviation"])
        end
    end
    x, y = zeros(40, 2), repeat([0, 1], 20)
    cv = V.PurgedKFoldCV(4, collect(1:40), collect(1:40))
    @test_throws DimensionMismatch V._purged_cv_score(identity, Dict(), x, y[1:end-1], cv; fit_estimator = identity, predict = identity)
    @test_throws ArgumentError V._purged_cv_score(identity, Dict(), x, y, cv; fit_estimator = identity)
    @test_throws ErrorException V._purged_cv_score(identity, Dict(), x, y, cv; fit_estimator = (model, x, y) -> error("fit failure"), predict = identity)
end

@testset "HPO: DSR gate with independently checked biased moments" begin
    V = RiskLabAI.Validation
    fixtures = TOML.parsefile(joinpath(@__DIR__, "fixtures", "hpo_python_320.toml"))
    for case in fixtures["gate"]
        actual = V.deflated_sharpe_gate(case["returns"], case["trials"], case["dispersion"]; threshold = case["threshold"])
        for name in (:observed_sharpe, :benchmark_sharpe, :deflated_sharpe)
            @test hpo_close(getproperty(actual, name), case["output"][String(name)])
        end
        @test actual.n_trials == case["output"]["n_trials"]
        @test actual.passes == case["output"]["passes"]
        equal_threshold = V.deflated_sharpe_gate(case["returns"], case["trials"], case["dispersion"]; threshold = actual.deflated_sharpe)
        @test !equal_threshold.passes
    end
end
