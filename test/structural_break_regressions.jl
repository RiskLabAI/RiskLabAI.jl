using Test
using TOML
using Statistics
using RiskLabAI

break_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isinf(right) ? left == right : isapprox(left, right; rtol = 5e-8, atol = 5e-9)
break_equal(left::AbstractArray, right::AbstractArray) = size(left) == size(right) && all(break_equal(a, b) for (a, b) in zip(left, right))
break_matrix(rows, columns) = isempty(rows) ? zeros(0, columns) : permutedims(reduce(hcat, Float64.(row) for row in rows))

@testset "Shared compressed-difference ADF designs and QR-checked OLS" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "structural_breaks_python_320.toml"))
    for case in fixture["design"]
        values = Float64[case["values"]...]
        original = copy(values)
        actual = F.prepare_data(values, case["constant"], case["lags"])
        @test break_equal(actual.y, Float64[case["y"]...])
        @test break_equal(actual.x, break_matrix(case["x"], case["columns"]))
        @test actual.index == case["index"]
        @test isequal(values, original)
    end
    for case in fixture["ols"]
        x, y = break_matrix(case["x"], length(case["mean"])), Float64.(case["y"])
        for dependent in (y, reshape(y, :, 1))
            coefficients, covariance = F.compute_beta(dependent, x)
            @test break_equal(vec(coefficients), Float64.(case["mean"]))
            @test break_equal(covariance, break_matrix(case["covariance"], length(case["mean"])))
        end
    end
end

@testset "Shared expanding, forward and backward ADF windows" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "structural_breaks_python_320.toml"))
    for case in fixture["expanding"]
        actual = F.get_expanding_window_adf(Float64.(case["values"]), case["minimum"], case["constant"], case["lags"])
        @test actual.index == case["index"]
        @test break_equal(actual.statistics, Float64.(case["output"]))
    end
    for case in fixture["sequences"]
        values = Float64.(case["values"])
        original = copy(values)
        arguments = (values, case["minimum"], case["constant"], case["lags"])
        sadf, bsadf = F.get_sadf_sequence(arguments...), F.get_bsadf_sequence(arguments...)
        @test sadf.index == bsadf.index == case["index"]
        @test break_equal(sadf.values, Float64.(case["sadf"]))
        @test break_equal(bsadf.values, Float64.(case["bsadf"]))
        @test break_equal(F.get_gsadf_statistic(arguments...), case["gsadf"])
        @test isequal(values, original)
    end
    @test isempty(F.get_sadf_sequence([1., 2.], 8).values)
    @test isnan(F.get_gsadf_statistic(fill(1., 20), 8))
end

@testset "Shared PSY minimum windows and exhaustive episode dates" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "structural_breaks_python_320.toml"))
    for case in fixture["minimum"]
        @test F.psy_minimum_window(case["length"]) == case["output"]
    end
    for case in fixture["episodes"]
        expected = Tuple{Int,Int}[Tuple(Int.(row)) for row in case["output"]]
        @test F.get_bubble_episodes(Int.(case["index"]), Float64.(case["values"]), Float64.(case["critical"]); min_duration = case["minimum"]) == expected
        @test F.get_bubble_episodes(Int.(case["index"]), Float64.(case["values"]), 0.; min_duration = case["minimum"]) == expected
    end
end

@testset "Shared normal null paths and PSY finite-sample quantiles" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "structural_breaks_python_320.toml"))
    for case in fixture["critical"]
        draws = break_matrix(case["draws"], case["length"])
        original = copy(draws)
        actual = F.simulate_psy_critical_values(case["length"], case["minimum"]; constant = case["constant"], lags = case["lags"], n_simulations = case["simulations"], level = case["level"], normal_draws = draws)
        for key in ("sample_length", "min_sample_length", "level")
            @test getproperty(actual, Symbol(key)) == case["output"][key]
        end
        for key in ("sadf_global_cv", "gsadf_global_cv", "sadf_sequence_cv", "bsadf_sequence_cv")
            @test break_equal(getproperty(actual, Symbol(key)), case["output"][key])
        end
        @test draws == original
    end
    @test_throws DimensionMismatch F.simulate_psy_critical_values(17, 8; n_simulations = 2, normal_draws = zeros(2, 16))
    @test_throws ArgumentError F.simulate_psy_critical_values(17, 8; n_simulations = 0)
end

@testset "Shared wild-bootstrap paths, corrected p-values and decisions" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "structural_breaks_python_320.toml"))
    for case in fixture["robust"]
        values = Float64.(case["values"])
        signs = break_matrix(case["signs"], length(values)-1)
        original = deepcopy((values, signs))
        actual = F.volatility_robust_sadf(values; min_length = case["minimum"], n_bootstrap = case["bootstrap"], sign_draws = signs)
        @test break_equal(actual.sadf, case["output"]["sadf"])
        @test break_equal(actual.gsadf, case["output"]["gsadf"])
        for key in ("sadf_pvalue", "gsadf_pvalue", "reject_sadf", "reject_gsadf")
            @test getproperty(actual, Symbol(key)) == case["output"][key]
        end
        @test isequal((values, signs), original)
    end
    @test_throws ArgumentError F.volatility_robust_sadf(collect(1.:7.))
    @test_throws ArgumentError F.volatility_robust_sadf(collect(1.:17.); n_bootstrap = 2, sign_draws = zeros(2, 16))
end

@testset "Shared PELT supplier pruning and stable segment costs" begin
    F = RiskLabAI.Features
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "structural_breaks_python_320.toml"))
    disagreements = 0
    for case in fixture["pelt"]
        values = Float64.(case["values"])
        original = copy(values)
        actual = F.pelt_change_points(values; model = case["model"], penalty_multiplier = case["multiplier"], min_size = case["minimum"], jump = case["jump"])
        @test actual == case["output"]
        @test values == original
        if length(values) >= 2*case["minimum"]
            endpoints = [0; actual; length(values)]
            objective = 0.
            for index in 1:(length(endpoints)-1)
                segment = values[(endpoints[index]+1):endpoints[index+1]]
                cost = case["model"] == "normal" ? length(segment)*log(var(segment; corrected = false)+1e-6) : length(segment)*var(segment; corrected = false)
                objective += cost+case["multiplier"]*log(length(values))
            end
            @test isapprox(objective, case["reference_objective"]; rtol = 1e-8, atol = 1e-7)
            @test objective >= case["objective"]-1e-7*max(1., abs(case["objective"]))
            disagreements += actual != case["unpruned_output"]
        end
    end
    @test disagreements == 7
end
