using Test
using TOML
using Dates
using RiskLabAI

backstat_equal(left::Number, right::Number) = isnan(right) ? isnan(left) : isinf(right) ? left == right : isapprox(left, right; rtol = 1e-11, atol = 1e-13)
backstat_equal(left::AbstractArray, right::AbstractArray) = size(left) == size(right) && all(backstat_equal(a, b) for (a, b) in zip(left, right))
backstat_dates(offsets) = DateTime(2020, 1, 1) .+ Millisecond.(Int.(offsets))

@testset "Shared backtest moments, positions and holding periods" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "backtest_statistics_python_320.toml"))
    for case in fixture["sharpe"]
        @test backstat_equal(sharpe_ratio(Float64[case["values"]...]; risk_free_rate = case["risk_free"]), case["output"])
    end
    for case in fixture["hhi"]
        @test backstat_equal(calculate_hhi(Float64[case["values"]...]), case["output"])
    end
    for case in fixture["positions"]
        values, dates = Float64[case["values"]...], backstat_dates(case["offsets"])
        original = copy(values)
        @test bet_timing(dates, values) == dates[Int.(case["bets"])]
        result = calculate_holding_period(dates, values)
        expected = case["holding"]
        @test result.holding_periods.index == dates[Int[row["position"] for row in expected]]
        @test backstat_equal(result.holding_periods.dT, Float64[row["duration"] for row in expected])
        @test backstat_equal(result.holding_periods.w, Float64[row["weight"] for row in expected])
        @test backstat_equal(result.mean_holding_period, case["mean_holding_period"])
        @test values == original
    end
    dates = Date(2020, 1, 1) .+ Day.(0:3)
    dated = calculate_holding_period(dates, [0., 2., 1., 0.])
    timestamped = calculate_holding_period(DateTime.(dates), [0., 2., 1., 0.])
    @test dated.holding_periods.dT == timestamped.holding_periods.dT
    @test dated.mean_holding_period == timestamped.mean_holding_period
end

@testset "Shared calendar concentration and missing-value drawdowns" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "backtest_statistics_python_320.toml"))
    for case in fixture["concentration"]
        values, dates = Float64[case["values"]...], DateTime.(case["dates"])
        actual = calculate_hhi_concentration(dates, values)
        @test backstat_equal(collect(actual), Float64.(case["output"]))
        @test backstat_equal(collect(calculate_hhi_concentration(Date.(dates), values)), collect(actual))
    end
    for case in fixture["drawdown"]
        values, dates = Float64[case["values"]...], backstat_dates(case["offsets"])
        original = copy(values)
        result = compute_drawdowns_time_under_water(dates, values; dollars = case["dollars"])
        rows = case["rows"]
        @test result.start == dates[Int[row["position"] for row in rows]]
        @test backstat_equal(result.drawdown, Float64[row["drawdown"] for row in rows])
        @test backstat_equal(result.time_under_water, Float64[row["time_under_water"] for row in rows])
        @test isequal(values, original)
    end
    values = [1., .8, 1.1, .9]
    dates = Date(2020, 1, 1) .+ Day.(0:3)
    @test compute_drawdowns_time_under_water(dates, values).time_under_water == compute_drawdowns_time_under_water(DateTime.(dates), values).time_under_water
    @test all(isnan, calculate_hhi_concentration(DateTime[], Float64[]))
end

@testset "Shared conditional drawdown and studentized Sharpe bootstrap" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "backtest_statistics_python_320.toml"))
    for case in fixture["ced"]
        actual = conditional_expected_drawdown(Float64[case["values"]...], case["horizon"], case["alpha"])
        @test backstat_equal(actual, case["output"])
    end
    for case in fixture["difference"]
        a, b = Float64.(case["a"]), Float64.(case["b"])
        original = copy(a), copy(b)
        draws = case["method"] == "naive" ? nothing : [Int.(indices) for indices in case["bootstrap_indices"]]
        actual = sharpe_difference_test(a, b; method = case["method"], block_length = case["block_length"], n_boot = case["n_boot"], bandwidth = case["bandwidth"], bootstrap_indices = draws)
        for (key, value) in pairs(actual)
            @test backstat_equal(value, case["output"][string(key)])
        end
        @test (a, b) == original
    end
    a, b = [.01, -.02, .04, -.01], [.02, .01, -.03, .01]
    @test sharpe_difference_test(a, b; n_boot = 0).pvalue == 1.
    @test sharpe_difference_test(a, b; n_boot = 7, random_state = 11) == sharpe_difference_test(a, b; n_boot = 7, random_state = 11)
    @test_throws ArgumentError sharpe_difference_test(a, b; method = "invalid")
    @test_throws ArgumentError sharpe_difference_test(a, b; block_length = 0)
    @test_throws ArgumentError sharpe_difference_test(a, b; n_boot = -1)
    @test_throws DimensionMismatch sharpe_difference_test(a, b; n_boot = 2, bootstrap_indices = [[1, 2, 3, 4]])
    @test_throws ArgumentError sharpe_difference_test(a, b; n_boot = 1, bootstrap_indices = [[1, 2, 3, 5]])
    @test_throws ArgumentError sharpe_difference_test(a, b; block_length = 2, n_boot = 1, bootstrap_indices = [[1, 3, 2, 3]])
    @test_throws ArgumentError conditional_expected_drawdown(a, 3, 1.)
end

@testset "Shared Sharpe probability, influence and HAC inference" begin
    fixture = TOML.parsefile(joinpath(@__DIR__, "fixtures", "backtest_statistics_python_320.toml"))
    for case in fixture["psr"]
        actual = probabilistic_sharpe_ratio(case["observed"], case["benchmark"], case["count"]; skewness_of_returns = case["skew"], kurtosis_of_returns = case["kurtosis"])
        statistic = probabilistic_sharpe_ratio(case["observed"], case["benchmark"], case["count"]; skewness_of_returns = case["skew"], kurtosis_of_returns = case["kurtosis"], return_test_statistic = true)
        @test backstat_equal(actual, case["output"])
        @test backstat_equal(statistic, case["statistic"])
    end
    for case in fixture["benchmark"]
        @test backstat_equal(benchmark_sharpe_ratio(Float64[case["values"]...]), case["output"])
    end
    for case in fixture["influence"]
        @test backstat_equal(sharpe_ratio_influence_function(Float64[case["values"]...]), Float64.(case["output"]))
    end
    for case in fixture["lrv"]
        @test backstat_equal(newey_west_long_run_variance(Float64[case["values"]...], case["lag"]), case["output"])
    end
    for case in fixture["lag"]
        @test newey_west_automatic_lag(case["count"]) == case["output"]
    end
    for case in fixture["inference"]
        actual = lplz_sharpe_inference(Float64[case["values"]...]; confidence_level = case["confidence"], lag = case["supplied_lag"] == -1 ? nothing : case["supplied_lag"], null_sharpe_ratio = case["null"])
        for (key, value) in pairs(actual)
            expected = case["output"][string(key)]
            @test value isa Tuple ? backstat_equal(collect(value), expected) : backstat_equal(value, expected)
        end
    end
end
